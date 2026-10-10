import os
os.environ["OMP_NUM_THREADS"] = "8"

import torch
import torch.nn as nn
from copy import copy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import YOLOv10DetectionModel
from ultralytics.utils.loss import v10DetectLoss

from train_D_midfusion_concat import (
    RGBTYOLOv10MidConcat,
    RGBTMidConcatValidator,
    load_midfusion_pretrained,
    DATA_YAML,
    BASE_MODEL_YAML,
    PRETRAINED,
)


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()

        hidden = max(channels // reduction, 8)
        padding = spatial_kernel // 2

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)

        channel_attn = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        x = x * channel_attn

        # Spatial attention
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)

        spatial_attn = self.sigmoid(
            self.spatial(torch.cat([avg_map, max_map], dim=1))
        )
        x = x * spatial_attn

        return x


class CMAFFStyleFusion(nn.Module):
    """
    CMAFF-style common/differential RGB-IR fusion.

    Input:
        rgb: [B, C, H, W]
        ir:  [B, C, H, W]

    Common branch:
        common = (rgb + ir) / 2

    Differential branch:
        diff = |rgb - ir|

    Output:
        fuse: [B, C, H, W]
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()

        hidden = max(channels // reduction, 8)

        # Common-modality attention
        self.common_attn = ChannelSpatialAttention(
            channels,
            reduction=reduction,
            spatial_kernel=spatial_kernel,
        )

        # Differential-modality attention
        self.diff_attn = ChannelSpatialAttention(
            channels,
            reduction=reduction,
            spatial_kernel=spatial_kernel,
        )

        # Modality selection logits guided by differential information
        self.rgb_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        self.ir_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        # Final projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb, ir):
        assert rgb.shape == ir.shape, (
            f"RGB and IR feature shapes must match, got {rgb.shape} and {ir.shape}"
        )

        # Shared evidence: both modalities agree
        common = 0.5 * (rgb + ir)
        common_feat = self.common_attn(common)

        # Difference evidence: modalities disagree
        diff = torch.abs(rgb - ir)
        diff_context = self.diff_attn(diff)

        # Use differential context to decide RGB/IR reliability
        rgb_logit = self.rgb_gate(rgb + diff_context)
        ir_logit = self.ir_gate(ir + diff_context)

        logits = torch.stack([rgb_logit, ir_logit], dim=1)  # [B, 2, C, H, W]
        weights = torch.softmax(logits, dim=1)

        differential_feat = weights[:, 0] * rgb + weights[:, 1] * ir

        # More stable than directly using common_feat + differential_feat
        fuse = self.out_conv(0.5 * (common_feat + differential_feat))

        return fuse


class RGBTYOLOv10MidFusionCMAFF(RGBTYOLOv10MidConcat):
    """
    Experiment E4: YOLOv10s-MidFusion-CMAFF-style

    Difference from previous E variants:
        CBAM-style: channel + spatial modality softmax
        MAA:        channel-only modality attention
        CA:         concat + coordinate attention
        CMAFF:      common/differential cross-modality attention
    """

    def __init__(self, cfg=BASE_MODEL_YAML, ch=4, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # Remove Experiment D concat + 1x1 fusion layers
        for name in ("fuse_p3", "fuse_p4", "fuse_p5"):
            if hasattr(self, name):
                delattr(self, name)

        # YOLOv10s P3/P4/P5 channels
        c3, c4, c5 = 128, 256, 512

        self.attn_fuse_p3 = CMAFFStyleFusion(c3, reduction=16, spatial_kernel=7)
        self.attn_fuse_p4 = CMAFFStyleFusion(c4, reduction=16, spatial_kernel=7)
        self.attn_fuse_p5 = CMAFFStyleFusion(c5, reduction=16, spatial_kernel=7)

        self.fusion_type = "CMAFF-style"

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        # During stride initialization, YOLOv10 may use 3-channel dummy input
        if x.shape[1] == 3:
            return YOLOv10DetectionModel._predict_once(
                self,
                x,
                profile=profile,
                visualize=visualize,
                embed=embed,
            )

        assert x.shape[1] == 4, (
            f"Expected 4-channel RGB-IR input, but got {x.shape[1]} channels."
        )

        rgb = x[:, 0:3, :, :]
        ir = x[:, 3:4, :, :]

        # RGB backbone
        rgb_p3, rgb_p4, rgb_p5, y = self._forward_backbone(
            rgb,
            self.model[: self.backbone_end + 1],
        )

        # IR backbone
        ir_p3, ir_p4, ir_p5, _ = self._forward_backbone(
            ir,
            self.ir_backbone,
        )

        assert rgb_p3.shape == ir_p3.shape, (
            f"P3 shape mismatch: {rgb_p3.shape} vs {ir_p3.shape}"
        )
        assert rgb_p4.shape == ir_p4.shape, (
            f"P4 shape mismatch: {rgb_p4.shape} vs {ir_p4.shape}"
        )
        assert rgb_p5.shape == ir_p5.shape, (
            f"P5 shape mismatch: {rgb_p5.shape} vs {ir_p5.shape}"
        )

        # CMAFF-style fusion
        fuse_p3 = self.attn_fuse_p3(rgb_p3, ir_p3)
        fuse_p4 = self.attn_fuse_p4(rgb_p4, ir_p4)
        fuse_p5 = self.attn_fuse_p5(rgb_p5, ir_p5)

        y[self.p3_index] = fuse_p3
        y[self.p4_index] = fuse_p4
        y[self.p5_index] = fuse_p5

        # Continue original YOLOv10 neck + head
        x = fuse_p5

        for m in self.model[self.backbone_end + 1:]:
            if m.f != -1:
                x_in = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f
                ]
            else:
                x_in = x

            x = m(x_in)
            y.append(x if m.i in self.save else None)

        return x

    def init_criterion(self):
        return v10DetectLoss(self)


class RGBTMidFusionCMAFFTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        nc = self.data["nc"]

        model = RGBTYOLOv10MidFusionCMAFF(
            cfg=BASE_MODEL_YAML,
            ch=4,
            nc=nc,
            verbose=verbose,
        )

        load_midfusion_pretrained(model, PRETRAINED)

        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"

        return RGBTMidConcatValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


def main():
    args = dict(
        model=BASE_MODEL_YAML,
        data=DATA_YAML,

        epochs=100,
        imgsz=640,
        batch=16,

        device=0,
        workers=8,

        amp=False,
        cache=False,

        project="runs/frgbt",
        name="E_midfusion_CMAFF_yolov10s",

        seed=0,
        deterministic=True,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )

    trainer = RGBTMidFusionCMAFFTrainer(overrides=args)
    trainer.train()


if __name__ == "__main__":
    main()
