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


class ModalityAttentionFusion(nn.Module):
    """
    True RGB-IR attention fusion without channel concat.

    Input:
        rgb: [B, C, H, W]
        ir:  [B, C, H, W]

    Output:
        fuse: [B, C, H, W]

    Fusion:
        fuse = W_rgb * rgb + W_ir * ir

    where W_rgb and W_ir are learned attention weights and:
        W_rgb + W_ir = 1
    along the modality dimension.

    This module uses both channel-level and spatial-level evidence to decide
    whether each feature position should rely more on RGB or IR.
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        padding = spatial_kernel // 2

        # Channel modality attention for RGB and IR.
        self.rgb_channel = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.ir_channel = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        # Spatial modality attention for RGB and IR.
        self.rgb_spatial = nn.Conv2d(
            1,
            1,
            kernel_size=spatial_kernel,
            padding=padding,
            bias=False,
        )
        self.ir_spatial = nn.Conv2d(
            1,
            1,
            kernel_size=spatial_kernel,
            padding=padding,
            bias=False,
        )

    def forward(self, rgb, ir):
        assert rgb.shape == ir.shape, f"RGB and IR feature shapes must match, got {rgb.shape} and {ir.shape}"

        # Channel evidence: [B, C, 1, 1]
        rgb_c = torch.mean(rgb, dim=(2, 3), keepdim=True) + torch.amax(rgb, dim=(2, 3), keepdim=True)
        ir_c = torch.mean(ir, dim=(2, 3), keepdim=True) + torch.amax(ir, dim=(2, 3), keepdim=True)

        rgb_c_logit = self.rgb_channel(rgb_c)
        ir_c_logit = self.ir_channel(ir_c)

        # Spatial evidence: [B, 1, H, W]
        rgb_s = torch.mean(rgb, dim=1, keepdim=True)
        ir_s = torch.mean(ir, dim=1, keepdim=True)

        rgb_s_logit = self.rgb_spatial(rgb_s)
        ir_s_logit = self.ir_spatial(ir_s)

        # Broadcast channel and spatial logits to [B, C, H, W].
        rgb_logit = rgb_c_logit + rgb_s_logit
        ir_logit = ir_c_logit + ir_s_logit

        # Softmax along modality dimension:
        # weights[:, 0] is RGB weight, weights[:, 1] is IR weight.
        logits = torch.stack([rgb_logit, ir_logit], dim=1)  # [B, 2, C, H, W]
        weights = torch.softmax(logits, dim=1)

        fuse = weights[:, 0] * rgb + weights[:, 1] * ir
        return fuse


class RGBTYOLOv10MidAttentionFusion(RGBTYOLOv10MidConcat):
    """
    Experiment E: YOLOv10s-MidFusion-AttentionFusion

    Difference from Experiment D:
        D: RGB_Pi + IR_Pi -> Concat -> 1x1 Conv -> Fuse_Pi
        E: RGB_Pi + IR_Pi -> ModalityAttentionFusion -> Fuse_Pi

    No channel concat is used in the fusion path.
    """

    def __init__(
        self,
        cfg=BASE_MODEL_YAML,
        ch=4,
        nc=None,
        verbose=True,
    ):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # Experiment E does not use the Concat + 1x1 Conv fusion layers from Experiment D.
        # Delete them so they will not become unused trainable parameters.
        for name in ("fuse_p3", "fuse_p4", "fuse_p5"):
            if hasattr(self, name):
                delattr(self, name)

        # YOLOv10s P3/P4/P5 channels.
        c3, c4, c5 = 128, 256, 512

        self.attn_fuse_p3 = ModalityAttentionFusion(c3, reduction=16, spatial_kernel=7)
        self.attn_fuse_p4 = ModalityAttentionFusion(c4, reduction=16, spatial_kernel=7)
        self.attn_fuse_p5 = ModalityAttentionFusion(c5, reduction=16, spatial_kernel=7)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        YOLOv10 forward pass with true RGB-IR attention fusion.

        If x has 3 channels, run the original YOLOv10 forward.
        If x has 4 channels, split it into RGB and IR and use attention fusion.
        """

        if not hasattr(self, "_debug_printed"):
            print("[Debug] input shape:", x.shape)
            self._debug_printed = True

        # During stride initialization or normal RGB test, YOLOv10 may pass 3-channel input.
        if x.shape[1] == 3:
            return YOLOv10DetectionModel._predict_once(
                self,
                x,
                profile=profile,
                visualize=visualize,
                embed=embed,
            )

        assert x.shape[1] == 4, f"RGBT MidFusion expects 4-channel input, but got {x.shape[1]} channels."

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

        # RGB and IR features must be aligned at each scale.
        assert rgb_p3.shape == ir_p3.shape, f"P3 feature mismatch: {rgb_p3.shape} vs {ir_p3.shape}"
        assert rgb_p4.shape == ir_p4.shape, f"P4 feature mismatch: {rgb_p4.shape} vs {ir_p4.shape}"
        assert rgb_p5.shape == ir_p5.shape, f"P5 feature mismatch: {rgb_p5.shape} vs {ir_p5.shape}"

        # True attention fusion, without Concat.
        fuse_p3 = self.attn_fuse_p3(rgb_p3, ir_p3)
        fuse_p4 = self.attn_fuse_p4(rgb_p4, ir_p4)
        fuse_p5 = self.attn_fuse_p5(rgb_p5, ir_p5)

        # Replace original RGB backbone P3/P4/P5 with fused features.
        y[self.p3_index] = fuse_p3
        y[self.p4_index] = fuse_p4
        y[self.p5_index] = fuse_p5

        # Continue original YOLOv10s neck + head.
        x = fuse_p5

        for m in self.model[self.backbone_end + 1:]:
            if m.f != -1:
                x_in = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x

            x = m(x_in)
            y.append(x if m.i in self.save else None)

        return x

    def init_criterion(self):
        return v10DetectLoss(self)


class RGBTMidAttentionFusionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        nc = self.data["nc"]

        model = RGBTYOLOv10MidAttentionFusion(
            cfg=BASE_MODEL_YAML,
            ch=4,
            nc=nc,
            verbose=verbose,
        )

        # Same initialization as Experiment D:
        # original YOLOv10s body loads yolov10s.pt,
        # IR backbone is copied from RGB backbone,
        # attention-fusion modules are randomly initialized.
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


# Alias for the test script.
RGBTMidAttentionFusionValidator = RGBTMidConcatValidator


def main():
    args = dict(
        # Keep original yolov10s.yaml, same as Experiment D.
        model=BASE_MODEL_YAML,

        # Keep the same RGBIR dataset config as Experiment D.
        data=DATA_YAML,

        # Keep settings unchanged for fair comparison with Experiment D.
        epochs=100,
        imgsz=640,
        batch=16,

        device=0,
        workers=8,

        amp=False,
        cache=False,

        project="runs/frgbt",
        name="E_midfusion_attention_fusion_yolov10s",

        # Keep HSV disabled to avoid augmentation errors on the IR channel.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )

    trainer = RGBTMidAttentionFusionTrainer(overrides=args)
    trainer.train()


if __name__ == "__main__":
    main()