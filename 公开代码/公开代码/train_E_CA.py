# train_E_CA.py
import os
os.environ["OMP_NUM_THREADS"] = "8"

import torch
import torch.nn as nn
from copy import copy

from train_D_midfusion_concat import (
    RGBTYOLOv10MidConcat,
    load_midfusion_pretrained,
    DATA_YAML,
    BASE_MODEL_YAML,
    PRETRAINED,
)

from ultralytics.models.yolo.detect import DetectionTrainer


# -------------------------------------------------
# Standard Coordinate Attention
# -------------------------------------------------
class CoordinateAttention(nn.Module):
    """
    Coordinate Attention:
    输入:  [B, C, H, W]
    输出:  [B, C, H, W]

    注意：
    x_h: [B, C, H, 1]
    x_w: [B, C, 1, W] -> permute -> [B, C, W, 1]
    然后沿 H/W 方向拼接为 [B, C, H+W, 1]
    """
    def __init__(self, channels, reduction=32):
        super().__init__()

        hidden = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        b, c, h, w = x.shape

        # [B, C, H, 1]
        x_h = torch.mean(x, dim=3, keepdim=True)

        # [B, C, 1, W] -> [B, C, W, 1]
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)

        # [B, C, H + W, 1]
        y = torch.cat([x_h, x_w], dim=2)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # x_h: [B, hidden, H, 1]
        # x_w: [B, hidden, W, 1]
        x_h, x_w = torch.split(y, [h, w], dim=2)

        # [B, hidden, W, 1] -> [B, hidden, 1, W]
        x_w = x_w.permute(0, 1, 3, 2)

        # [B, C, H, 1]
        a_h = self.sigmoid(self.conv_h(x_h))

        # [B, C, 1, W]
        a_w = self.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w

        return out


# -------------------------------------------------
# RGB + IR Coordinate Attention Fusion
# -------------------------------------------------
class CoordinateAttentionFusion(nn.Module):
    """
    双模态 Coordinate Attention 融合模块：

    RGB feature: [B, C, H, W]
    IR feature:  [B, C, H, W]

    concat:      [B, 2C, H, W]
    reduce:      [B, C, H, W]
    CA:          [B, C, H, W]
    """
    def __init__(self, channels, reduction=32):
        super().__init__()

        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.ca = CoordinateAttention(channels, reduction=reduction)

    def forward(self, rgb, ir):
        x = torch.cat([rgb, ir], dim=1)
        x = self.reduce(x)
        x = self.ca(x)
        return x


# -------------------------------------------------
# YOLOv10s Mid Fusion + Coordinate Attention
# -------------------------------------------------
class RGBTYOLOv10MidAttentionFusionCA(RGBTYOLOv10MidConcat):
    def __init__(self, cfg=BASE_MODEL_YAML, ch=4, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # 删除实验 D 中 concat + 1x1 Conv 的融合层
        for name in ("fuse_p3", "fuse_p4", "fuse_p5"):
            if hasattr(self, name):
                delattr(self, name)

        # YOLOv10s P3/P4/P5 通道数
        c3, c4, c5 = 128, 256, 512

        self.attn_fuse_p3 = CoordinateAttentionFusion(c3)
        self.attn_fuse_p4 = CoordinateAttentionFusion(c4)
        self.attn_fuse_p5 = CoordinateAttentionFusion(c5)

        self.fusion_type = "CA"

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        # 初始化 stride 时会走 3 通道 dummy input
        if x.shape[1] == 3:
            return super()._predict_once(
                x,
                profile=profile,
                visualize=visualize,
                embed=embed,
            )

        assert x.shape[1] == 4, f"Expected 4-channel RGB-IR input, but got {x.shape[1]} channels."

        rgb = x[:, 0:3, :, :]
        ir = x[:, 3:4, :, :]

        # RGB backbone
        rgb_p3, rgb_p4, rgb_p5, y = self._forward_backbone(
            rgb,
            self.model[:self.backbone_end + 1],
        )

        # IR backbone
        ir_p3, ir_p4, ir_p5, _ = self._forward_backbone(
            ir,
            self.ir_backbone,
        )

        # shape 检查
        assert rgb_p3.shape == ir_p3.shape, f"P3 shape mismatch: {rgb_p3.shape} vs {ir_p3.shape}"
        assert rgb_p4.shape == ir_p4.shape, f"P4 shape mismatch: {rgb_p4.shape} vs {ir_p4.shape}"
        assert rgb_p5.shape == ir_p5.shape, f"P5 shape mismatch: {rgb_p5.shape} vs {ir_p5.shape}"

        # P3/P4/P5 双模态 Coordinate Attention 融合
        fuse_p3 = self.attn_fuse_p3(rgb_p3, ir_p3)
        fuse_p4 = self.attn_fuse_p4(rgb_p4, ir_p4)
        fuse_p5 = self.attn_fuse_p5(rgb_p5, ir_p5)

        y[self.p3_index] = fuse_p3
        y[self.p4_index] = fuse_p4
        y[self.p5_index] = fuse_p5

        # Neck + Head
        x = fuse_p5

        for m in self.model[self.backbone_end + 1:]:
            if m.f != -1:
                if isinstance(m.f, int):
                    x_in = y[m.f]
                else:
                    x_in = [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x

            x = m(x_in)
            y.append(x if m.i in self.save else None)

        return x

    def init_criterion(self):
        from ultralytics.utils.loss import v10DetectLoss
        return v10DetectLoss(self)


# -------------------------------------------------
# Trainer
# -------------------------------------------------
class RGBTMidAttentionFusionCATrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        nc = self.data["nc"]

        model = RGBTYOLOv10MidAttentionFusionCA(
            cfg=BASE_MODEL_YAML,
            ch=4,
            nc=nc,
            verbose=verbose,
        )

        load_midfusion_pretrained(model, PRETRAINED)

        return model

    def get_validator(self):
        from train_D_midfusion_concat import RGBTMidConcatValidator

        self.loss_names = "box_loss", "cls_loss", "dfl_loss"

        return RGBTMidConcatValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


# -------------------------------------------------
# Main
# -------------------------------------------------
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
        project="runs",
        name="E_midfusion_CA_yolov10s",

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )

    trainer = RGBTMidAttentionFusionCATrainer(overrides=args)
    trainer.train()


if __name__ == "__main__":
    main()