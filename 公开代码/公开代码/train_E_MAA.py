# train_E_MAA.py
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

# ----------------------
# Modality-Aware Attention
# ----------------------
class ModalityAwareAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.rgb_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )
        self.ir_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

    def forward(self, rgb, ir):
        rgb_logit = self.rgb_fc(rgb)
        ir_logit = self.ir_fc(ir)
        logits = torch.stack([rgb_logit, ir_logit], dim=1)
        weights = torch.softmax(logits, dim=1)
        fuse = weights[:,0] * rgb + weights[:,1] * ir
        return fuse

# ----------------------
# YOLOv10s Mid Attention Fusion (MAA)
# ----------------------
class RGBTYOLOv10MidAttentionFusionMAA(RGBTYOLOv10MidConcat):
    def __init__(self, cfg=BASE_MODEL_YAML, ch=4, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        for name in ("fuse_p3","fuse_p4","fuse_p5"):
            if hasattr(self, name):
                delattr(self, name)
        c3,c4,c5 = 128,256,512
        self.attn_fuse_p3 = ModalityAwareAttention(c3)
        self.attn_fuse_p4 = ModalityAwareAttention(c4)
        self.attn_fuse_p5 = ModalityAwareAttention(c5)
        self.fusion_type = "MAA"

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if x.shape[1] == 3:
            return super()._predict_once(x, profile=profile, visualize=visualize, embed=embed)

        assert x.shape[1] == 4
        rgb, ir = x[:,0:3,:,:], x[:,3:4,:,:]

        # Backbone 特征
        rgb_p3,rgb_p4,rgb_p5,y = self._forward_backbone(rgb, self.model[:self.backbone_end+1])
        ir_p3, ir_p4, ir_p5,_ = self._forward_backbone(ir, self.ir_backbone)

        # 融合
        fuse_p3 = self.attn_fuse_p3(rgb_p3, ir_p3)
        fuse_p4 = self.attn_fuse_p4(rgb_p4, ir_p4)
        fuse_p5 = self.attn_fuse_p5(rgb_p5, ir_p5)
        y[self.p3_index], y[self.p4_index], y[self.p5_index] = fuse_p3, fuse_p4, fuse_p5

        # Neck + Head
        x = fuse_p5
        for m in self.model[self.backbone_end+1:]:
            if m.f != -1:
                if isinstance(m.f, int):
                    x_in = y[m.f]
                else:  # m.f 是列表
                    x_in = [x if j==-1 else y[j] for j in m.f]
            else:
                x_in = x
            x = m(x_in)
            y.append(x if m.i in self.save else None)
        return x

    def init_criterion(self):
        from ultralytics.utils.loss import v10DetectLoss
        return v10DetectLoss(self)

# ----------------------
# Trainer
# ----------------------
class RGBTMidAttentionFusionMAATrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        nc = self.data["nc"]
        model = RGBTYOLOv10MidAttentionFusionMAA(cfg=BASE_MODEL_YAML, ch=4, nc=nc)
        load_midfusion_pretrained(model, PRETRAINED)
        return model

    def get_validator(self):
        from train_D_midfusion_concat import RGBTMidConcatValidator
        self.loss_names = "box_loss","cls_loss","dfl_loss"
        return RGBTMidConcatValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

# ----------------------
# Main
# ----------------------
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
        name="E_midfusion_MAA_yolov10s",
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    )
    trainer = RGBTMidAttentionFusionMAATrainer(overrides=args)
    trainer.train()

if __name__=="__main__":
    main()