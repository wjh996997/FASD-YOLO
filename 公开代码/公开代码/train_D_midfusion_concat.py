import os
os.environ["OMP_NUM_THREADS"] = "8"

import torch
import torch.nn as nn
from copy import deepcopy, copy

from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import YOLOv10DetectionModel
from ultralytics.nn.modules import Conv
from ultralytics.utils.loss import v10DetectLoss


DATA_YAML = "/root/autodl-tmp/yolov10/ultralytics/cfg/datasets/frgbt_rgbir.yaml"
BASE_MODEL_YAML = "/root/autodl-tmp/yolov10/ultralytics/cfg/models/v10/yolov10s.yaml"
PRETRAINED = "/root/autodl-tmp/yolov10/yolov10s.pt"


class RGBTYOLOv10MidConcat(YOLOv10DetectionModel):
    """
    Experiment D: YOLOv10s-MidFusion-Concat

    Input:
        x: [B, 4, H, W]
        RGB: x[:, 0:3, :, :]
        IR:  x[:, 3:4, :, :]

    Fusion:
        RGB_P3 + IR_P3 -> Concat -> 1x1 Conv -> Fuse_P3
        RGB_P4 + IR_P4 -> Concat -> 1x1 Conv -> Fuse_P4
        RGB_P5 + IR_P5 -> Concat -> 1x1 Conv -> Fuse_P5
    """

    def __init__(
        self,
        cfg=BASE_MODEL_YAML,
        ch=4,
        nc=None,
        verbose=True,
    ):
        # 注意：这里先用 ch=3 初始化原始 YOLOv10s。
        # 这样 self.model 是完整的原 YOLOv10s：
        # RGB backbone + Neck + Head。
        super().__init__(cfg=cfg, ch=3, nc=nc, verbose=verbose)

        # YOLOv10s 的 backbone 通常是 0-10 层。
        # 0-10: Backbone
        # 11-end: Neck + Head
        self.backbone_end = 10

        # YOLOv10s 中通常：
        # layer 4  -> P3/8
        # layer 6  -> P4/16
        # layer 10 -> P5/32
        self.p3_index = 4
        self.p4_index = 6
        self.p5_index = 10

        # 复制一份 RGB backbone 作为 IR backbone
        self.ir_backbone = deepcopy(self.model[: self.backbone_end + 1])

        # 把 IR 分支第一层卷积从 3 通道改成 1 通道
        self._convert_ir_first_conv_to_1ch()

        # YOLOv10s 的 P3/P4/P5 通道数。
        # 对 yolov10s.yaml 通常是 128, 256, 512。
        c3, c4, c5 = 128, 256, 512

        # Concat 后通道数翻倍，所以用 1x1 Conv 压回原通道数
        self.fuse_p3 = Conv(c3 * 2, c3, k=1, s=1)
        self.fuse_p4 = Conv(c4 * 2, c4, k=1, s=1)
        self.fuse_p5 = Conv(c5 * 2, c5, k=1, s=1)

    def _convert_ir_first_conv_to_1ch(self):
        """
        将 IR backbone 的第一层 Conv2d 从 3 输入通道改成 1 输入通道。
        初始化时使用 RGB 三通道权重均值。
        """

        first = self.ir_backbone[0]

        # Ultralytics 的 Conv 模块内部一般是 first.conv
        old_conv = first.conv

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=False,
        )

        with torch.no_grad():
            if old_conv.weight.shape[1] == 3:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        first.conv = new_conv

    def _forward_backbone(self, x, modules):
        """
        只前向传播 backbone，并返回 P3, P4, P5。

        对 YOLOv10s:
            layer 4  -> P3/8
            layer 6  -> P4/16
            layer 10 -> P5/32
        """

        y = []

        for m in modules:
            if m.f != -1:
                x_in = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x

            x = m(x_in)
            y.append(x)

        p3 = y[self.p3_index]
        p4 = y[self.p4_index]
        p5 = y[self.p5_index]

        return p3, p4, p5, y

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        重写 YOLOv10 的前向传播。

        如果输入是 3 通道：
            走原始 YOLOv10s 前向。
            这个分支主要用于模型初始化、stride 计算等内部过程。

        如果输入是 4 通道：
            走 RGBT 中期融合。
        """

        if not hasattr(self, "_debug_printed"):
            print("[Debug] input shape:", x.shape)
            self._debug_printed = True
        # 初始化 stride 或普通 RGB 测试时可能会传入 3 通道
        if x.shape[1] == 3:
            return super()._predict_once(
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

        # 保险检查：RGB 与 IR 同尺度特征尺寸必须一致
        assert rgb_p3.shape[2:] == ir_p3.shape[2:], f"P3 size mismatch: {rgb_p3.shape} vs {ir_p3.shape}"
        assert rgb_p4.shape[2:] == ir_p4.shape[2:], f"P4 size mismatch: {rgb_p4.shape} vs {ir_p4.shape}"
        assert rgb_p5.shape[2:] == ir_p5.shape[2:], f"P5 size mismatch: {rgb_p5.shape} vs {ir_p5.shape}"

        # P3/P4/P5 中期融合：Concat + 1x1 Conv
        fuse_p3 = self.fuse_p3(torch.cat([rgb_p3, ir_p3], dim=1))
        fuse_p4 = self.fuse_p4(torch.cat([rgb_p4, ir_p4], dim=1))
        fuse_p5 = self.fuse_p5(torch.cat([rgb_p5, ir_p5], dim=1))

        # 用融合特征替换原 RGB backbone 中的 P3/P4/P5
        y[self.p3_index] = fuse_p3
        y[self.p4_index] = fuse_p4
        y[self.p5_index] = fuse_p5

        # 从 backbone 后一层开始，继续走原 YOLOv10s 的 Neck + Head
        x = fuse_p5

        for m in self.model[self.backbone_end + 1:]:
            if m.f != -1:
                x_in = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x

            x = m(x_in)

            # 这里保持 Ultralytics 的保存逻辑
            y.append(x if m.i in self.save else None)

        return x

    def init_ir_backbone_from_rgb(self):
        """
        加载 yolov10s.pt 之后调用。
        将 RGB backbone 的权重复制到 IR backbone。
        第一层卷积由于输入通道不同，所以用 RGB 三通道均值初始化。
        """

        rgb_backbone = self.model[: self.backbone_end + 1]
        rgb_sd = rgb_backbone.state_dict()
        ir_sd = self.ir_backbone.state_dict()

        load_sd = {}

        for k, v in rgb_sd.items():
            if k in ir_sd and ir_sd[k].shape == v.shape:
                load_sd[k] = v

        msg = self.ir_backbone.load_state_dict(load_sd, strict=False)
        print("[MidFusion] IR backbone copied from RGB backbone.")
        print(msg)

        rgb_first = self.model[0].conv
        ir_first = self.ir_backbone[0].conv

        with torch.no_grad():
            if rgb_first.weight.shape[1] == 3 and ir_first.weight.shape[1] == 1:
                ir_first.weight.copy_(rgb_first.weight.mean(dim=1, keepdim=True))
                print("[MidFusion] IR first conv initialized by RGB mean.")

    def init_criterion(self):
        return v10DetectLoss(self)


def load_midfusion_pretrained(model, pretrained_path):
    """
    实验 D 的预训练加载方式：

    1. 原 YOLOv10s 的 self.model 加载 yolov10s.pt；
       包括 RGB backbone、Neck、Head。
    2. IR backbone 从 RGB backbone 复制；
    3. IR 第一层使用 RGB 三通道均值初始化；
    4. fuse_p3/fuse_p4/fuse_p5 随机初始化。
    """

    try:
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(pretrained_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        src_model = ckpt["model"]
        if hasattr(src_model, "float"):
            src_sd = src_model.float().state_dict()
        else:
            src_sd = src_model
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        src_sd = ckpt["state_dict"]
    else:
        src_sd = ckpt

    dst_sd = model.state_dict()
    new_sd = dst_sd.copy()

    loaded = 0
    skipped = 0

    for k, dst_w in dst_sd.items():
        # 只加载原 YOLOv10s 主体 self.model 的权重
        # ir_backbone 和 fuse 层不直接从 ckpt 加载
        if not k.startswith("model."):
            skipped += 1
            continue

        if k not in src_sd:
            skipped += 1
            continue

        src_w = src_sd[k]

        if hasattr(src_w, "detach"):
            src_w = src_w.detach()

        if src_w.shape == dst_w.shape:
            new_sd[k] = src_w
            loaded += 1
        else:
            skipped += 1

    msg = model.load_state_dict(new_sd, strict=False)

    print(f"[MidFusion] Loaded YOLOv10s weights: {loaded}")
    print(f"[MidFusion] Skipped weights: {skipped}")
    print(msg)

    # 用已经加载好的 RGB backbone 初始化 IR backbone
    model.init_ir_backbone_from_rgb()

class RGBTMidConcatValidator(DetectionValidator):
    """
    Validator for YOLOv10 mid-fusion model.

    YOLOv10 v10Detect may output a dict during validation.
    The standard YOLOv8 DetectionValidator expects a Tensor before NMS.
    This validator extracts the inference prediction Tensor from YOLOv10 output.
    """

    def postprocess(self, preds):
        # YOLOv10 often returns dict-like predictions:
        # {'one2many': ..., 'one2one': ...}
        if isinstance(preds, dict):
            if "one2one" in preds:
                preds = preds["one2one"]
            elif "one2many" in preds:
                preds = preds["one2many"]
            else:
                # fallback: take the first value
                preds = next(iter(preds.values()))

        # Some heads return tuple/list, usually first element is inference prediction
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        return super().postprocess(preds)


class RGBTMidConcatTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        nc = self.data["nc"]

        model = RGBTYOLOv10MidConcat(
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
        # 这里填原始 yolov10s.yaml，不要填 yolov10s-rgbir.yaml
        model=BASE_MODEL_YAML,

        # 继续使用实验 C 的 RGBIR 数据配置
        data=DATA_YAML,

        epochs=100,
        imgsz=640,

        # 双 backbone 显存更大，先用 8。
        # 如果显存不够，改成 4。
        # 如果显存足够，为了和实验 C 更公平，可以再试 batch=16。
        batch=16,

        device=0,
        workers=8,

        # 和你的实验 C 保持一致
        amp=False,
        cache=False,

        project="runs/frgbt",
        name="D_midfusion_concat_yolov10s",

        # 和实验 C 保持一致，避免 HSV 增强错误影响 IR 通道
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )

    trainer = RGBTMidConcatTrainer(overrides=args)
    trainer.train()


if __name__ == "__main__":
    main()