# test Params_FLOPs.py
# 鐢ㄩ€旓細妫€鏌ュ疄楠孌 MidFusion-Concat 妯″瀷鐨� Params 鍜� FLOPs
# 璇存槑锛氳繖涓� best.pt 涓寘鍚嚜瀹氫箟绫� RGBTYOLOv10MidConcat锛�
#      鎵€浠ュ繀椤诲湪 YOLO(pt_path) 涔嬪墠瀹氫箟璇ョ被锛屽惁鍒欎細鎶ワ細
#      AttributeError: Can't get attribute 'RGBTYOLOv10MidConcat'

import os
os.environ["OMP_NUM_THREADS"] = "8"

import torch
import torch.nn as nn
from copy import deepcopy

from ultralytics import YOLO
from ultralytics.nn.tasks import YOLOv10DetectionModel
from ultralytics.nn.modules import Conv
from ultralytics.utils.loss import v10DetectLoss


# 杩欎釜璺緞鍙湪鈥滈噸鏂板疄渚嬪寲妯″瀷鈥濇椂鏈夌敤锛�
# 鍔犺浇 best.pt 鏃堕€氬父涓嶄細鐪熸鐢ㄥ埌 cfg锛屼絾淇濈暀瀹冨彲浠ラ伩鍏嶇被鍒濆鍖栨椂鎶ラ敊銆�
BASE_MODEL_YAML = r"D:\yolov10\ultralytics\cfg\models\v10\yolov10s.yaml"


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

    def __init__(self, cfg=BASE_MODEL_YAML, ch=4, nc=None, verbose=True):
        # 鍏堢敤 ch=3 鍒濆鍖栧師濮� YOLOv10s
        super().__init__(cfg=cfg, ch=3, nc=nc, verbose=verbose)

        self.backbone_end = 10
        self.p3_index = 4
        self.p4_index = 6
        self.p5_index = 10

        # 澶嶅埗 RGB backbone 浣滀负 IR backbone
        self.ir_backbone = deepcopy(self.model[: self.backbone_end + 1])

        # 灏� IR 鍒嗘敮绗竴灞傚嵎绉粠 3 閫氶亾鏀规垚 1 閫氶亾
        self._convert_ir_first_conv_to_1ch()

        # YOLOv10s P3/P4/P5 閫氶亾鏁�
        c3, c4, c5 = 128, 256, 512

        self.fuse_p3 = Conv(c3 * 2, c3, k=1, s=1)
        self.fuse_p4 = Conv(c4 * 2, c4, k=1, s=1)
        self.fuse_p5 = Conv(c5 * 2, c5, k=1, s=1)

    def _convert_ir_first_conv_to_1ch(self):
        first = self.ir_backbone[0]
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
        # 鍒濆鍖� stride 鎴栨櫘閫� RGB 娴嬭瘯鏃跺彲鑳戒紶鍏� 3 閫氶亾
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

        assert rgb_p3.shape[2:] == ir_p3.shape[2:], f"P3 size mismatch: {rgb_p3.shape} vs {ir_p3.shape}"
        assert rgb_p4.shape[2:] == ir_p4.shape[2:], f"P4 size mismatch: {rgb_p4.shape} vs {ir_p4.shape}"
        assert rgb_p5.shape[2:] == ir_p5.shape[2:], f"P5 size mismatch: {rgb_p5.shape} vs {ir_p5.shape}"

        fuse_p3 = self.fuse_p3(torch.cat([rgb_p3, ir_p3], dim=1))
        fuse_p4 = self.fuse_p4(torch.cat([rgb_p4, ir_p4], dim=1))
        fuse_p5 = self.fuse_p5(torch.cat([rgb_p5, ir_p5], dim=1))

        y[self.p3_index] = fuse_p3
        y[self.p4_index] = fuse_p4
        y[self.p5_index] = fuse_p5

        x = fuse_p5

        for m in self.model[self.backbone_end + 1:]:
            if m.f != -1:
                x_in = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x

            x = m(x_in)
            y.append(x if m.i in self.save else None)

        return x

    def init_ir_backbone_from_rgb(self):
        rgb_backbone = self.model[: self.backbone_end + 1]
        rgb_sd = rgb_backbone.state_dict()
        ir_sd = self.ir_backbone.state_dict()

        load_sd = {}

        for k, v in rgb_sd.items():
            if k in ir_sd and ir_sd[k].shape == v.shape:
                load_sd[k] = v

        self.ir_backbone.load_state_dict(load_sd, strict=False)

        rgb_first = self.model[0].conv
        ir_first = self.ir_backbone[0].conv

        with torch.no_grad():
            if rgb_first.weight.shape[1] == 3 and ir_first.weight.shape[1] == 1:
                ir_first.weight.copy_(rgb_first.weight.mean(dim=1, keepdim=True))

    def init_criterion(self):
        return v10DetectLoss(self)


def main():
    # 杩欓噷鏀规垚浣犺娴嬬殑 best.pt
    pt_path = r"D:\yolov10\runs\实验D 中期融合（双分支+简单拼接）\D_midfusion_concat_yolov10s\weights\best.pt"

    model = YOLO(pt_path)

    params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    print(f"Params: {params / 1e6:.3f} M")
    print(f"Trainable Params: {trainable_params / 1e6:.3f} M")

    try:
        from ultralytics.utils.torch_utils import get_flops

        flops_640 = get_flops(model.model, imgsz=640)
        flops_704 = get_flops(model.model, imgsz=704)

        print(f"FLOPs @640: {flops_640:.3f} G")
        print(f"FLOPs @704: {flops_704:.3f} G")

    except Exception as e:
        print("get_flops 涓嶆敮鎸佹垨璇ヨ嚜瀹氫箟妯″瀷鏃犳硶鐩存帴缁熻 FLOPs銆�")
        print("Error:", e)

        # 濡傛灉 get_flops 澶辫触锛屽彲浠ョ敤 model.info() 鍏堢湅榛樿 FLOPs
        print("\nUltralytics model.info():")
        model.info(verbose=True)


if __name__ == "__main__":
    main()