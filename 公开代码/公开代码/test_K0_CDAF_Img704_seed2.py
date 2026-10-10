# -*- coding: utf-8 -*-
"""
Test script for K1_J3_CMAFF_Img704_FireRefine_yolov10s.

注意：K1 使用 imgsz=704 训练，因此测试也默认使用 imgsz=704。
"""

import os
os.environ["OMP_NUM_THREADS"] = "8"

import sys
import torch
from copy import copy
from pathlib import Path

from train_E_CMAFF import (
    ChannelSpatialAttention,
    CMAFFStyleFusion,
    RGBTYOLOv10MidFusionCMAFF,
    BASE_MODEL_YAML,
    DATA_YAML,
)
from train_D_midfusion_concat import RGBTMidConcatValidator

# 关键修正：让 torch.load 能找到 best.pt 里的自定义类
sys.modules["__main__"].ChannelSpatialAttention = ChannelSpatialAttention
sys.modules["__main__"].CMAFFStyleFusion = CMAFFStyleFusion
sys.modules["__main__"].RGBTYOLOv10MidFusionCMAFF = RGBTYOLOv10MidFusionCMAFF

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg


EXP_NAME = "K0_CDAF_Img704_seed2_yolov10s"
BEST_PT = f"/root/autodl-tmp/yolov10/runs/frgbt/{EXP_NAME}/weights/best.pt"


def main():
    data = check_det_dataset(DATA_YAML)

    model = RGBTYOLOv10MidFusionCMAFF(
        cfg=BASE_MODEL_YAML,
        ch=4,
        nc=data["nc"],
        verbose=False,
    )

    ckpt = torch.load(BEST_PT, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if ckpt.get("ema", None) is not None:
            src = ckpt["ema"]
            state_dict = src.float().state_dict() if hasattr(src, "float") else src
        elif "model" in ckpt:
            src = ckpt["model"]
            state_dict = src.float().state_dict() if hasattr(src, "float") else src
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    msg = model.load_state_dict(state_dict, strict=False)
    print("[Load best.pt]", msg)

    model.eval().cuda()

    args = get_cfg(DEFAULT_CFG)
    args.data = DATA_YAML
    args.imgsz = 704
    args.batch = 12
    args.split = "test"
    args.device = 0
    args.workers = 8
    args.plots = True
    args.save_json = False
    args.conf = None
    args.iou = 0.7
    args.max_det = 300
    args.half = False
    args.dnn = False
    args.task = "detect"
    args.mode = "val"

    dataset = build_yolo_dataset(
        args,
        data["test"],
        args.batch,
        data,
        mode="val",
        rect=False,
        stride=32,
    )

    dataloader = build_dataloader(
        dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=False,
        rank=-1,
    )

    batch = next(iter(dataloader))
    print("[Check] batch['img'].shape =", batch["img"].shape)
    print("[Check] batch['img'].dtype =", batch["img"].dtype)

    validator = RGBTMidConcatValidator(
        dataloader=dataloader,
        save_dir=Path(f"runs/frgbt/{EXP_NAME}_test"),
        args=copy(args),
        _callbacks=None,
    )
    validator.data = data

    results = validator(model=model)

    print(f"\n[Test results] {EXP_NAME}")
    print(results)


if __name__ == "__main__":
    main()
