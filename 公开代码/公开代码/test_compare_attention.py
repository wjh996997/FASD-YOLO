# test_compare_attention.py
import os
os.environ["OMP_NUM_THREADS"] = "8"

import sys
import torch
from copy import copy
from pathlib import Path
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg

# 导入三种模型
from train_E_midfusion_attention_fusion import ModalityAttentionFusion, RGBTYOLOv10MidAttentionFusion, RGBTMidAttentionFusionValidator
from train_E_MAA import ModalityAwareAttention, RGBTYOLOv10MidAttentionFusionMAA, RGBTMidAttentionFusionMAATrainer
from train_E_CA import CoordinateAttention, RGBTYOLOv10MidAttentionFusionCA, RGBTMidAttentionFusionCATrainer
from train_D_midfusion_concat import DATA_YAML, BASE_MODEL_YAML

# 解决 torch.load 自定义类问题
sys.modules["__main__"].ModalityAttentionFusion = ModalityAttentionFusion
sys.modules["__main__"].RGBTYOLOv10MidAttentionFusion = RGBTYOLOv10MidAttentionFusion
sys.modules["__main__"].ModalityAwareAttention = ModalityAwareAttention
sys.modules["__main__"].RGBTYOLOv10MidAttentionFusionMAA = RGBTYOLOv10MidAttentionFusionMAA
sys.modules["__main__"].CoordinateAttention = CoordinateAttention
sys.modules["__main__"].RGBTYOLOv10MidAttentionFusionCA = RGBTYOLOv10MidAttentionFusionCA

# 权重路径
BEST_PT = {
    "CBAM": "runs/frgbt/E_midfusion_attention_fusion_yolov10s/weights/best.pt",
    "MAA":  "runs/frgbt/E_midfusion_MAA_yolov10s/weights/best.pt",
    "CA":   "runs/frgbt/E_midfusion_CA_yolov10s/weights/best.pt",
}

MODELS = {
    "CBAM": RGBTYOLOv10MidAttentionFusion,
    "MAA":  RGBTYOLOv10MidAttentionFusionMAA,
    "CA":   RGBTYOLOv10MidAttentionFusionCA,
}

VALIDATORS = {
    "CBAM": RGBTMidAttentionFusionValidator,
    "MAA":  RGBTMidAttentionFusionMAATrainer.get_validator,
    "CA":   RGBTMidAttentionFusionCATrainer.get_validator,
}

SAVE_DIR = "runs/frgbt/compare_attention_test"

def test_model(name):
    print(f"\n[Test] {name} model")
    data = check_det_dataset(DATA_YAML)

    model_cls = MODELS[name]
    model = model_cls(cfg=BASE_MODEL_YAML, ch=4, nc=data["nc"], verbose=False)
    ckpt = torch.load(BEST_PT[name], map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if hasattr(state_dict, "float"):
        state_dict = state_dict.float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    model.eval().cuda()
    print("[Load weights]", BEST_PT[name])

    # 构建 dataloader
    args = get_cfg(DEFAULT_CFG)
    args.data = DATA_YAML
    args.imgsz = 640
    args.batch = 16
    args.split = "test"
    args.device = 0
    args.workers = 8
    args.plots = True

    dataset = build_yolo_dataset(args, check_det_dataset(DATA_YAML)["test"], args.batch, data, mode="val", rect=False, stride=32)
    dataloader = build_dataloader(dataset, batch=args.batch, workers=args.workers, shuffle=False, rank=-1)

    validator_cls = VALIDATORS[name]
    if callable(validator_cls):
        validator = validator_cls(None)  # 对于 MA/CA 需要实例化
    else:
        validator = validator_cls(dataloader=dataloader, save_dir=Path(SAVE_DIR+"/"+name), args=copy(args), _callbacks=None)
    validator.data = data

    results = validator(model=model)
    print(f"[Results] {name}: {results}")
    return results

def main():
    results_dict = {}
    for name in ["CBAM","MAA","CA"]:
        results_dict[name] = test_model(name)

    # 打印对比表格
    print("\n=== Attention Mechanism Comparison ===")
    print(f"{'Method':<8} {'Precision':>8} {'Recall':>8} {'mAP50':>8} {'mAP50-95':>10}")
    for name, r in results_dict.items():
        print(f"{name:<8} {r['Precision']:.3f} {r['Recall']:.3f} {r['mAP50']:.3f} {r['mAP50-95']:.3f}")

if __name__=="__main__":
    main()