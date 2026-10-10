import torch
from ultralytics import YOLOv10


DATA_YAML = "/root/autodl-tmp/yolov10/ultralytics/cfg/datasets/frgbt_rgbir.yaml"
MODEL_YAML = "/root/autodl-tmp/yolov10/ultralytics/cfg/models/v10/yolov10s-rgbir.yaml"
PRETRAINED = "/root/autodl-tmp/yolov10/yolov10s.pt"


def load_rgbir_pretrained(model, pretrained_path):
    """Load 3-channel YOLOv10s weights into 4-channel RGBIR model.

    First conv:
    RGB channels use pretrained weights.
    IR channel uses the mean of RGB weights.
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

    dst_sd = model.model.state_dict()
    new_sd = dst_sd.copy()

    loaded = 0
    expanded = 0
    skipped = 0

    for k, dst_w in dst_sd.items():
        if k not in src_sd:
            skipped += 1
            continue

        src_w = src_sd[k]
        if hasattr(src_w, "detach"):
            src_w = src_w.detach()

        if src_w.shape == dst_w.shape:
            new_sd[k] = src_w
            loaded += 1

        elif (
            src_w.ndim == 4
            and dst_w.ndim == 4
            and src_w.shape[0] == dst_w.shape[0]
            and src_w.shape[1] == 3
            and dst_w.shape[1] == 4
            and src_w.shape[2:] == dst_w.shape[2:]
        ):
            # Conv weight: [out_channels, in_channels, k, k]
            w = dst_w.clone()
            w[:, :3, :, :] = src_w
            w[:, 3:4, :, :] = src_w.mean(dim=1, keepdim=True)
            new_sd[k] = w
            expanded += 1
            print(f"[Expanded RGB->RGBIR] {k}: {tuple(src_w.shape)} -> {tuple(w.shape)}")

        else:
            skipped += 1

    msg = model.model.load_state_dict(new_sd, strict=False)
    print(f"Loaded layers: {loaded}, expanded conv layers: {expanded}, skipped: {skipped}")
    print(msg)


def main():
    model = YOLOv10(MODEL_YAML)
    load_rgbir_pretrained(model, PRETRAINED)

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        amp=False,
        cache=False,
        project="runs/frgbt",
        name="C_rgbir_input_fusion_yolov10s",

        # RGBIR 4通道实验先关闭颜色增强，保证 IR 通道不被错误处理
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )


if __name__ == "__main__":
    main()