from ultralytics import YOLOv10


def main():
    weights = "runs/frgbt/C_rgbir_input_fusion_yolov10s/weights/best.pt"

    # 改成你训练时用的 data yaml
    data = "/root/autodl-tmp/yolov10/ultralytics/cfg/datasets/frgbt_rgbir.yaml"

    model = YOLOv10(weights)

    metrics = model.val(
        data=data,
        imgsz=640,
        batch=1,
        device=0,
        split="val",
        workers=0,
        project="runs/frgbt",
        name="C_rgbir_input_fusion_yolov10s_val",
        exist_ok=True,
    )

    print("\n====== Validation Results ======")
    print(f"Precision:  {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall:     {metrics.results_dict['metrics/recall(B)']:.4f}")
    print(f"mAP50:      {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95:   {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"Fitness:    {metrics.results_dict['fitness']:.4f}")
    print(f"Save dir:   {metrics.save_dir}")


if __name__ == "__main__":
    main()