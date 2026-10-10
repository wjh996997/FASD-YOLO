# FASD-YOLO

Code accompanying **FASD-YOLO: False-Positive-Aware Visible–Thermal Infrared Dual-Modal Fusion for Low-Altitude Forest Fire Detection**.

**Authors:** Jin Wang, Chao Sun, and Yuechan Liu.

**Target journal:** IEEE Geoscience and Remote Sensing Letters (GRSL).

## Contents

- `公开代码/公开代码/`: original RGB–thermal infrared training and evaluation scripts, fusion baselines and attention ablations, three CDAF seed runs, parameter/FLOPs analysis, and label preparation.
- `yolov10/yolov10-main/`: the accompanying YOLOv10 source tree, configurations, dependencies, upstream documentation, examples, and logs.

The original directory structure and research code are preserved. Datasets and model checkpoints are not included. The manuscript PDF is not included.

## Setup and usage

Use a Python environment with PyTorch suitable for your hardware. Install the included YOLOv10 source tree from the repository root:

```bash
pip install -e yolov10/yolov10-main
```

The scripts retain paths from the original machines, including `/root/autodl-tmp/yolov10` and `D:\yolov10`. Before running them, adapt dataset, model configuration, checkpoint, and output paths to your environment. Provide your own authorized paired RGB–thermal dataset and required checkpoints.

Run research scripts from `公开代码/公开代码/` so their local imports resolve. The main CDAF scripts are `train_K0_CDAF_Img704_seed0.py`, `train_K0_CDAF_Img704_seed1.py`, and `train_K0_CDAF_Img704_seed2.py`; corresponding `test_K0_CDAF_Img704_seed*.py` files perform evaluation. These training scripts reference precursor checkpoints; inspect their paths before execution.

This publication preparation did not run training or verify experiment reproducibility.

## Attribution and licensing

The included YOLOv10 project retains its original README, attribution, license notices, and `yolov10/yolov10-main/LICENSE` (GNU AGPL v3). See the upstream project: https://github.com/THU-MIG/yolov10.

Third-party files remain subject to their respective license terms. No new license is assigned to the authors' research scripts by this upload; public availability alone does not grant unrestricted reuse rights.

## Citation

Jin Wang, Chao Sun, and Yuechan Liu. “FASD-YOLO: False-Positive-Aware Visible–Thermal Infrared Dual-Modal Fusion for Low-Altitude Forest Fire Detection.” Manuscript for IEEE Geoscience and Remote Sensing Letters.