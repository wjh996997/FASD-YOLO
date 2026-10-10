from pathlib import Path
from collections import Counter

DATASET_ROOT = Path(r"D:\yolov10\F-RGBT")

OLD_LABEL_ROOT = DATASET_ROOT / "labels"
NEW_LABEL_ROOT = DATASET_ROOT / "labels_2cls"

KEEP_CLASSES = {0, 1}

stats = {
    "train": Counter(),
    "val": Counter(),
    "test": Counter()
}

for split in ["train", "val", "test"]:
    old_dir = OLD_LABEL_ROOT / split
    new_dir = NEW_LABEL_ROOT / split
    new_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in old_dir.rglob("*.txt"):
        rel_path = txt_path.relative_to(old_dir)
        new_txt_path = new_dir / rel_path
        new_txt_path.parent.mkdir(parents=True, exist_ok=True)

        old_lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        new_lines = []

        for line in old_lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            cls_id = int(float(parts[0]))

            stats[split][f"class_{cls_id}_old"] += 1

            if cls_id in KEEP_CLASSES:
                new_lines.append(line)
                stats[split][f"class_{cls_id}_kept"] += 1
            else:
                stats[split][f"class_{cls_id}_removed"] += 1

        new_txt_path.write_text("\n".join(new_lines), encoding="utf-8")

print("=" * 80)
print("person 类别清理完成")
print("新标签目录:", NEW_LABEL_ROOT)
print("=" * 80)

for split in ["train", "val", "test"]:
    print(f"\n{split}:")
    for k, v in sorted(stats[split].items()):
        print(f"  {k}: {v}")