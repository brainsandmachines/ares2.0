import random, shutil
from pathlib import Path

SRC_VAL = Path("/storage/test/bml_group/tomerash/datasets/imagenet/val")
DST_VAL = Path("/storage/test/bml_group/tomerash/datasets/imagenet_sample/val")

for cls_dir in DST_VAL.iterdir():
    if not cls_dir.is_dir():
        continue
    images = list(cls_dir.glob("*"))
    if len(images) == 0:
        src_cls = SRC_VAL / cls_dir.name
        if not src_cls.exists():
            print(f"[WARN] Missing source folder for {cls_dir.name}")
            continue
        src_imgs = list(src_cls.glob("*"))
        if not src_imgs:
            print(f"[WARN] No source images found for {cls_dir.name}")
            continue
        chosen = random.choice(src_imgs)
        shutil.copy(chosen, cls_dir / chosen.name)
        print(f"[FIXED] Added one image for empty class {cls_dir.name}")
