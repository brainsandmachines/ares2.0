import random
import shutil
from pathlib import Path

SRC_TRAIN = Path("/mnt/data/datasets/imagenet/train")
DST_TRAIN = Path("/mnt/data/datasets/imagenet_sample/train")
SRC_VAL = Path("/mnt/data/datasets/imagenet/val")
DST_VAL = Path("/mnt/data/datasets/imagenet_sample/val")

# Ensure root directories exist
DST_TRAIN.mkdir(parents=True, exist_ok=True)
DST_VAL.mkdir(parents=True, exist_ok=True)

def pick_two_images(imgs):
    """Return 2 images, duplicate if only one."""
    if len(imgs) >= 2:
        return random.sample(imgs, 2)
    return [imgs[0], imgs[0]]

# ---------------------------------------------------------
# TRAIN: COPY EXACTLY 2 IMAGES PER CLASS
# ---------------------------------------------------------

for src_cls in SRC_TRAIN.iterdir():
    if not src_cls.is_dir():
        continue

    dst_cls = DST_TRAIN / src_cls.name
    dst_cls.mkdir(parents=True, exist_ok=True)

    src_imgs = list(src_cls.glob("*"))
    if not src_imgs:
        print(f"[WARN] No source images in {src_cls.name}")
        continue

    chosen = pick_two_images(src_imgs)

    for idx, src in enumerate(chosen, 1):
        dst_name = src.name
        if chosen[0] == chosen[1]:  # duplicate case
            dst_name = f"{src.stem}_copy{idx}{src.suffix}"
        shutil.copy(src, dst_cls / dst_name)

    print(f"[TRAIN] Copied 2 images for class {src_cls.name}")

# ---------------------------------------------------------
# VAL: COPY EXACTLY 1 IMAGE PER CLASS
# ---------------------------------------------------------

for src_cls in SRC_VAL.iterdir():
    if not src_cls.is_dir():
        continue

    dst_cls = DST_VAL / src_cls.name
    dst_cls.mkdir(parents=True, exist_ok=True)

    src_imgs = list(src_cls.glob("*"))
    if not src_imgs:
        print(f"[WARN] No source images in val/{src_cls.name}")
        continue

    chosen = random.choice(src_imgs)
    shutil.copy(chosen, dst_cls / chosen.name)

    print(f"[VAL] Copied 1 image for class {src_cls.name}")
