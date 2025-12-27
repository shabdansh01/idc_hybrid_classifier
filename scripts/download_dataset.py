import os
import shutil
import random
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

TRAIN_RATIO = 0.8
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

KAGGLE_DATASET = "paultimothymooney/breast-histopathology-images"

# -----------------------------
def ensure_dirs():
    for split in [TRAIN_DIR, VAL_DIR]:
        for cls in ["0", "1"]:
            (split / cls).mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
def download_kaggle_dataset():
    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset from Kaggle...")
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=RAW_DIR,
        unzip=True,
        quiet=False,
    )

# -----------------------------
def collect_images():
    """
    Kaggle structure:
    raw/
      └── IDC_regular_ps50_idx5/
          ├── 10253/
          │   ├── 0/
          │   └── 1/
          └── ...
    """
    source_root = RAW_DIR / "IDC_regular_ps50_idx5"
    class_images = {"0": [], "1": []}

    for patient_dir in source_root.iterdir():
        if not patient_dir.is_dir():
            continue
        for cls in ["0", "1"]:
            cls_dir = patient_dir / cls
            if cls_dir.exists():
                for img in cls_dir.iterdir():
                    if img.suffix.lower() in IMAGE_EXTENSIONS:
                        class_images[cls].append(img)

    return class_images

# -----------------------------
def split_and_copy(class_images):
    for cls, images in class_images.items():
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for img in tqdm(train_imgs, desc=f"Copying train class {cls}"):
            shutil.copy(img, TRAIN_DIR / cls / img.name)

        for img in tqdm(val_imgs, desc=f"Copying val class {cls}"):
            shutil.copy(img, VAL_DIR / cls / img.name)

        print(f"Class {cls}: {len(train_imgs)} train, {len(val_imgs)} val")

# -----------------------------
def main():
    ensure_dirs()
    download_kaggle_dataset()
    class_images = collect_images()
    split_and_copy(class_images)

    print("\n✅ Dataset ready")
    print(f"Train samples: {sum(len(list((TRAIN_DIR / c).iterdir())) for c in ['0','1'])}")
    print(f"Val samples:   {sum(len(list((VAL_DIR / c).iterdir())) for c in ['0','1'])}")

# -----------------------------
if __name__ == "__main__":
    main()
