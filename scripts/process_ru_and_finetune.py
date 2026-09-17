# ### FILE: scripts/process_ru_and_finetune.py
"""
Automated Pipeline: Unpacks Russian School Notebooks, extracts handwriting crops,
merges datasets into unified manifest, and fine-tunes the CRNN model on CUDA.
"""

import json
import os
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.prepare_dataset import extract_crops_from_dataset, build_full_dataset
from scripts.train_htr import train_model


def main():
    ru_dir = ROOT_DIR / "data" / "datasets" / "school_notebooks" / "ru"
    ru_zip = ru_dir / "images.zip"
    images_dir = ru_dir / "images"

    if not ru_zip.exists():
        print(f"[-] {ru_zip} does not exist yet.")
        return

    print(f"[*] Unpacking {ru_zip} ({ru_zip.stat().st_size / (1024*1024):.1f} MB)...")
    images_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ru_zip, "r") as z:
        z.extractall(images_dir)
    print(f"[+] Russian images extracted successfully to: {images_dir}")

    print("[*] Rebuilding full dataset with Russian, English, and synthetic handwriting...")
    build_full_dataset(
        datasets_root=ROOT_DIR / "data" / "datasets" / "school_notebooks",
        output_root=ROOT_DIR / "data" / "processed_htr",
    )

    print("\n[*] Launching fine-tuning on unified dataset (CUDA RTX 3060 Ti)...")
    train_model(epochs=10, batch_size=64, learning_rate=3e-4)
    print("\n[✓] ALL STAGES COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
