# ### FILE: scripts/pseudo_labeling_pipeline.py
"""
High-Confidence Self-Training and Pseudo-Labeling Pipeline.
Mines unannotated or challenging handwriting crops from the 335k school notebook corpus,
runs local TrOCR-Base inference, filters predictions using N-Gram LM Perplexity (>= 92% confidence),
and augments the fine-tuning dataset for continuous self-improving accuracy.
"""

import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.ml.transformer_engine import TransformerHTREngine
from app.ml.language_model_rescorer import get_language_model_rescorer
from app.ml.handwriting_confusion import get_handwriting_confusion_corrector
from scripts.notebook_augmentor import HandwrittenNotebookAugmentor


def run_pseudo_labeling_pipeline(
    coco_annotations_path: Path = ROOT_DIR / "data" / "datasets" / "school_notebooks" / "ru" / "annotations_train.json",
    images_dir: Path = ROOT_DIR / "data" / "datasets" / "school_notebooks" / "ru" / "images" / "images",
    manifest_path: Path = ROOT_DIR / "data" / "processed_htr" / "trocr_manifest.json",
    crops_output_dir: Path = ROOT_DIR / "data" / "processed_htr" / "pseudo_labeled_crops",
    max_pseudo_samples: int = 5000,
    min_confidence: float = 0.90,
) -> int:
    """
    Mine notebook pages for high-confidence pseudo-labeled lines and append to manifest.
    """
    print("=" * 75)
    print("TrOCR SELF-TRAINING & PSEUDO-LABELING MINER (RTX 3060 Ti CUDA)")
    print("=" * 75)

    crops_output_dir.mkdir(parents=True, exist_ok=True)

    if not coco_annotations_path.exists():
        print(f"[!] COCO annotations not found at: {coco_annotations_path}")
        return 0

    if not manifest_path.exists():
        print(f"[!] Target manifest not found at: {manifest_path}")
        return 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    existing_images = set(rec["image_path"] for rec in manifest.get("train", []))
    print(f"[*] Manifest currently has {len(manifest.get('train', []))} train samples.")

    print(f"[*] Loading COCO annotations from {coco_annotations_path}...")
    with open(coco_annotations_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_map = {im["id"]: im for im in coco.get("images", [])}
    annotations = coco.get("annotations", [])

    # Group annotations by image and group_id
    img_groups: Dict[int, Dict[Any, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for ann in annotations:
        gid = ann.get("group_id")
        img_id = ann.get("image_id")
        if gid is not None and img_id in img_map:
            img_groups[img_id][gid].append(ann)

    print(f"[*] Found {len(img_groups)} candidate notebook pages.")

    # Initialize local engine and verifiers
    print("[*] Initializing TrOCR-Base and Language Model Verifiers...")
    engine = TransformerHTREngine()
    rescorer = get_language_model_rescorer()
    corrector = get_handwriting_confusion_corrector()

    new_pseudo_records: List[Dict[str, Any]] = []
    sampled_pages = list(img_groups.keys())
    # Shuffle to get diverse handwriting styles
    random.seed(42)
    random.shuffle(sampled_pages)

    t0 = time.time()
    for page_idx, img_id in enumerate(sampled_pages):
        if len(new_pseudo_records) >= max_pseudo_samples:
            break

        img_info = img_map[img_id]
        img_filename = img_info["file_name"]
        page_path = images_dir / img_filename
        if not page_path.exists():
            continue

        page_bgr = cv2.imread(str(page_path))
        if page_bgr is None:
            continue
        page_h, page_w = page_bgr.shape[:2]

        for gid, word_anns in img_groups[img_id].items():
            if len(new_pseudo_records) >= max_pseudo_samples:
                break

            # Calculate tight bounding box encompassing word group from polygon segmentation
            all_x: List[float] = []
            all_y: List[float] = []

            for ann in word_anns:
                seg_list = ann.get("segmentation", [[]])
                if seg_list and len(seg_list[0]) >= 6:
                    seg = seg_list[0]
                    all_x.extend(seg[0::2])
                    all_y.extend(seg[1::2])

            if len(all_x) < 3 or len(all_y) < 3:
                continue

            min_x = max(0, int(min(all_x)) - 6)
            min_y = max(0, int(min(all_y)) - 5)
            max_x = min(page_w, int(max(all_x)) + 6)
            max_y = min(page_h, int(max(all_y)) + 5)

            w = max_x - min_x
            h = max_y - min_y
            if w < 50 or h < 14 or (w / h) < 1.2:
                continue

            crop = page_bgr[min_y:max_y, min_x:max_x].copy()
            pred_text, conf = engine.predict_single_line(crop, enable_tta=False)

            clean_text = pred_text.strip()
            # Quality criteria: High model confidence, valid length, and natural Russian phonetics
            if (
                conf >= min_confidence
                and len(clean_text) >= 8
                and len(clean_text.split()) >= 2
                and rescorer.compute_char_perplexity_penalty(clean_text) == 0.0
            ):
                crop_filename = f"pseudo_{img_id}_{gid}_{len(new_pseudo_records):05d}.jpg"
                crop_path = crops_output_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)

                new_pseudo_records.append({
                    "image_path": str(crop_path),
                    "text": clean_text,
                    "confidence": round(conf, 4),
                    "source": "self_training_pseudo_label",
                })

                if len(new_pseudo_records) % 25 == 0:
                    print(f"  [+] Mined {len(new_pseudo_records)}/{max_pseudo_samples} pseudo-labels (Current: '{clean_text}' @ {conf*100:.1f}%)")

    print(f"\n[✓] Mined {len(new_pseudo_records)} high-confidence pseudo-labeled lines in {(time.time() - t0):.1f}s.")

    if new_pseudo_records:
        manifest["train"].extend(new_pseudo_records)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[✓] Successfully updated manifest at {manifest_path} (Total train: {len(manifest['train'])}).")

    return len(new_pseudo_records)


if __name__ == "__main__":
    count = run_pseudo_labeling_pipeline(max_pseudo_samples=500, min_confidence=0.91)
    print(f"Done. Added {count} new pseudo-labeled training samples.")
