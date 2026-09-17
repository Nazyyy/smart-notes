# ### FILE: scripts/prepare_dataset.py
"""
Data Preparation Pipeline for Handwritten Text Recognition (HTR).
Extracts, preprocesses, and normalizes line/word crops from School Notebooks datasets
and generates complementary synthetic cursive samples.
"""

import json
import os
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.ml.vocab import VOCAB


def get_available_fonts() -> List[str]:
    """Find installed TrueType / OpenType fonts supporting Latin & Cyrillic."""
    candidate_paths = [
        "/usr/share/fonts/TTF/Comic.TTF",
        "/usr/share/fonts/TTF/OpenSans-CondensedExtraBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSerif-BoldItalic.ttf",
        "/home/dima/.fonts/Roboto/Roboto-BoldItalic.ttf",
        "/usr/share/fonts/TTF/JetBrainsMono-BoldItalic.ttf",
        "/usr/share/fonts/noto/NotoSerif-Regular.ttf",
    ]
    valid_fonts = [p for p in candidate_paths if os.path.exists(p)]
    if not valid_fonts:
        # Fallback to any ttf on system
        import glob
        valid_fonts = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)[:5]
    return valid_fonts


def generate_synthetic_sample(
    text: str,
    font_path: str,
    font_size: int = 28,
    target_height: int = 32,
    target_width: int = 256,
) -> np.ndarray:
    """Generate an augmented synthetic text image simulating handwritten ink on notebook paper."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Measure text bounding box
    dummy_img = Image.new("L", (10, 10), color=255)
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])

    # Canvas dimensions
    canvas_w = max(text_w + 20, 64)
    canvas_h = max(text_h + 10, target_height)

    # Slight off-white background with subtle paper noise
    bg_val = random.randint(235, 255)
    img = Image.new("L", (canvas_w, canvas_h), color=bg_val)
    draw = ImageDraw.Draw(img)

    # Dark ink with slight color variation (black, dark blue, pencil gray)
    ink_val = random.randint(15, 60)
    draw.text((10 - bbox[0], (canvas_h - text_h) // 2 - bbox[1]), text, font=font, fill=ink_val)

    # Convert to OpenCV numpy array
    np_img = np.array(img, dtype=np.uint8)

    # Augmentations: Random Gaussian blur, salt-and-pepper ink noise, slight rotation
    if random.random() > 0.4:
        np_img = cv2.GaussianBlur(np_img, (3, 3), sigmaX=random.uniform(0.3, 0.8))

    # Random stroke dilation or erosion (simulating pen pressure)
    r_aug = random.random()
    if r_aug < 0.2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        np_img = cv2.erode(np_img, kernel, iterations=1)
    elif r_aug < 0.4:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        np_img = cv2.dilate(np_img, kernel, iterations=1)

    # Perspective shear / slant (italic tilt)
    if random.random() > 0.5:
        shear = random.uniform(-0.25, 0.25)
        h, w = np_img.shape
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        np_img = cv2.warpAffine(np_img, M, (w, h), borderValue=255)

    # Resize preserving aspect ratio to target_height (32) and pad to target_width (256)
    h, w = np_img.shape
    scale = target_height / float(h)
    new_w = max(1, min(int(w * scale), target_width))
    resized = cv2.resize(np_img, (new_w, target_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_height, target_width), 255, dtype=np.uint8)
    canvas[:, :new_w] = resized
    return canvas


def extract_crops_from_dataset(
    annotations_json_path: Path,
    images_dir: Path,
    output_dir: Path,
    lang_tag: str,
    max_samples: int = 50000,
) -> List[Dict[str, Any]]:
    """Extract annotated polygon crops from school notebooks COCO format."""
    if not annotations_json_path.exists():
        print(f"[-] Annotation file missing: {annotations_json_path}")
        return []

    print(f"[*] Parsing annotations: {annotations_json_path}")
    with open(annotations_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build image map
    img_map = {im["id"]: im["file_name"] for im in coco.get("images", [])}
    annotations = coco.get("annotations", [])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-cache image file locations
    available_image_files = {}
    for p in images_dir.glob("**/*"):
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            available_image_files[p.name] = p

    print(f"[*] Found {len(available_image_files)} source page images on disk.")

    records = []
    current_img_id = None
    loaded_img = None

    for ann in annotations:
        if len(records) >= max_samples:
            break

        trans = ann.get("attributes", {}).get("translation", "") if ann.get("attributes") else ""
        if not trans:
            continue
        trans = trans.strip()
        if len(trans) == 0:
            continue

        # Filter characters to ensure compatibility with model vocabulary
        clean_text = "".join(ch for ch in trans if ch in VOCAB.char_to_id)
        if len(clean_text) == 0:
            continue

        image_id = ann.get("image_id")
        file_name = img_map.get(image_id)
        if not file_name or file_name not in available_image_files:
            continue

        # Load image if changed
        if image_id != current_img_id:
            current_img_id = image_id
            loaded_img = cv2.imread(str(available_image_files[file_name]))

        if loaded_img is None:
            continue

        h_img, w_img = loaded_img.shape[:2]
        seg = ann.get("segmentation", [])
        if not seg or len(seg[0]) < 6:
            continue

        try:
            pts = np.array(seg[0]).reshape(-1, 2)
            x_min = max(0, int(np.floor(np.min(pts[:, 0]))) - 3)
            y_min = max(0, int(np.floor(np.min(pts[:, 1]))) - 2)
            x_max = min(w_img, int(np.ceil(np.max(pts[:, 0]))) + 3)
            y_max = min(h_img, int(np.ceil(np.max(pts[:, 1]))) + 2)

            crop_w = x_max - x_min
            crop_h = y_max - y_min
            if crop_w < 8 or crop_h < 8:
                continue

            crop = loaded_img[y_min:y_max, x_min:x_max]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

            # Normalize to 32 x 256
            scale = 32.0 / float(crop_h)
            new_w = max(1, min(int(crop_w * scale), 256))
            resized = cv2.resize(gray_crop, (new_w, 32), interpolation=cv2.INTER_AREA)

            canvas = np.full((32, 256), 255, dtype=np.uint8)
            canvas[:, :new_w] = resized

            crop_id = f"{lang_tag}_{len(records):06d}.png"
            crop_path = output_dir / crop_id
            cv2.imwrite(str(crop_path), canvas)

            records.append({
                "image_path": str(crop_path.resolve()),
                "text": clean_text,
                "lang": lang_tag,
            })

            if len(records) % 2000 == 0:
                print(f"    [+] Extracted {len(records)} crops ({lang_tag})...")

        except Exception:
            continue

    print(f"[+] Total crops extracted from {lang_tag}: {len(records)}")
    return records


def generate_supplementary_synthetic_data(
    output_dir: Path,
    num_samples: int = 15000,
) -> List[Dict[str, Any]]:
    """Generate synthetic words and phrases covering math, punctuation, numbers, and core lexicon."""
    fonts = get_available_fonts()
    if not fonts:
        print("[-] No TrueType fonts found for synthetic generation.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Generating {num_samples} supplementary synthetic handwriting samples...")

    # Lexicon pool: Math, English words, Russian words, formulas, numbers
    sample_phrases = [
        "Qwe", "qwe", "Hello", "world", "Lecture", "Math", "Analysis", "Theorem",
        "Function", "x", "y", "z", "f(x)", "y = 2x + 1", "E = mc^2", "a + b = c",
        "f(x) = sin(x)", "\\int f(x) dx", "x_1, x_2", "S = \\pi r^2", "a^2 + b^2",
        "12345", "67890", "2026", "N1", "N2", "N3", "1.1", "2.5",
        "Лекция", "Тема", "Теорема", "Определение", "Формула", "Свойство",
        "Интеграл", "Производная", "Матрица", "Вектор", "Предел", "Сумма",
        "конспект", "задача", "решение", "ответ", "пример", "доказательство",
        "рукописный", "текст", "студент", "школа", "класс", "тетрадь",
        "непрерывна", "отрезок", "число", "график", "уравнение", "система",
        "x > 0", "x <= 1", "t = 0", "k = 1", "n -> \\infty", "(a + b)",
        "[a, b]", "{1, 2, 3}", "dx/dt", "dy/dx", "log(x)", "ln(x)",
    ]

    records = []
    for i in range(num_samples):
        # Pick text: randomly from phrases, or combine numbers/words
        choice = random.random()
        if choice < 0.6:
            text = random.choice(sample_phrases)
        elif choice < 0.8:
            # Random number or formula
            num = random.randint(0, 99999)
            text = f"{random.choice(['№', 'No.', 'x=', 'y=', 'n='])}{num}"
        else:
            # Pair of words
            w1 = random.choice(sample_phrases)
            w2 = random.choice(sample_phrases)
            text = f"{w1} {w2}"[:25]

        # Clean text by vocab
        clean_text = "".join(ch for ch in text if ch in VOCAB.char_to_id).strip()
        if not clean_text:
            continue

        font_path = random.choice(fonts)
        font_size = random.randint(22, 30)
        img = generate_synthetic_sample(clean_text, font_path, font_size=font_size)

        sample_name = f"synth_{i:06d}.png"
        sample_path = output_dir / sample_name
        cv2.imwrite(str(sample_path), img)

        records.append({
            "image_path": str(sample_path.resolve()),
            "text": clean_text,
            "lang": "synth",
        })

        if (i + 1) % 5000 == 0:
            print(f"    [+] Generated {i + 1} / {num_samples} synthetic samples...")

    print(f"[+] Total synthetic samples generated: {len(records)}")
    return records


def build_full_dataset(
    datasets_root: Path = Path("data/datasets/school_notebooks"),
    output_root: Path = Path("data/processed_htr"),
):
    """Orchestrate extraction of EN, RU school notebooks and synthetic data."""
    output_crops_dir = output_root / "crops"
    output_crops_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    # 1. Extract EN School Notebooks
    en_ann_train = datasets_root / "en" / "annotations_train.json"
    en_images = datasets_root / "en" / "images"
    en_records = extract_crops_from_dataset(en_ann_train, en_images, output_crops_dir, "en")
    all_records.extend(en_records)

    # 2. Extract RU School Notebooks (Val & Train if downloaded)
    ru_ann_val = datasets_root / "ru" / "annotations_val.json"
    if not ru_ann_val.exists():
        ru_ann_val = datasets_root / "annotations_val.json"
    ru_ann_train = datasets_root / "ru" / "annotations_train.json"
    ru_images = datasets_root / "ru" / "images"

    if ru_images.exists():
        if ru_ann_train.exists():
            ru_records = extract_crops_from_dataset(ru_ann_train, ru_images, output_crops_dir, "ru", max_samples=40000)
            all_records.extend(ru_records)
        elif ru_ann_val.exists():
            ru_records = extract_crops_from_dataset(ru_ann_val, ru_images, output_crops_dir, "ru", max_samples=25000)
            all_records.extend(ru_records)

    # 3. Generate complementary synthetic handwritten samples
    synth_records = generate_supplementary_synthetic_data(output_crops_dir, num_samples=12000)
    all_records.extend(synth_records)

    random.shuffle(all_records)
    print(f"\n==========================================")
    print(f"[+] COMPLETE DATASET BUILT: {len(all_records)} samples")
    print(f"==========================================")

    # Split train (90%) and val (10%)
    split_idx = int(len(all_records) * 0.9)
    train_data = all_records[:split_idx]
    val_data = all_records[split_idx:]

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"train": train_data, "val": val_data}, f, ensure_ascii=False, indent=2)

    print(f"[+] Saved manifest: {manifest_path} (Train: {len(train_data)}, Val: {len(val_data)})")


if __name__ == "__main__":
    build_full_dataset()
