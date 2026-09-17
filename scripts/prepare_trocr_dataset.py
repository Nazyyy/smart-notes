# ### FILE: scripts/prepare_trocr_dataset.py
"""
High-Performance TrOCR Dataset Builder.
Extracts real handwritten line crops from Russian School Notebooks COCO annotations,
synthesizes domain academic phrases (Medicine, Biology, Chemistry, Physics),
and builds train/val manifests for seq2seq VisionEncoderDecoder fine-tuning.
"""

import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.prepare_dataset import get_available_fonts


def clean_transcription(text: str) -> str:
    """Normalize text and strip invalid characters while preserving Russian, Latin, and punctuation."""
    if not text:
        return ""
    # Strip carriage returns and excessive whitespace
    s = re.sub(r'[\r\n\t]+', ' ', text).strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def generate_synthetic_academic_crop(
    text: str,
    font_path: str,
    font_size: int = 30,
) -> np.ndarray:
    """Generate realistic synthetic handwriting crop for domain terms."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])

    canvas_w = text_w + 30
    canvas_h = max(text_h + 20, 36)

    # Notebook background tint (subtle off-white / light cream)
    bg_color = (random.randint(245, 255), random.randint(245, 255), random.randint(240, 250))
    img = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Ink color: ballpoint blue, gel black, dark graphite
    ink_choice = random.random()
    if ink_choice < 0.6:  # Blue ballpoint
        ink_color = (random.randint(15, 35), random.randint(30, 60), random.randint(110, 160))
    elif ink_choice < 0.85:  # Dark black / charcoal
        v = random.randint(20, 50)
        ink_color = (v, v, v)
    else:  # Violet / dark purple
        ink_color = (random.randint(70, 110), random.randint(20, 50), random.randint(110, 150))

    draw.text((15 - bbox[0], (canvas_h - text_h) // 2 - bbox[1]), text, font=font, fill=ink_color)
    np_img = np.array(img, dtype=np.uint8)

    # Random slant/shear
    if random.random() > 0.4:
        shear = random.uniform(-0.2, 0.2)
        h, w = np_img.shape[:2]
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        np_img = cv2.warpAffine(np_img, M, (w, h), borderValue=bg_color)

    # Mild blur / ink bleed
    if random.random() > 0.5:
        np_img = cv2.GaussianBlur(np_img, (3, 3), sigmaX=random.uniform(0.3, 0.7))

    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)


def build_trocr_dataset(
    max_real_samples: int = 20000,
    synthetic_samples: int = 4000,
    output_dir: Path = ROOT_DIR / "data" / "processed_htr",
) -> Dict[str, Any]:
    """
    Extract multi-word lines and words from Russian school notebook dataset,
    supplement with academic synthetic lines, and generate JSON manifest.
    """
    annotations_path = ROOT_DIR / "data" / "datasets" / "school_notebooks" / "ru" / "annotations_train.json"
    images_dir = ROOT_DIR / "data" / "datasets" / "school_notebooks" / "ru" / "images" / "images"
    crops_dir = output_dir / "trocr_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading annotations: {annotations_path}...")
    with open(annotations_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_map = {im["id"]: im for im in coco.get("images", [])}
    annotations = coco.get("annotations", [])

    print(f"[*] Grouping annotations into lines across {len(img_map)} images...")
    img_groups: Dict[int, Dict[Any, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    isolated_words: List[Dict[str, Any]] = []

    for ann in annotations:
        trans = ann.get("attributes", {}).get("translation", "")
        if not trans or not trans.strip():
            continue

        img_id = ann.get("image_id")
        gid = ann.get("group_id")
        if gid is not None:
            img_groups[img_id][gid].append(ann)
        else:
            isolated_words.append(ann)

    print(f"[*] Found {sum(len(g) for g in img_groups.values())} grouped lines and {len(isolated_words)} isolated words.")

    # Cache image file paths
    image_paths_cache = {p.name: p for p in images_dir.glob("*.*") if p.is_file()}
    print(f"[*] Verified {len(image_paths_cache)} page images on disk.")

    records: List[Dict[str, Any]] = []
    t0 = time.time()

    # 1. Process multi-word grouped lines
    print(f"[*] Extracting up to {max_real_samples} line crops from notebook pages...")
    for img_id, groups in img_groups.items():
        if len(records) >= max_real_samples:
            break

        img_info = img_map.get(img_id)
        if not img_info:
            continue

        file_name = img_info["file_name"]
        full_img_path = image_paths_cache.get(file_name)
        if not full_img_path or not full_img_path.exists():
            continue

        page_bgr = cv2.imread(str(full_img_path))
        if page_bgr is None:
            continue
        page_h, page_w = page_bgr.shape[:2]

        for gid, word_anns in groups.items():
            if len(records) >= max_real_samples:
                break

            # Collect line bounding box and words
            parsed_words = []
            all_xs = []
            all_ys = []

            for w_ann in word_anns:
                trans = clean_transcription(w_ann.get("attributes", {}).get("translation", ""))
                if not trans:
                    continue
                seg = w_ann.get("segmentation", [[]])[0]
                if len(seg) < 6:
                    continue
                xs = seg[0::2]
                ys = seg[1::2]
                min_x = min(xs)
                all_xs.extend(xs)
                all_ys.extend(ys)
                parsed_words.append((min_x, trans))

            if not parsed_words or len(all_xs) < 3:
                continue

            parsed_words.sort(key=lambda item: item[0])
            line_text = " ".join(item[1] for item in parsed_words).strip()
            if len(line_text) < 2 or not re.search(r'[А-Яа-яA-Za-z0-9]', line_text):
                continue

            # Bounding box with safety margin
            x1 = max(0, int(np.floor(min(all_xs))) - 6)
            y1 = max(0, int(np.floor(min(all_ys))) - 4)
            x2 = min(page_w, int(np.ceil(max(all_xs))) + 6)
            y2 = min(page_h, int(np.ceil(max(all_ys))) + 4)

            w = x2 - x1
            h = y2 - y1
            if w < 24 or h < 10:
                continue

            crop = page_bgr[y1:y2, x1:x2]
            crop_filename = f"line_{len(records):06d}.jpg"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])

            records.append({
                "image_path": str(crop_path.resolve()),
                "text": line_text,
                "source": "school_notebooks_ru",
            })

            if len(records) % 2500 == 0:
                elapsed = time.time() - t0
                print(f"    [+] Extracted {len(records)} line crops ({elapsed:.1f}s)...")

    print(f"[+] Total real line crops extracted: {len(records)}")

    # 2. Add domain-specific synthetic academic lines
    print(f"[*] Generating {synthetic_samples} synthetic academic lines (Medicine, Chemistry, Biology)...")
    fonts = get_available_fonts()
    academic_phrases = [
        "Не более 10 мл",
        "Если ускорить - кладем грелку",
        "замедлить - пузырь со льдом",
        "Внутривенно:",
        "Болюсное: внутривенно-струйно",
        "Инфузионное (внутривенно капельно)",
        "Комбинированное",
        "Абсорбция (всасывание) - процесс поступления ЛС из",
        "места введения в кровеносную и/или лимфати-",
        "ческую систему через био. мембраны",
        "Основные пути всасывания:",
        "1. Пассивная диффузия - поступление в-в идет",
        "по градиенту концентрации",
        "2. Фильтрация - процесс поступления вещества",
        "через поры в мембране",
        "3. Пиноцитоз - процесс проникновения через мембрану",
        "с образованием вакуоли",
        "Антропогенез - теория возникновения",
        "Социогенез - теория становления",
        "и развитие человеческого общества",
        "• Антропосоциогенез - происхождение",
        "и развития человека и общества.",
        "• Ф. Энгельс - человека создал труд.",
        "Человек - биосоциальное существо,",
        "высшая ступень развития живых организмов.",
        "Биологическая | Социальная",
        "1. Анатомия | 1. Способность к общес-",
        "2. Физиология | твенно-полезному труду",
        "2. Сознание и разум",
        "3. Свобода и ответственность",
        "Фармакокинетика и фармакодинамика лекарственных средств",
        "Биодоступность препарата при пероральном введении",
        "Клиренс и период полувыведения t_1/2",
        "Гематоэнцефалический барьер и транспорт молекул",
        "2NaOH + H2SO4 = Na2SO4 + 2H2O",
        "CaCO3 + 2HCl = CaCl2 + H2O + CO2",
        "CH3-COOH + C2H5OH <=> CH3-COOC2H5 + H2O",
        "C6H12O6 + 6O2 -> 6CO2 + 6H2O + E",
        "pH = -log[H+] = 7.4",
        "dF/dt = m * a",
        "E = mc^2",
        "f(x) = sin(x) + cos(x)",
        "\\int_0^1 x^2 dx = 1/3",
        "Определение. Функция f(x) непрерывна на отрезке [a, b]",
        "Теорема Коши о среднем значении",
        "Упражнение 503. Классная работа.",
        "Двенадцатое апреля. Домашняя работа.",
    ]

    for i in range(synthetic_samples):
        text = random.choice(academic_phrases)
        font_path = random.choice(fonts) if fonts else None
        crop = generate_synthetic_academic_crop(text, font_path=font_path, font_size=random.randint(24, 32))

        synth_name = f"synth_acad_{i:05d}.jpg"
        synth_path = crops_dir / synth_name
        cv2.imwrite(str(synth_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])

        records.append({
            "image_path": str(synth_path.resolve()),
            "text": text,
            "source": "synthetic_academic",
        })

    random.shuffle(records)
    print(f"\n[+] Total combined dataset size: {len(records)} samples.")

    # Split into 95% train and 5% validation
    val_size = max(10, min(500, int(len(records) * 0.05)))
    val_data = records[:val_size]
    train_data = records[val_size:]

    manifest_path = output_dir / "trocr_manifest.json"
    manifest = {
        "train": train_data,
        "val": val_data,
        "metadata": {
            "total_samples": len(records),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[✓] Manifest saved to {manifest_path} (Train: {len(train_data)}, Val: {len(val_data)})")
    return manifest


if __name__ == "__main__":
    build_trocr_dataset(max_real_samples=16000, synthetic_samples=3000)
