# ### FILE: scripts/generate_synthetic_htr_dataset.py
"""
Script to generate synthetic Russian handwriting dataset with varied styles,
messy fonts, and paper grid backgrounds.
"""

import argparse
import json
from pathlib import Path
import cv2
from tqdm import tqdm

from app.ml.synthetic_generator import SyntheticHandwritingGenerator, ACADEMIC_CORPUS_TEMPLATES


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Cyrillic handwriting crops")
    parser.add_argument("--count", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_dataset", help="Output directory")
    parser.add_argument("--target_height", type=int, default=48, help="Target crop height in px")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    gen = SyntheticHandwritingGenerator()
    print(f"Loaded {len(gen.font_paths)} fonts from {gen.fonts_dir}")
    print(f"Generating {args.count} synthetic handwriting images in {out_dir}...")

    manifest_entries = []
    for i in tqdm(range(args.count), desc="Synthesizing"):
        img, text = gen.generate_line(target_height=args.target_height)
        img_name = f"synth_{i:06d}.jpg"
        img_path = img_dir / img_name
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        manifest_entries.append({
            "image_path": str(img_path.relative_to(out_dir)),
            "text": text,
            "height": img.shape[0],
            "width": img.shape[1],
        })

    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[✓] Done! Generated {len(manifest_entries)} images. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
