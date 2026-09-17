import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from app.cv import (
    rectify_document_geometry,
    suppress_shadows_and_denoise,
    adaptive_binarize,
    segment_text_lines,
)
from app.ml.transformer_engine import TransformerHTREngine
from app.services.structurer_service import NoteStructurerService
from app.models.entities import TextLine
from uuid import uuid4

def evaluate_all():
    image_paths = [
        "/home/dima/.gemini/antigravity/brain/7677399d-423b-4618-963a-5d71fd8f882e/.user_uploaded/media_1789596090276.jpg",
        "/home/dima/.gemini/antigravity/brain/7677399d-423b-4618-963a-5d71fd8f882e/.user_uploaded/media_1789593160494.jpg",
        "/home/dima/.gemini/antigravity/brain/7677399d-423b-4618-963a-5d71fd8f882e/.user_uploaded/media_1789593160486.jpg",
        "/home/dima/.gemini/antigravity/brain/7677399d-423b-4618-963a-5d71fd8f882e/.user_uploaded/media_1789592366068.jpg",
    ]

    print("Initializing TransformerHTREngine...")
    engine = TransformerHTREngine()
    structurer = NoteStructurerService()

    for p_str in image_paths:
        p = Path(p_str)
        if not p.exists():
            print(f"Skipping {p.name}: file not found")
            continue

        raw_bgr = cv2.imread(str(p))
        h, w = raw_bgr.shape[:2]
        print("\n" + "="*70)
        print(f"EVALUATING IMAGE: {p.name} (Resolution: {w}x{h})")
        print("="*70)

        rectified_bgr, skew_angle = rectify_document_geometry(raw_bgr)
        shadow_suppressed = suppress_shadows_and_denoise(rectified_bgr)
        gray = cv2.cvtColor(shadow_suppressed, cv2.COLOR_BGR2GRAY)
        binary_mask = adaptive_binarize(gray)
        bboxes, crops, hpp = segment_text_lines(rectified_bgr, binary_mask)
        print(f"Detected {len(crops)} lines (Skew angle: {skew_angle:.2f}°)")

        predictions = engine.predict_batch(crops)

        line_entities = []
        confs = []
        print("\nRecognized lines:")
        for idx, ((bx, by, bw, bh), (text, conf)) in enumerate(zip(bboxes, predictions)):
            confs.append(conf)
            if text:
                print(f"  [{idx:02d}] ({conf*100:.1f}%) {text}")
            line_entities.append(
                TextLine(
                    page_id=uuid4(),
                    line_index=idx,
                    bbox_x=bx,
                    bbox_y=by,
                    bbox_w=bw,
                    bbox_h=bh,
                    cropped_image_path="",
                    recognized_text=text,
                    confidence=conf,
                )
            )

        avg_conf = (sum(confs) / len(confs)) * 100 if confs else 0.0
        print(f"\nAverage Line Confidence: {avg_conf:.1f}%")

        md = structurer.structure_lines_to_markdown(line_entities, f"Конспект {p.stem}")
        print("\nStructured Markdown Output Preview:")
        preview_lines = md.strip().split("\n")[:12]
        print("\n".join(preview_lines))

if __name__ == '__main__':
    evaluate_all()
