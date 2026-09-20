# ### FILE: scripts/train_messy_handwriting.py
"""
Adaptive Fine-Tuning of TrOCR-Base on Messy, Hasty, and Irregular Russian Student Handwriting.
Trains on-the-fly with elastic deformations, variable slant, baseline drift, and notebook grid lines.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.ml.synthetic_generator import SyntheticHandwritingGenerator, ACADEMIC_CORPUS_TEMPLATES
from scripts.notebook_augmentor import HandwrittenNotebookAugmentor


def compute_quick_cer(pred: str, target: str) -> float:
    """Compute character error rate."""
    if not target:
        return 0.0 if not pred else 1.0
    import difflib
    matcher = difflib.SequenceMatcher(None, target.strip(), pred.strip())
    return max(0.0, 1.0 - matcher.ratio())


def train_messy_handwriting(
    model_path: str = "./data/weights/trocr_base_academic",
    output_dir: str = "./data/weights/trocr_base_academic",
    max_steps: int = 120,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    lr: float = 3e-6,
):
    """
    Run targeted fine-tuning on challenging messy handwriting lines.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Messy Handwriting Fine-Tuning on {device} ===")

    # Initialize generator
    generator = SyntheticHandwritingGenerator()
    print(f"Loaded generator with {len(generator.font_paths)} handwriting fonts.")

    # Load model and processor
    print(f"Loading checkpoint from {model_path}...")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.train()

    # Freeze early layers to prevent catastrophic forgetting and comfortably fit in VRAM
    model.encoder.requires_grad_(False)
    model.encoder.encoder.layer[-1].requires_grad_(True)
    model.decoder.model.decoder.embed_tokens.requires_grad_(False)
    model.decoder.model.decoder.embed_positions.requires_grad_(False)
    for i in range(8):
        model.decoder.model.decoder.layers[i].requires_grad_(False)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Targeted fine-tuning parameters: {trainable_count:,} / {total_count:,} ({trainable_count*100/total_count:.1f}%)")

    # Enable gradient checkpointing to drastically reduce memory usage
    model.encoder.gradient_checkpointing_enable()
    model.decoder.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=max_steps,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    pad_id = processor.tokenizer.pad_token_id
    running_loss = 0.0
    step = 0
    t0 = time.time()

    print(f"Running {max_steps} optimization steps (effective batch size: {batch_size * grad_accum_steps})...")

    while step < max_steps:
        optimizer.zero_grad()

        for accum_idx in range(grad_accum_steps):
            # Synthesize messy batch on the fly
            batch_images = []
            batch_texts = []
            for _ in range(batch_size):
                text = random.choice(ACADEMIC_CORPUS_TEMPLATES)
                # Generate with messy handwriting mode + augmentations
                bgr_img, _ = generator.generate_line(
                    text=text,
                    target_height=48,
                    add_baseline_wave=True,
                    messy_handwriting=True,
                )
                aug_bgr = HandwrittenNotebookAugmentor.augment(bgr_img)
                rgb_img = cv2_to_pil(aug_bgr)
                batch_images.append(rgb_img)
                batch_texts.append(text)

            pixel_values = processor(batch_images, return_tensors="pt").pixel_values.to(device)
            encoded_labels = processor.tokenizer(
                batch_texts,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            labels = torch.where(encoded_labels == pad_id, -100, encoded_labels)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum_steps

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1

        if step % 20 == 0 or step == max_steps:
            avg_loss = running_loss / 20.0 if step % 20 == 0 else running_loss / (step % 20 or 1)
            running_loss = 0.0
            elapsed = time.time() - t0
            print(f"Step [{step}/{max_steps}] - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e} - Elapsed: {elapsed:.1f}s")

    # Validation check on 8 messy samples
    print("\n=== Validation on Messy Handwriting Samples ===")
    model.eval()
    val_cers: List[float] = []

    for val_idx in range(8):
        text = random.choice(ACADEMIC_CORPUS_TEMPLATES)
        bgr_img, _ = generator.generate_line(text=text, target_height=48, messy_handwriting=True)
        aug_bgr = HandwrittenNotebookAugmentor.augment(bgr_img)
        pil_img = cv2_to_pil(aug_bgr)

        pixel_val = processor([pil_img], return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_val, max_new_tokens=48)
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        cer = compute_quick_cer(pred_text, text)
        val_cers.append(cer)
        print(f"Sample {val_idx + 1}:")
        print(f"  Target: {text}")
        print(f"  Pred:   {pred_text}")
        print(f"  CER:    {cer * 100:.1f}%\n")

    mean_cer = float(np.mean(val_cers))
    print(f"Mean Validation CER on Messy Handwriting: {mean_cer * 100:.2f}%")

    # Save upgraded model
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving upgraded weights to {out_path}...")
    model.save_pretrained(str(out_path))
    processor.save_pretrained(str(out_path))
    print("=== Fine-Tuning Complete and Weights Saved Successfully ===")


def cv2_to_pil(bgr_img: np.ndarray) -> Image.Image:
    import cv2
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


if __name__ == "__main__":
    train_messy_handwriting(
        model_path="./data/weights/trocr_base_academic",
        output_dir="./data/weights/trocr_base_academic",
        max_steps=50,
        batch_size=1,
        grad_accum_steps=4,
    )
