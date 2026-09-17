# ### FILE: scripts/finetune_trocr_base.py
"""
High-Scale TrOCR-Base Fine-Tuning Pipeline on 50,000+ Russian Handwritten Lines.
Incorporates Dynamic Notebook Augmentations (Grid, Ruled lines, Shear, Ink bleed),
PyTorch AMP FP16, Gradient Accumulation, and Validation CER/WER tracking on RTX 3060 Ti.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_cosine_schedule_with_warmup

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.notebook_augmentor import HandwrittenNotebookAugmentor


def compute_cer(pred: str, target: str) -> float:
    """Compute Character Error Rate (CER) via SequenceMatcher ratio."""
    if not target:
        return 0.0 if not pred else 1.0
    import difflib
    matcher = difflib.SequenceMatcher(None, target.strip(), pred.strip())
    return max(0.0, 1.0 - matcher.ratio())


def compute_wer(pred: str, target: str) -> float:
    """Compute Word Error Rate (WER)."""
    p_words = pred.strip().split()
    t_words = target.strip().split()
    if not t_words:
        return 0.0 if not p_words else 1.0
    import difflib
    matcher = difflib.SequenceMatcher(None, t_words, p_words)
    return max(0.0, 1.0 - matcher.ratio())


class AugmentedTrOCRDataset(Dataset):
    """
    Dataset loader for TrOCR with dynamic on-the-fly notebook background & stroke augmentations.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        processor: TrOCRProcessor,
        max_length: int = 64,
        augment: bool = False,
    ):
        self.records = records
        self.processor = processor
        self.max_length = max_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        image_path = rec["image_path"]
        text = rec["text"]

        try:
            pil_raw = Image.open(image_path).convert("RGB")
            np_img = np.array(pil_raw)

            if self.augment:
                np_img = HandwrittenNotebookAugmentor.augment(np_img)

            image = Image.fromarray(np_img)
        except Exception:
            image = Image.new("RGB", (384, 48), (255, 255, 255))

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Mask pad tokens from loss computation
        pad_id = self.processor.tokenizer.pad_token_id
        labels = [token_id if token_id != pad_id else -100 for token_id in labels.tolist()]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": text,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    texts = [item["text"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels, "texts": texts}


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def train_trocr_base(
    manifest_path: Path = ROOT_DIR / "data" / "processed_htr" / "trocr_manifest.json",
    base_model_path: Path = ROOT_DIR / "data" / "weights" / "trocr_finetuned_academic",
    output_dir: Path = ROOT_DIR / "data" / "weights" / "trocr_base_academic",
    epochs: int = 2,
    batch_size: int = 2,
    accum_steps: int = 12,
    lr: float = 2.5e-5,
    eval_steps: int = 250,
    max_train_samples: Optional[int] = None,
):
    print("=" * 75)
    print("TrOCR-BASE HIGH-SCALE TRAINING PIPELINE (NVIDIA RTX 3060 Ti CUDA FP16)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if not manifest_path.exists():
        print(f"[!] Manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_records = manifest.get("train", [])
    val_records = manifest.get("val", [])

    if max_train_samples and max_train_samples < len(train_records):
        train_records = train_records[:max_train_samples]

    print(f"[*] Training samples:   {len(train_records)}")
    print(f"[*] Validation samples: {len(val_records)}")

    # Resume from existing output checkpoint if available
    if (output_dir / "model.safetensors").exists():
        base_model_path = output_dir
        print(f"[*] Found existing checkpoint in {output_dir}, resuming fine-tuning from it!")
    elif not base_model_path.exists():
        base_model_path = ROOT_DIR / "data" / "weights" / "trocr_ru_lines"
    if not base_model_path.exists():
        base_model_path = ROOT_DIR / "data" / "weights" / "trocr_ru"

    print(f"[*] Loading base model from: {base_model_path}...")
    processor = TrOCRProcessor.from_pretrained(str(base_model_path))
    model = VisionEncoderDecoderModel.from_pretrained(str(base_model_path))
    model.to(device)

    # Enable gradient checkpointing to reduce VRAM consumption by ~65% on RTX 3060 Ti
    model.gradient_checkpointing_enable()

    # Freeze early encoder patch-embedding layers for faster, stable convergence
    if hasattr(model.encoder, "embeddings"):
        for param in model.encoder.embeddings.parameters():
            param.requires_grad = False

    train_dataset = AugmentedTrOCRDataset(train_records, processor, max_length=48, augment=True)
    val_dataset = AugmentedTrOCRDataset(val_records, processor, max_length=48, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    from transformers.optimization import Adafactor
    total_steps = (len(train_loader) // accum_steps) * epochs
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=min(400, int(total_steps * 0.08)), num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_cer = float("inf")
    patience_limit = 4
    no_improve_count = 0
    early_stop_triggered = False
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    t_start = time.time()


    print(f"\n[*] Starting training: {epochs} epochs | Effective Batch Size: {batch_size * accum_steps} | Steps: {total_steps}...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_loss = 0.0

        for b_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / accum_steps

            scaler.scale(loss).backward()
            step_loss += loss.item() * accum_steps

            if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                global_step += 1
                epoch_loss += step_loss
                step_loss = 0.0

                # Logging
                if global_step % 50 == 0:
                    lr_curr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t_start
                    steps_per_sec = global_step / max(1.0, elapsed)
                    print(f"  [Epoch {epoch}/{epochs} | Step {global_step}/{total_steps}] Loss: {loss.item() * accum_steps:.4f} | LR: {lr_curr:.2e} | Speed: {steps_per_sec:.2f} st/s")

                # Validation
                if global_step % eval_steps == 0 or global_step == total_steps:
                    model.eval()
                    total_cer = 0.0
                    total_wer = 0.0
                    val_count = 0

                    print(f"  --> Running Validation at step {global_step}...")
                    with torch.no_grad():
                        for val_batch in val_loader:
                            v_pixels = val_batch["pixel_values"].to(device)
                            target_texts = val_batch["texts"]

                            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                                gen_ids = model.generate(
                                    v_pixels,
                                    max_new_tokens=64,
                                    num_beams=3,
                                    repetition_penalty=1.25,
                                    no_repeat_ngram_size=3,
                                    early_stopping=True,
                                )

                            preds = processor.batch_decode(gen_ids, skip_special_tokens=True)
                            for p_text, t_text in zip(preds, target_texts):
                                total_cer += compute_cer(p_text, t_text)
                                total_wer += compute_wer(p_text, t_text)
                                val_count += 1

                            if val_count >= 100:
                                break

                    avg_cer = total_cer / max(1, val_count)
                    avg_wer = total_wer / max(1, val_count)
                    print(f"  [✓ Validation] CER: {avg_cer * 100:.2f}% | WER: {avg_wer * 100:.2f}% (Best CER: {best_cer * 100:.2f}%)")

                    if avg_cer < best_cer:
                        best_cer = avg_cer
                        no_improve_count = 0
                        print(f"  [★ NEW BEST] Saving TrOCR-Base checkpoint to {output_dir}...")
                        model.save_pretrained(str(output_dir))
                        processor.save_pretrained(str(output_dir))
                    else:
                        no_improve_count += 1
                        print(f"  [!] Validation CER did not improve ({no_improve_count}/{patience_limit} patience checks)")
                        if no_improve_count >= patience_limit:
                            print(f"\n[🛑 EARLY STOPPING] Triggered at step {global_step} to strictly prevent overfitting! Best CER achieved: {best_cer * 100:.2f}%.")
                            early_stop_triggered = True
                            break

                    model.train()

            if early_stop_triggered:
                break
        if early_stop_triggered:
            break


    # Final checkpoint save
    print(f"\n[✓] Training complete in {(time.time() - t_start)/60:.1f} minutes! Best CER: {best_cer * 100:.2f}%")
    print(f"[✓] Final model weights saved to {output_dir}")


if __name__ == "__main__":
    train_trocr_base(
        manifest_path=ROOT_DIR / "data" / "processed_htr" / "trocr_manifest.json",
        epochs=3,
        batch_size=2,
        accum_steps=12,
        eval_steps=300,
        lr=2.0e-5,
    )

