# ### FILE: scripts/finetune_trocr.py
"""
Production TrOCR Fine-Tuning Pipeline for Russian and Academic Handwritten Text.
Fine-tunes VisionEncoderDecoderModel (ViT + RoBERTa) on RTX 3060 Ti GPU (CUDA FP16).
Features AMP Mixed Precision, Gradient Accumulation, and Validation CER Evaluation.
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


def compute_cer(pred: str, target: str) -> float:
    """Compute Character Error Rate (CER) via Levenshtein distance."""
    if not target:
        return 0.0 if not pred else 1.0
    import difflib
    matcher = difflib.SequenceMatcher(None, target, pred)
    # Edit distance approximation via difflib
    return 1.0 - matcher.ratio()


class TrOCRDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], processor: TrOCRProcessor, max_length: int = 64):
        self.records = records
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        image_path = rec["image_path"]
        text = rec["text"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # Fallback blank image on read failure
            image = Image.new("RGB", (384, 48), (255, 255, 255))

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token ids with -100 to ignore in CrossEntropyLoss
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


def train_trocr(
    manifest_path: Path = ROOT_DIR / "data" / "processed_htr" / "trocr_manifest.json",
    base_model_path: Path = ROOT_DIR / "data" / "weights" / "trocr_ru_lines",
    output_dir: Path = ROOT_DIR / "data" / "weights" / "trocr_finetuned_academic",
    epochs: int = 3,
    batch_size: int = 8,
    accum_steps: int = 2,
    lr: float = 3e-5,
    eval_steps: int = 300,
    max_train_samples: Optional[int] = None,
):
    print("=" * 70)
    print("TrOCR FINE-TUNING PIPELINE (NVIDIA RTX 3060 Ti CUDA)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if not manifest_path.exists():
        print(f"[-] Manifest not found: {manifest_path}. Running prepare_trocr_dataset first...")
        from scripts.prepare_trocr_dataset import build_trocr_dataset
        build_trocr_dataset()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_records = manifest.get("train", [])
    val_records = manifest.get("val", [])

    if max_train_samples and len(train_records) > max_train_samples:
        train_records = train_records[:max_train_samples]

    print(f"[*] Loaded {len(train_records)} training samples and {len(val_records)} validation samples.")

    # Load Base Model & Processor
    model_source = str(base_model_path) if base_model_path.exists() else "raxtemur/trocr-base-ru"
    print(f"[*] Loading model from {model_source}...")
    processor = TrOCRProcessor.from_pretrained(model_source)
    model = VisionEncoderDecoderModel.from_pretrained(model_source)

    # Configure special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # Freeze entire ViT encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Freeze token/position embeddings and lower 6 decoder layers to focus training on cross-attention and upper LM
    if hasattr(model.decoder, "model") and hasattr(model.decoder.model, "decoder"):
        dec = model.decoder.model.decoder
        if hasattr(dec, "embed_tokens"):
            for p in dec.embed_tokens.parameters():
                p.requires_grad = False
        if hasattr(dec, "embed_positions"):
            for p in dec.embed_positions.parameters():
                p.requires_grad = False
        if hasattr(dec, "layers"):
            for layer in dec.layers[:6]:
                for p in layer.parameters():
                    p.requires_grad = False
        print("[*] Frozen ViT encoder + lower 6 decoder layers. Training top 6 decoder layers & cross-attention.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

    model.to(device)

    train_dataset = TrOCRDataset(train_records, processor)
    val_dataset = TrOCRDataset(val_records, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        foreach=False,
    )

    total_steps = (len(train_loader) // accum_steps) * epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    output_dir.mkdir(parents=True, exist_ok=True)
    best_cer = 1.0
    global_step = 0

    print(f"[*] Starting training: {epochs} epochs, {len(train_loader)} batches/epoch, total steps: {total_steps}...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accum_steps

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % 50 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    cur_loss = epoch_loss / (batch_idx + 1)
                    print(f"  Epoch {epoch}/{epochs} | Step {global_step}/{total_steps} | Loss: {cur_loss:.4f} | LR: {current_lr:.2e}")

                if global_step % eval_steps == 0 or global_step == total_steps:
                    # Quick validation evaluation
                    model.eval()
                    val_cer_scores = []
                    sample_preds = []

                    with torch.no_grad():
                        for v_idx, val_batch in enumerate(val_loader):
                            if v_idx >= 15 and global_step < total_steps:
                                break
                            v_pixels = val_batch["pixel_values"].to(device)
                            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                                gen_ids = model.generate(v_pixels, max_new_tokens=64, num_beams=1)
                            preds = processor.batch_decode(gen_ids, skip_special_tokens=True)
                            for p_txt, t_txt in zip(preds, val_batch["texts"]):
                                cer = compute_cer(p_txt, t_txt)
                                val_cer_scores.append(cer)
                                if len(sample_preds) < 4:
                                    sample_preds.append((t_txt, p_txt, cer))

                    mean_cer = float(np.mean(val_cer_scores)) if val_cer_scores else 1.0
                    print(f"\n[>>> EVALUATION STEP {global_step} <<<] Validation CER: {mean_cer * 100:.2f}% (Best: {best_cer * 100:.2f}%)")
                    for t_txt, p_txt, c_val in sample_preds:
                        print(f"    GT:   '{t_txt}'")
                        print(f"    PRED: '{p_txt}' (CER: {c_val * 100:.1f}%)")
                    print("-" * 50)

                    if mean_cer < best_cer:
                        best_cer = mean_cer
                        print(f"[+] New best CER {best_cer * 100:.2f}%! Saving checkpoint to {output_dir}...")
                        model.save_pretrained(str(output_dir))
                        processor.save_pretrained(str(output_dir))
                        print("[✓] Model checkpoint and processor saved.")

                    model.train()

        elapsed_epoch = time.time() - t_epoch_start
        print(f"\n[✓] Epoch {epoch}/{epochs} finished in {elapsed_epoch / 60:.1f}m | Avg Loss: {epoch_loss / len(train_loader):.4f}")

    print("\n" + "=" * 70)
    print(f"[✓] FINE-TUNING COMPLETE! Best Validation CER: {best_cer * 100:.2f}%")
    print(f"[✓] Saved model to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TrOCR Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--eval-steps", type=int, default=150)
    args = parser.parse_args()

    train_trocr(
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        eval_steps=args.eval_steps,
        max_train_samples=args.max_samples,
    )
