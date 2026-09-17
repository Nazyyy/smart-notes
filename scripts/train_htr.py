# ### FILE: scripts/train_htr.py
"""
High-Performance PyTorch Training Engine for Handwritten Text Recognition (HTR).
Utilizes Connectionist Temporal Classification (CTC) Loss, Mixed Precision (AMP),
and Character Error Rate (CER) validation tracking on NVIDIA CUDA GPU.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from app.ml.crnn_model import CRNN
from app.ml.vocab import VOCAB


class HTRDataset(Dataset):
    """PyTorch Dataset loading preprocessed 32x256 grayscale HTR line crops."""

    def __init__(self, samples: List[Dict[str, Any]], is_train: bool = True) -> None:
        self.samples = samples
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        item = self.samples[idx]
        img_path = item["image_path"]
        text = item["text"]

        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback blank canvas
            img = np.full((32, 256), 255, dtype=np.uint8)

        # Ensure correct shape (32, 256)
        h, w = img.shape
        if h != 32 or w != 256:
            scale = 32.0 / float(h)
            new_w = max(1, min(int(w * scale), 256))
            resized = cv2.resize(img, (new_w, 32), interpolation=cv2.INTER_AREA)
            canvas = np.full((32, 256), 255, dtype=np.uint8)
            canvas[:, :new_w] = resized
            img = canvas

        # Augmentations for training
        if self.is_train:
            # Random slight brightness / contrast
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-10, 10)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # Normalize to [-1.0, 1.0]
        img_tensor = (torch.from_numpy(img).float() / 127.5) - 1.0
        img_tensor = img_tensor.unsqueeze(0)  # Shape: (1, 32, 256)

        # Encode text label into CTC token indices
        tokens = VOCAB.encode(text)
        target_tensor = torch.tensor(tokens, dtype=torch.long)
        target_length = len(tokens)

        return img_tensor, target_tensor, target_length, text


def collate_fn(batch):
    """Collate function assembling variable-length targets for PyTorch CTCLoss."""
    images, targets, target_lengths, raw_texts = zip(*batch)

    # Stack images into tensor: (B, 1, 32, 256)
    images_batch = torch.stack(images, dim=0)

    # Concatenate targets into flat 1D tensor
    flat_targets = torch.cat(targets, dim=0)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)

    # Output time sequence length for W=256 is 64
    B = len(batch)
    input_lengths_tensor = torch.full((B,), 64, dtype=torch.long)

    return images_batch, flat_targets, input_lengths_tensor, target_lengths_tensor, raw_texts


def calculate_levenshtein(s1: str, s2: str) -> int:
    """Compute character edit distance between two strings."""
    if len(s1) < len(s2):
        return calculate_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def decode_greedy(log_probs: torch.Tensor) -> List[str]:
    """Greedy CTC Argmax decoder."""
    # log_probs: (T, B, num_classes)
    predictions = torch.argmax(log_probs, dim=2)  # (T, B)
    T, B = predictions.size()

    decoded_strings = []
    for b in range(B):
        tokens = []
        prev_idx = -1
        for t in range(T):
            idx = int(predictions[t, b].item())
            if idx != 0 and idx != prev_idx:
                tokens.append(idx)
            prev_idx = idx

        char_list = [VOCAB.id_to_char.get(idx, "") for idx in tokens if idx in VOCAB.id_to_char]
        decoded_strings.append("".join(char_list).strip())

    return decoded_strings


def train_model(
    manifest_path: Path = ROOT_DIR / "data" / "processed_htr" / "manifest.json",
    output_weights_path: Path = ROOT_DIR / "data" / "weights" / "crnn_htr_weights.pt",
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
):
    """Execute end-to-end neural training with mixed precision on CUDA."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_samples = data.get("train", [])
    val_samples = data.get("val", [])
    print(f"[*] Dataset: {len(train_samples)} training samples, {len(val_samples)} validation samples.")

    train_dataset = HTRDataset(train_samples, is_train=True)
    val_dataset = HTRDataset(val_samples, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Initialize CRNN Model
    model = CRNN(
        img_channel=1,
        num_classes=len(VOCAB),
        rnn_hidden_size=256,
        dropout_prob=0.2,
    ).to(device)

    # Load existing checkpoint if present for warm start
    if output_weights_path.exists():
        try:
            ckpt = torch.load(output_weights_path, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
                print(f"[+] Loaded existing model weights from: {output_weights_path}")
        except Exception as e:
            print(f"[-] Could not load existing checkpoint: {e}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_cer = float("inf")
    output_weights_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n========================================================")
    print(f"[*] STARTING NEURAL NETWORK TRAINING ({epochs} EPOCHS)")
    print("========================================================\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for step, (images, targets, in_lens, tgt_lens, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            in_lens = in_lens.to(device, non_blocking=True)
            tgt_lens = tgt_lens.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                # Forward pass: log_probs has shape (T=64, B, num_classes)
                log_probs = model(images)
                loss = criterion(log_probs, targets, in_lens, tgt_lens)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            num_batches += 1

            if (step + 1) % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] | Step [{step+1}/{len(train_loader)}] | CTC Loss: {loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss / max(1, num_batches)
        elapsed = time.time() - t0

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        total_edit_dist = 0
        total_chars = 0
        sample_preds = []

        with torch.no_grad():
            for images, targets, in_lens, tgt_lens, raw_texts in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                in_lens = in_lens.to(device, non_blocking=True)
                tgt_lens = tgt_lens.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    log_probs = model(images)
                    loss = criterion(log_probs, targets, in_lens, tgt_lens)

                val_loss += loss.item()
                val_batches += 1

                # Decode predictions
                decoded_batch = decode_greedy(log_probs.detach().cpu())
                for pred, ground_truth in zip(decoded_batch, raw_texts):
                    dist = calculate_levenshtein(pred, ground_truth)
                    total_edit_dist += dist
                    total_chars += max(1, len(ground_truth))
                    if len(sample_preds) < 5:
                        sample_preds.append((ground_truth, pred))

        avg_val_loss = val_loss / max(1, val_batches)
        cer = (total_edit_dist / max(1, total_chars)) * 100.0

        print(f"\n--- Epoch {epoch}/{epochs} Complete in {elapsed:.1f}s ---")
        print(f"    Train CTC Loss: {avg_train_loss:.4f} | Val CTC Loss: {avg_val_loss:.4f} | Val CER: {cer:.2f}%")
        print("    Sample Predictions (GT -> Pred):")
        for gt, pred in sample_preds[:4]:
            print(f'      "{gt}" -> "{pred}"')
        print("--------------------------------------------------\n")

        # Save model checkpoint
        if cer < best_cer or epoch == epochs:
            best_cer = min(cer, best_cer)
            checkpoint = {
                "architecture": "CRNN_VGG_BiGRU_CTC",
                "num_classes": len(VOCAB),
                "vocab_size": len(VOCAB),
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "cer": cer,
                "trained_on": "School Notebooks RU/EN + Augmented Synthetic Lexicon",
            }
            torch.save(checkpoint, output_weights_path)
            print(f"[+] Saved checkpoint to {output_weights_path} (Best CER: {best_cer:.2f}%)\n")

    print(f"\n[✓] TRAINING SUCCESSFULLY COMPLETED! Final Best CER: {best_cer:.2f}%")


if __name__ == "__main__":
    train_model(epochs=12, batch_size=64)
