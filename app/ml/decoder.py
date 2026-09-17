# ### FILE: app/ml/decoder.py
"""
CTC (Connectionist Temporal Classification) Decoders.
Provides both Greedy (best path) and Prefix Beam Search decoding with confidence scoring.
"""

import math
from collections import defaultdict
from typing import List, Tuple
import numpy as np
import torch
from app.ml.vocab import Vocabulary, VOCAB


class CTCGreedyDecoder:
    """Greedy argmax decoder collapsing consecutive duplicates and blank tokens."""

    def __init__(self, vocab: Vocabulary = VOCAB, blank_idx: int = 0) -> None:
        self.vocab = vocab
        self.blank_idx = blank_idx

    def decode(
        self,
        log_probs: torch.Tensor,
    ) -> List[Tuple[str, float]]:
        """
        Args:
            log_probs: Tensor of shape (T, B, C) containing log-probabilities.
        Returns:
            List of (decoded_text, confidence_score) for each item in batch.
        """
        # Detach and convert to CPU
        probs = torch.exp(log_probs).detach().cpu()  # Shape: (T, B, C)
        t_len, batch_size, num_classes = probs.size()

        max_probs, argmax_indices = torch.max(probs, dim=2)  # Shapes: (T, B)

        results: List[Tuple[str, float]] = []

        for b in range(batch_size):
            char_list: List[str] = []
            confidence_list: List[float] = []
            prev_idx = -1

            for t in range(t_len):
                idx = int(argmax_indices[t, b].item())
                prob = float(max_probs[t, b].item())

                if idx != self.blank_idx:
                    if idx != prev_idx:
                        char = self.vocab.get_char(idx)
                        if char and char not in (self.vocab.BLANK_TOKEN, self.vocab.UNK_TOKEN):
                            char_list.append(char)
                            confidence_list.append(prob)
                prev_idx = idx

            text = "".join(char_list)
            avg_conf = (
                float(sum(confidence_list) / len(confidence_list))
                if confidence_list
                else 0.0
            )
            # Clamp confidence to [0.0, 1.0]
            avg_conf = max(0.0, min(1.0, avg_conf))
            results.append((text, avg_conf))

        return results


class CTCBeamSearchDecoder:
    """Prefix Beam Search decoder for Connectionist Temporal Classification."""

    def __init__(
        self,
        vocab: Vocabulary = VOCAB,
        beam_width: int = 5,
        blank_idx: int = 0,
    ) -> None:
        self.vocab = vocab
        self.beam_width = beam_width
        self.blank_idx = blank_idx

    def decode(
        self,
        log_probs: torch.Tensor,
    ) -> List[Tuple[str, float]]:
        """
        Args:
            log_probs: Tensor of shape (T, B, C) containing log-probabilities.
        Returns:
            List of (decoded_text, confidence_score) for each item in batch.
        """
        probs = torch.exp(log_probs).detach().cpu().numpy()
        t_len, batch_size, num_classes = probs.shape

        batch_results: List[Tuple[str, float]] = []

        for b in range(batch_size):
            # Beam entry: prefix_tuple -> (prob_blank, prob_non_blank)
            beams = {(): (1.0, 0.0)}

            for t in range(t_len):
                next_beams = defaultdict(lambda: (0.0, 0.0))
                p_blank = float(probs[t, b, self.blank_idx])

                # 1. Extend with blank
                for prefix, (pb, pnb) in beams.items():
                    curr_pb, curr_pnb = next_beams[prefix]
                    next_beams[prefix] = (curr_pb + (pb + pnb) * p_blank, curr_pnb)

                # 2. Extend with non-blank characters
                for c in range(num_classes):
                    if c == self.blank_idx:
                        continue
                    p_char = float(probs[t, b, c])
                    if p_char < 1e-6:
                        continue

                    for prefix, (pb, pnb) in beams.items():
                        last_c = prefix[-1] if prefix else None

                        if c == last_c:
                            # Same character repeated:
                            # - Coming from blank starts a new repeated character
                            # - Coming from non-blank merges into the same character
                            curr_pb, curr_pnb = next_beams[prefix]
                            next_beams[prefix] = (curr_pb, curr_pnb + pnb * p_char)

                            new_prefix = prefix + (c,)
                            np_b, np_nb = next_beams[new_prefix]
                            next_beams[new_prefix] = (np_b, np_nb + pb * p_char)
                        else:
                            new_prefix = prefix + (c,)
                            np_b, np_nb = next_beams[new_prefix]
                            next_beams[new_prefix] = (np_b, np_nb + (pb + pnb) * p_char)

                # Prune to beam_width
                sorted_beams = sorted(
                    next_beams.items(),
                    key=lambda item: item[1][0] + item[1][1],
                    reverse=True,
                )
                beams = dict(sorted_beams[: self.beam_width])

            # Select best beam
            best_prefix, (pb, pnb) = max(
                beams.items(), key=lambda item: item[1][0] + item[1][1]
            )

            decoded_chars = [
                self.vocab.get_char(idx)
                for idx in best_prefix
                if idx not in (self.vocab.blank_id, self.vocab.unk_id)
            ]
            decoded_text = "".join(decoded_chars)

            # Compute confidence from peak token probabilities along the sequence
            if best_prefix:
                char_confs = [
                    float(np.max(probs[:, b, idx]))
                    for idx in best_prefix
                    if idx not in (self.vocab.blank_id, self.vocab.unk_id)
                ]
                confidence = float(np.mean(char_confs)) if char_confs else 0.5
            else:
                confidence = 0.0

            confidence = max(0.0, min(1.0, confidence))
            batch_results.append((decoded_text, confidence))

        return batch_results
