# ### FILE: tests/test_ml_crnn.py
"""
Unit and Integration Tests for PyTorch CRNN Machine Learning Subsystem.
Verifies vocabulary collation, model forward pass shapes, CTC loss convergence,
Greedy & Beam Search decoding, and batch inference.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

from app.ml.vocab import VOCAB, Vocabulary
from app.ml.crnn_model import CRNN
from app.ml.decoder import CTCGreedyDecoder, CTCBeamSearchDecoder
from app.ml.inference import CRNNInferenceEngine
from app.ml.weights_initializer import ensure_weights_exist, initialize_model_weights


def test_vocabulary_encoding_decoding():
    """Verify bidirectional token mapping for Cyrillic, Latin, and Math symbols."""
    sample_text = "Теорема 1: f(x) = x^2 + \\int_0^1 g(t) dt"
    encoded = VOCAB.encode(sample_text)

    assert isinstance(encoded, list)
    assert len(encoded) == len(sample_text)
    assert all(isinstance(idx, int) for idx in encoded)

    decoded = VOCAB.decode(encoded)
    assert decoded == sample_text


def test_crnn_forward_pass_dimensions():
    """Verify CRNN tensor transformations from (B, 1, 32, W) to (W/4, B, num_classes)."""
    batch_size = 2
    height = 32
    width = 256
    num_classes = len(VOCAB)

    model = CRNN(img_channel=1, num_classes=num_classes)
    model.eval()

    # Input tensor: (B, 1, 32, 256)
    dummy_input = torch.randn(batch_size, 1, height, width)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected time steps T = width / 4 = 64
    expected_t = width // 4
    assert output.shape == (expected_t, batch_size, num_classes)

    # Verify log-softmax normalization: sum of exp(log_probs) across classes is 1.0
    probs = torch.exp(output)
    sums = torch.sum(probs, dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_ctc_loss_calculation():
    """Verify standard PyTorch CTCLoss computes finite loss on model output."""
    model = CRNN(img_channel=1, num_classes=len(VOCAB))
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    batch_size = 2
    width = 128
    dummy_input = torch.randn(batch_size, 1, 32, width)

    log_probs = model(dummy_input)  # Shape: (32, 2, C)
    input_lengths = torch.full((batch_size,), width // 4, dtype=torch.long)

    # Targets: "abc", "тест"
    targets = torch.tensor([VOCAB.char_to_id["a"], VOCAB.char_to_id["b"], VOCAB.char_to_id["c"],
                            VOCAB.char_to_id["т"], VOCAB.char_to_id["е"], VOCAB.char_to_id["с"], VOCAB.char_to_id["т"]],
                           dtype=torch.long)
    target_lengths = torch.tensor([3, 4], dtype=torch.long)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert float(loss.item()) > 0.0


def test_ctc_greedy_decoder():
    """Verify greedy CTC decoder collapses repeats and ignores blank tokens."""
    decoder = CTCGreedyDecoder(vocab=VOCAB, blank_idx=0)

    # Simulate logits of sequence with blanks and repeated indices:
    # Tokens: [blank, 'а', 'а', blank, 'б', 'б', 'в', blank] -> 'абв'
    idx_a = VOCAB.char_to_id["а"]
    idx_b = VOCAB.char_to_id["б"]
    idx_v = VOCAB.char_to_id["в"]

    time_steps = [0, idx_a, idx_a, 0, idx_b, idx_b, idx_v, 0]
    t_len = len(time_steps)

    # Create one-hot log-probabilities
    log_probs = torch.full((t_len, 1, len(VOCAB)), -100.0)
    for t, char_idx in enumerate(time_steps):
        log_probs[t, 0, char_idx] = 0.0  # exp(0) = 1.0

    results = decoder.decode(log_probs)
    assert len(results) == 1
    decoded_text, confidence = results[0]

    assert decoded_text == "абв"
    assert confidence > 0.95


def test_ctc_beam_search_decoder():
    """Verify beam search decoder on predictable sequence."""
    decoder = CTCBeamSearchDecoder(vocab=VOCAB, beam_width=3, blank_idx=0)

    idx_x = VOCAB.char_to_id["x"]
    idx_y = VOCAB.char_to_id["y"]

    time_steps = [idx_x, 0, idx_y]
    log_probs = torch.full((3, 1, len(VOCAB)), -100.0)
    for t, char_idx in enumerate(time_steps):
        log_probs[t, 0, char_idx] = 0.0

    results = decoder.decode(log_probs)
    assert len(results) == 1
    decoded_text, _ = results[0]
    assert decoded_text == "xy"


def test_crnn_inference_engine_execution(synthetic_line_crop: np.ndarray, temp_dir: Path):
    """Verify full CRNN inference engine pipeline on cropped line images."""
    weights_path = temp_dir / "weights" / "engine_test.pt"
    ensure_weights_exist(weights_path)

    engine = CRNNInferenceEngine(weights_path=weights_path, device="cpu", batch_size=4)

    # Preprocessing test
    tensor = engine.preprocess_line_image(synthetic_line_crop)
    assert tensor.shape == (1, 1, 32, 256)
    assert tensor.dtype == torch.float32

    # Batch inference test
    crops = [synthetic_line_crop, synthetic_line_crop]
    predictions = engine.predict_batch(crops, use_beam_search=False)

    assert len(predictions) == 2
    for text, conf in predictions:
        assert isinstance(text, str)
        assert 0.0 <= conf <= 1.0
