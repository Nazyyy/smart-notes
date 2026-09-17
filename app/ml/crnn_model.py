# ### FILE: app/ml/crnn_model.py
"""
PyTorch CRNN (Convolutional Recurrent Neural Network) Architecture.
Combines VGG-style CNN feature extraction with 2-layer Bidirectional GRU
and linear projection for Connectionist Temporal Classification (CTC).
"""

import torch
import torch.nn as nn
from typing import Tuple


class BidirectionalGRU(nn.Module):
    """Bidirectional GRU wrapper with linear projection."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=False,
        )
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (T, B, input_size)
        Output: (T, B, output_size)
        """
        recurrent, _ = self.rnn(x)
        t, b, h = recurrent.size()
        t_rec = recurrent.view(t * b, h)
        output = self.embedding(t_rec)
        output = output.view(t, b, -1)
        return output


class CRNN(nn.Module):
    """
    CRNN architecture for Handwritten Text Recognition (HTR).
    Input: (B, 1, 32, W) grayscale line image tensor.
    Output: (T, B, num_classes) log-probabilities suitable for torch.nn.CTCLoss.
    """

    def __init__(
        self,
        img_channel: int = 1,
        num_classes: int = 220,
        rnn_hidden_size: int = 256,
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Conv 1: (B, 1, 32, W) -> (B, 64, 16, W/2)
            nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv 2: (B, 64, 16, W/2) -> (B, 128, 8, W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv 3: (B, 128, 8, W/4) -> (B, 256, 8, W/4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Conv 4: (B, 256, 8, W/4) -> (B, 256, 4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Conv 5: (B, 256, 4, W/4) -> (B, 512, 4, W/4)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Conv 6: (B, 512, 4, W/4) -> (B, 512, 2, W/4)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Conv 7: (B, 512, 2, W/4) -> (B, 512, 1, W/4)
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Bidirectional Recurrent Sequence Modeling
        self.dropout = nn.Dropout(dropout_prob)
        self.rnn1 = BidirectionalGRU(
            input_size=512,
            hidden_size=rnn_hidden_size,
            output_size=rnn_hidden_size,
        )
        self.rnn2 = BidirectionalGRU(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            output_size=num_classes,
        )

        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 1, H=32, W)
        Returns:
            Log-probabilities of shape (T, B, num_classes) where T = W / 4
        """
        # Feature extraction
        features = self.cnn(x)  # Shape: (B, 512, 1, W/4)

        b, c, h, w = features.size()
        assert h == 1, f"Expected height 1 after CNN pooling, got {h}"

        # Squeeze height dimension: (B, C, W)
        features = features.squeeze(2)

        # Permute to (T=W, B, C) for recurrent processing
        sequence = features.permute(2, 0, 1)

        # Recurrent sequence translation
        sequence = self.dropout(sequence)
        sequence = self.rnn1(sequence)
        logits = self.rnn2(sequence)  # Shape: (T, B, num_classes)

        # Output normalized log-probabilities for CTC loss
        log_probs = self.log_softmax(logits)
        return log_probs
