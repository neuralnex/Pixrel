from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import PixrelEncoder


class PixrelSegmenter(nn.Module):
    """Dense prediction adapter that trains the encoder's fused spatial path."""

    def __init__(self, encoder: PixrelEncoder, num_classes: int, decoder_channels: int = 64):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if decoder_channels < 1:
            raise ValueError("decoder_channels must be positive")
        self.encoder = encoder
        self.num_classes = num_classes
        self.decoder_channels = decoder_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder.spatial_dim, decoder_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, num_classes, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.encoder(images, return_spatial=True)
        if output.spatial is None:
            raise RuntimeError("encoder did not return fused spatial features")
        logits = self.decoder(output.spatial)
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits
