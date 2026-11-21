from __future__ import annotations

import torch
from torch.nn import functional as F


def block_mask(images: torch.Tensor, patch_size: int = 4, mask_ratio: float = 0.4,
               generator: torch.Generator | None = None):
    """Replace random aligned patches with the normalized-dataset mean (zero)."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    batch, _, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("image height and width must be divisible by patch_size")
    grid_h, grid_w = height // patch_size, width // patch_size
    grid = torch.rand((batch, 1, grid_h, grid_w), device=images.device, generator=generator) < mask_ratio
    # Guarantee at least one prediction target for every image.
    empty = ~grid.flatten(1).any(dim=1)
    if empty.any():
        grid[empty, 0, 0, 0] = True
    mask = F.interpolate(grid.float(), size=(height, width), mode="nearest").bool()
    return images.masked_fill(mask.expand_as(images), 0.0), mask
