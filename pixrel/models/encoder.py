from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EncoderOutput:
    embedding: torch.Tensor
    stages: tuple[torch.Tensor, ...]
    spatial: torch.Tensor | None
    relationships: torch.Tensor | None = None


class PixelStem(nn.Module):
    """Per-pixel RGB and normalized-coordinate embedding."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(5, channels, 1), nn.GELU(), nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack((gx, gy), dim=0).expand(b, -1, -1, -1)
        return self.proj(torch.cat((x, coords), dim=1))


class PixelRelationshipBlock(nn.Module):
    """Shared 3x3 local message aggregation with adaptive or uniform weights."""

    def __init__(self, channels: int, mode: str = "adaptive"):
        super().__init__()
        if mode not in {"adaptive", "fixed"}:
            raise ValueError(f"unsupported relationship mode: {mode}")
        self.mode = mode
        self.norm = nn.GroupNorm(1, channels)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.score = nn.Sequential(nn.Conv2d(channels * 2 + 2, channels, 1), nn.GELU(), nn.Conv2d(channels, 1, 1))
        self.update = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GELU(), nn.Conv2d(channels, channels, 1))
        self.offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]

    def forward(self, x: torch.Tensor, return_relationships: bool = False):
        b, c, h, w = x.shape
        center = self.norm(x)
        # Gather the full 3x3 neighborhood once, then process all offsets in
        # batched convolutions. This avoids 9 separate Python-level conv calls.
        neighbors = F.unfold(center, kernel_size=3, padding=1)
        neighbors = neighbors.view(b, c, 9, h, w).permute(0, 2, 1, 3, 4).contiguous()
        value_stack = self.value(neighbors.view(b * 9, c, h, w)).view(b, 9, c, h, w)
        valid_stack = F.unfold(center.new_ones((1, 1, h, w)), kernel_size=3, padding=1)
        valid_stack = valid_stack.view(1, 9, 1, h, w).bool()
        if self.mode == "adaptive":
            offsets = center.new_tensor([[dx, dy] for dy, dx in self.offsets]).view(1, 9, 2, 1, 1)
            offsets = offsets.expand(b, -1, -1, h, w)
            centers = center[:, None].expand(-1, 9, -1, -1, -1)
            score_input = torch.cat((centers, neighbors, offsets), dim=2).reshape(b * 9, 2 * c + 2, h, w)
            logits = self.score(score_input).view(b, 9, 1, h, w).masked_fill(~valid_stack, torch.finfo(x.dtype).min)
            weights = torch.softmax(logits, dim=1)
        else:
            weights = valid_stack.to(x.dtype) / valid_stack.sum(dim=1, keepdim=True).clamp_min(1)
        message = (weights * value_stack).sum(dim=1)
        out = x + self.update(message)
        return (out, weights.squeeze(2)) if return_relationships else out


class ConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(1, channels), nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x + self.net(x)


class PixrelEncoder(nn.Module):
    """Hierarchical encoder; output includes global and dense features."""

    def __init__(self, num_classes: int = 100, variant: str = "adaptive", widths=(32, 32, 48, 64),
                 high_resolution_fusion: bool = True):
        super().__init__()
        if variant not in {"adaptive", "fixed", "conv"}:
            raise ValueError("variant must be adaptive, fixed, or conv")
        self.variant = variant
        self.high_resolution_fusion = high_resolution_fusion
        self.stem = PixelStem(widths[0])
        blocks, downs = [], []
        for i, width in enumerate(widths):
            block = ConvBlock(width) if variant == "conv" else PixelRelationshipBlock(width, variant)
            blocks.append(block)
            if i < len(widths) - 1:
                downs.append(nn.Sequential(nn.Conv2d(width, widths[i + 1], 3, stride=2, padding=1), nn.GELU()))
        self.blocks = nn.ModuleList(blocks)
        self.downs = nn.ModuleList(downs)
        fusion_width = sum(widths if high_resolution_fusion else widths[1:])
        self.fuse = nn.Sequential(nn.Conv2d(fusion_width, widths[0] * 2, 1), nn.GELU(), nn.Conv2d(widths[0] * 2, widths[0] * 2, 3, padding=1))
        self.classifier = nn.Linear(widths[-1], num_classes)
        self.embedding_dim = widths[-1]
        self.spatial_dim = widths[0] * 2

    def forward(self, x: torch.Tensor, return_relationships: bool = False, classify: bool = False,
                return_spatial: bool = True):
        y = self.stem(x)
        stages, rel = [], None
        for i, block in enumerate(self.blocks):
            if return_relationships and i == 0 and isinstance(block, PixelRelationshipBlock):
                y, rel = block(y, True)
            else:
                y = block(y)
            stages.append(y)
            if i < len(self.downs):
                y = self.downs[i](y)
        target_size = stages[0].shape[-2:] if self.high_resolution_fusion else stages[1].shape[-2:]
        embedding = F.adaptive_avg_pool2d(stages[-1], 1).flatten(1)
        if classify:
            return self.classifier(embedding)
        spatial = None
        if return_spatial:
            features = stages if self.high_resolution_fusion else stages[1:]
            aligned = [f if f.shape[-2:] == target_size else F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in features]
            spatial = self.fuse(torch.cat(aligned, dim=1))
        result = EncoderOutput(embedding=embedding, stages=tuple(stages), spatial=spatial, relationships=rel)
        return result
