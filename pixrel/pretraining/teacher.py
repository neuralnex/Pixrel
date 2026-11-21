from __future__ import annotations

import math

import torch
from torch import nn


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for target, online in zip(teacher.parameters(), student.parameters()):
        target.lerp_(online, 1.0 - momentum)


def cosine_ema_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return start + (end - start) * (1.0 - math.cos(math.pi * progress)) / 2.0
