from __future__ import annotations

from pathlib import Path

import torch

from .encoder import PixrelEncoder


def load_pretrained_encoder(checkpoint: str | Path, *, num_classes: int = 100,
                            map_location: str | torch.device = "cpu") -> tuple[PixrelEncoder, dict]:
    """Load a PIXREL encoder checkpoint and return (model, preprocessing metadata).

    Supports raw encoder state dictionaries, self-describing `encoder_bundle.pt`
    files, and supervised `best.pt` checkpoints produced by pixrel-train.
    The classifier head is reinitialized when its class count differs.
    """
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload and "format_version" in payload:
        metadata = dict(payload)
        state = payload["state_dict"]
    elif isinstance(payload, dict) and "model" in payload and "variant" in payload:
        metadata = {"variant": payload["variant"], "format_version": 0}
        state = payload["model"]
    else:
        metadata = {"variant": "adaptive", "format_version": 0}
        state = payload

    variant = metadata.get("variant", "adaptive")
    widths = tuple(metadata.get("widths", (32, 32, 48, 64)))
    high_resolution_fusion = metadata.get("high_resolution_fusion", True)
    model = PixrelEncoder(num_classes=num_classes, variant=variant, widths=widths,
                          high_resolution_fusion=high_resolution_fusion)
    state = dict(state)
    for name in ("classifier.weight", "classifier.bias"):
        if name in state and state[name].shape != model.state_dict()[name].shape:
            state.pop(name)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {"classifier.weight", "classifier.bias"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint does not match the PIXREL encoder architecture: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    preprocessing = {
        "input_size": metadata.get("input_size", 32),
        "normalization": metadata.get("normalization", {
            "mean": (.5071, .4867, .4408), "std": (.2675, .2565, .2761)
        }),
        "pretraining_dataset": metadata.get("pretraining_dataset", "unknown"),
    }
    return model, preprocessing
