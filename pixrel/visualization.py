from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from pixrel.pretraining.masking import block_mask


def _unit_map(features: torch.Tensor, output_size: tuple[int, int]) -> np.ndarray:
    if features.shape[-2:] != output_size:
        features = F.interpolate(features.float(), size=output_size, mode="bilinear", align_corners=False)
    values = torch.linalg.vector_norm(features.float(), dim=1)[0].detach().cpu().numpy()
    low, high = float(values.min()), float(values.max())
    return (values - low) / max(high - low, 1e-8)


@torch.inference_mode()
def save_feature_snapshot(student, teacher, pixel_predictor, spatial_predictor, rgb_predictor, images,
                           mean, std, epoch: int, path: str | Path, seed: int,
                           patch_size: int = 4, mask_ratio: float = 0.4) -> None:
    """Save input, masked view, stage activations, dense predictions and local affinities."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(student.parameters()).device
    images = images.to(device)
    # Keep the diagnostic mask fixed so epoch-to-epoch changes reflect the model, not a new mask.
    generator = torch.Generator(device=device).manual_seed(seed)
    masked, mask = block_mask(images, patch_size=patch_size, mask_ratio=mask_ratio, generator=generator)

    was_training = student.training
    student.eval(); teacher.eval()
    student_out = student(masked, return_spatial=True, return_relationships=True)
    teacher_out = teacher(images, return_spatial=True)
    pixel_pred = pixel_predictor(student_out.stages[0])
    spatial_pred = spatial_predictor(student_out.spatial)
    rgb_predictions = rgb_predictor(student_out.stages, images.shape[-2:])
    if was_training:
        student.train()

    mean_t = images.new_tensor(mean).view(1, 3, 1, 1)
    std_t = images.new_tensor(std).view(1, 3, 1, 1)
    originals = (images * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    masked_rgb = (masked * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_outputs = [
        (prediction * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        for prediction in rgb_predictions
    ]
    masked_pixels = mask[:, 0].detach().cpu().numpy()
    count = min(images.shape[0], 4)
    output_size = tuple(images.shape[-2:])
    fig, axes = plt.subplots(count, 16, figsize=(32, max(4.0, count * 3.4)), squeeze=False)
    for row in range(count):
        col = 0
        axes[row, col].imshow(originals[row]); axes[row, col].set_title("Input image"); col += 1
        axes[row, col].imshow(masked_rgb[row]); axes[row, col].imshow(masked_pixels[row], cmap="Reds", alpha=.18)
        axes[row, col].set_title("Masked pixels"); col += 1
        for index, prediction in enumerate(rgb_outputs):
            axes[row, col].imshow(prediction[row]); axes[row, col].set_title(f"RGB reconstruction H{index}"); col += 1
        final_error = np.abs(rgb_outputs[-1][row] - originals[row]).mean(axis=-1) * masked_pixels[row]
        low, high = float(final_error.min()), float(final_error.max())
        final_error = (final_error - low) / max(high - low, 1e-8)
        axes[row, col].imshow(originals[row]); axes[row, col].imshow(final_error, cmap="inferno", alpha=.7)
        axes[row, col].set_title("Masked RGB error (H3)"); col += 1

        stage_maps = [_unit_map(stage[row:row + 1], output_size) for stage in teacher_out.stages]
        h0_target = stage_maps[0]
        h0_pred = _unit_map(pixel_pred[row:row + 1], output_size)
        axes[row, col].imshow(originals[row]); axes[row, col].imshow(h0_target, cmap="magma", alpha=.58)
        axes[row, col].set_title("Teacher H0 response"); col += 1
        axes[row, col].imshow(originals[row]); axes[row, col].imshow(h0_pred, cmap="magma", alpha=.58)
        axes[row, col].set_title("Student H0 prediction"); col += 1

        for index, feature_map in enumerate(stage_maps):
            axes[row, col].imshow(originals[row]); axes[row, col].imshow(feature_map, cmap="viridis", alpha=.58)
            axes[row, col].set_title(f"Teacher stage H{index}"); col += 1

        fused_target = _unit_map(teacher_out.spatial[row:row + 1], output_size)
        fused_prediction = _unit_map(spatial_pred[row:row + 1], output_size)
        axes[row, col].imshow(originals[row]); axes[row, col].imshow(fused_target, cmap="plasma", alpha=.58)
        axes[row, col].set_title("Teacher fused map"); col += 1
        axes[row, col].imshow(originals[row]); axes[row, col].imshow(fused_prediction, cmap="plasma", alpha=.58)
        axes[row, col].set_title("Student fused prediction"); col += 1

        relationship = student_out.relationships
        if relationship is not None:
            cy, cx = relationship.shape[-2] // 2, relationship.shape[-1] // 2
            weights = relationship[row, :, cy, cx].detach().float().cpu().reshape(3, 3).numpy()
            axes[row, col].imshow(weights, cmap="Blues", vmin=0)
            axes[row, col].set_xticks(range(3), ["left", "center", "right"])
            axes[row, col].set_yticks(range(3), ["up", "center", "down"])
            axes[row, col].set_title("H0 local weights\n(center pixel)")
        else:
            axes[row, col].text(.5, .5, "No relationship\nweights in this variant", ha="center", va="center")
            axes[row, col].set_title("Local weights")
        col += 1
        for axis in axes[row, :-1]:
            axis.set_xticks([]); axis.set_yticks([])
        if relationship is None:
            axes[row, -1].set_xticks([]); axes[row, -1].set_yticks([])
    fig.suptitle(f"PIXREL feature-path snapshot — epoch {epoch}", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_learning_curves(history: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for key, label in (("total_loss", "total"), ("local_pixel_cosine_loss", "local pixel"),
                       ("fused_spatial_cosine_loss", "fused spatial"), ("global_cosine_loss", "global"),
                       ("masked_rgb_reconstruction_loss", "masked RGB")):
        axes[0].plot(epochs, [row[key] for row in history], label=label)
    axes[0].set(title="Self-supervised objectives", xlabel="Epoch", ylabel="Objective loss")
    axes[0].grid(alpha=.25); axes[0].legend()
    axes[1].plot(epochs, [row["teacher_feature_std"] for row in history], label="local teacher feature std")
    axes[1].plot(epochs, [row["teacher_embedding_channel_std"] for row in history], label="global embedding channel std")
    axes[1].set(title="Feature variation diagnostics", xlabel="Epoch", ylabel="Standard deviation")
    axes[1].grid(alpha=.25); axes[1].legend()
    axes[2].plot(epochs, [row["masked_rgb_mae_01"] for row in history], color="tab:blue", label="masked RGB MAE")
    axes[2].set(title="Masked RGB reconstruction", xlabel="Epoch", ylabel="MAE in [0, 1] RGB")
    axes[2].grid(alpha=.25)
    psnr_axis = axes[2].twinx()
    psnr_axis.plot(epochs, [row["masked_rgb_psnr_db"] for row in history], color="tab:orange", label="masked PSNR")
    psnr_axis.set_ylabel("PSNR (dB)")
    lines = axes[2].get_lines() + psnr_axis.get_lines()
    axes[2].legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def save_progress_gif(snapshot_paths: list[str | Path], path: str | Path, duration_ms: int = 600) -> None:
    frames = []
    for snapshot in snapshot_paths:
        with Image.open(snapshot) as image:
            frame = image.convert("RGB")
            frame.thumbnail((1600, 900))
            frames.append(frame)
    if not frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)
    for frame in frames:
        frame.close()
