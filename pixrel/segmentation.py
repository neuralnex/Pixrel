from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from pixrel.models.checkpoint import load_pretrained_encoder
from pixrel.models.encoder import PixrelEncoder
from pixrel.models.segmentation import PixrelSegmenter


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESAMPLING = getattr(Image, "Resampling", Image)


class PairedMaskDataset(Dataset):
    """Image/mask tree pair; masks are single-channel integer class IDs."""

    def __init__(self, images_dir, masks_dir, image_size, num_classes, ignore_index=255,
                 training=False, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.images_dir = Path(images_dir).expanduser()
        self.masks_dir = Path(masks_dir).expanduser()
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.training = training
        self.mean = tuple(mean)
        self.std = tuple(std)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"images directory does not exist: {self.images_dir}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"masks directory does not exist: {self.masks_dir}")
        valid_extensions = {extension.lower() for extension in IMG_EXTENSIONS}
        images = sorted(path for path in self.images_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in valid_extensions)
        if not images:
            raise ValueError(f"no supported images found in {self.images_dir}")
        self.samples = []
        missing = []
        for image_path in images:
            relative_stem = image_path.relative_to(self.images_dir).with_suffix("")
            candidates = [self.masks_dir / Path(f"{relative_stem}{suffix}")
                          for suffix in (".png", ".tif", ".tiff")]
            mask_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if mask_path is None:
                missing.append(str(relative_stem))
            else:
                self.samples.append((image_path, mask_path))
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"missing masks for {len(missing)} images under {self.masks_dir}: {preview}")
        if self.image_size < 1 or self.num_classes < 2:
            raise ValueError("image_size must be positive and num_classes must be at least 2")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB").resize(
                (self.image_size, self.image_size), RESAMPLING.BILINEAR
            )
        with Image.open(mask_path) as source_mask:
            if source_mask.mode not in {"L", "P", "I", "I;16"}:
                raise ValueError(
                    f"mask must be a single-channel class-ID image (L/P/I mode): {mask_path} has {source_mask.mode}"
                )
            mask = source_mask.resize((self.image_size, self.image_size), RESAMPLING.NEAREST)
            mask_array = np.asarray(mask, dtype=np.int64).copy()

        if mask_array.ndim != 2:
            raise ValueError(f"mask must be single-channel: {mask_path}")
        invalid = (mask_array != self.ignore_index) & ((mask_array < 0) | (mask_array >= self.num_classes))
        if invalid.any():
            values = np.unique(mask_array[invalid])[:10].tolist()
            raise ValueError(
                f"mask {mask_path} has class IDs outside [0, {self.num_classes - 1}] "
                f"and ignore_index={self.ignore_index}: {values}"
            )

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, self.mean, self.std)
        mask_tensor = torch.from_numpy(mask_array)
        if self.training and torch.rand(()) < 0.5:
            image_tensor = torch.flip(image_tensor, dims=(-1,))
            mask_tensor = torch.flip(mask_tensor, dims=(-1,))
        return image_tensor, mask_tensor


def _confusion_matrix(predictions, targets, num_classes, ignore_index):
    valid = targets != ignore_index
    if not valid.any():
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=targets.device)
    encoded = targets[valid] * num_classes + predictions[valid]
    return torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def segmentation_metrics(confusion, class_names=None):
    confusion = confusion.to(torch.float64)
    true_positive = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - true_positive
    present = union > 0
    iou = torch.zeros_like(union)
    iou[present] = true_positive[present] / union[present]
    total = confusion.sum()
    pixel_accuracy = (true_positive.sum() / total).item() if total > 0 else 0.0
    mean_iou = iou[present].mean().item() if present.any() else 0.0
    names = class_names or [str(index) for index in range(confusion.shape[0])]
    return {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "per_class_iou": {name: iou[index].item() for index, name in enumerate(names)},
        "confusion_matrix": confusion.to(torch.int64).tolist(),
    }


def _run_epoch(model, loader, device, num_classes, ignore_index, optimizer=None):
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    loss_sum = valid_pixels = image_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            valid_count = int((masks != ignore_index).sum().item())
            image_count += images.shape[0]
            if valid_count == 0:
                continue
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, masks, ignore_index=ignore_index)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite segmentation loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if valid_count:
                loss_sum += loss.item() * valid_count
                valid_pixels += valid_count
            predictions = logits.argmax(dim=1)
            confusion += _confusion_matrix(predictions, masks, num_classes, ignore_index)
    if valid_pixels == 0:
        raise ValueError("dataset contains no valid labeled pixels")
    metrics = segmentation_metrics(confusion)
    metrics.update({
        "loss": loss_sum / max(valid_pixels, 1),
        "valid_pixels": valid_pixels,
        "images": image_count,
    })
    return metrics


def _resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_main():
    parser = argparse.ArgumentParser(description="Fine-tune a PIXREL encoder for semantic segmentation")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="pretrained PIXREL encoder_bundle.pt")
    source.add_argument("--scratch", action="store_true", help="initialize the PIXREL encoder randomly")
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-masks", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-masks", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--class-names", nargs="*", help="optional names in mask-ID order")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--variant", choices=["adaptive", "fixed", "conv"], default="adaptive")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--output", default="./outputs/segmentation")
    args = parser.parse_args()
    if args.num_classes < 2 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("num_classes must be >=2; epochs, batch_size, and lr must be positive")
    if args.class_names and len(args.class_names) != args.num_classes:
        parser.error("--class-names must provide one name per class ID")
    _train(args)


def _train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = _resolve_device(args.device)
    if args.scratch:
        image_size = args.image_size or 128
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        encoder = PixrelEncoder(num_classes=args.num_classes, variant=args.variant)
        source = "scratch"
    else:
        encoder, preprocessing = load_pretrained_encoder(args.checkpoint, num_classes=args.num_classes)
        image_size = args.image_size or preprocessing["input_size"]
        mean = tuple(preprocessing["normalization"]["mean"])
        std = tuple(preprocessing["normalization"]["std"])
        source = str(args.checkpoint)
    model = PixrelSegmenter(encoder, args.num_classes).to(device)
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)
    train_data = PairedMaskDataset(args.train_images, args.train_masks, image_size,
                                   args.num_classes, args.ignore_index, training=True,
                                   mean=mean, std=std)
    val_data = PairedMaskDataset(args.val_images, args.val_masks, image_size,
                                 args.num_classes, args.ignore_index, training=False,
                                 mean=mean, std=std)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=generator,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    history = []
    began = time.perf_counter()
    class_names = args.class_names or [str(index) for index in range(args.num_classes)]
    for epoch in range(args.epochs):
        train_metrics = _run_epoch(model, train_loader, device, args.num_classes, args.ignore_index, optimizer)
        val_metrics = _run_epoch(model, val_loader, device, args.num_classes, args.ignore_index)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_mean_iou": train_metrics["mean_iou"],
            "val_loss": val_metrics["loss"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {
            "model": model.state_dict(),
            "variant": model.encoder.variant,
            "widths": model.encoder.widths,
            "high_resolution_fusion": model.encoder.high_resolution_fusion,
            "num_classes": args.num_classes,
            "decoder_channels": model.decoder_channels,
            "class_names": class_names,
            "ignore_index": args.ignore_index,
            "preprocessing": {"input_size": image_size, "normalization": {"mean": mean, "std": std}},
            "seed": args.seed,
            "source": source,
            "config": vars(args),
            "epoch": epoch + 1,
        }
        torch.save(state, output / "last.pt")
        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            torch.save(state, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "source": source,
        "variant": model.encoder.variant,
        "seed": args.seed,
        "classes": args.num_classes,
        "class_names": class_names,
        "best_val_mean_iou": best_iou,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "elapsed_seconds": time.perf_counter() - began,
        "config": vars(args),
    }
    (output / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def evaluate_main():
    parser = argparse.ArgumentParser(description="Evaluate a PIXREL semantic segmentation checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output", default="./outputs/segmentation/eval.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = _resolve_device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = PixrelEncoder(num_classes=state["num_classes"], variant=state["variant"],
                            widths=tuple(state["widths"]),
                            high_resolution_fusion=state["high_resolution_fusion"])
    model = PixrelSegmenter(encoder, state["num_classes"],
                            decoder_channels=state.get("decoder_channels", 64))
    model.load_state_dict(state["model"])
    model.to(device)
    preprocessing = state["preprocessing"]
    dataset = PairedMaskDataset(args.images_dir, args.masks_dir,
                                preprocessing["input_size"], state["num_classes"],
                                state["ignore_index"], training=False,
                                mean=preprocessing["normalization"]["mean"],
                                std=preprocessing["normalization"]["std"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    result = _run_epoch(model, loader, device, state["num_classes"], state["ignore_index"])
    result.update({
        "checkpoint": str(args.checkpoint),
        "class_names": state["class_names"],
        "device": str(device),
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    train_main()
