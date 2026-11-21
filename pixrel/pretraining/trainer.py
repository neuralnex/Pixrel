from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

from pixrel.models.encoder import PixrelEncoder
from pixrel.pretraining.masking import block_mask
from pixrel.pretraining.teacher import cosine_ema_momentum, ema_update
from pixrel.visualization import save_feature_snapshot, save_learning_curves, save_progress_gif


class PixelPredictor(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GELU(), nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return self.net(x)


class GlobalPredictor(nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimensions, dimensions * 2), nn.GELU(), nn.Linear(dimensions * 2, dimensions))

    def forward(self, x):
        return self.net(x)


class RGBReconstructionHead(nn.Module):
    """Predict normalized RGB pixels from every encoder stage."""
    def __init__(self, widths=(32, 32, 48, 64)):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 3, 1))
            for width in widths
        ])

    def forward(self, stages, output_size):
        predictions = []
        for head, feature in zip(self.heads, stages):
            prediction = head(feature)
            if prediction.shape[-2:] != output_size:
                prediction = F.interpolate(prediction, size=output_size, mode="bilinear", align_corners=False)
            predictions.append(prediction)
        return predictions


class UnlabeledImageFolder(Dataset):
    """Recursively load images from a flat or nested directory without class labels."""
    def __init__(self, root, transform):
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"ImageFolder data directory does not exist: {self.root}. "
                "Mount/copy your images to Colab and pass that real directory with --data."
            )
        valid_extensions = {extension.lower() for extension in IMG_EXTENSIONS}
        self.paths = sorted(path for path in self.root.rglob("*")
                            if path.is_file() and path.suffix.lower() in valid_extensions)
        if not self.paths:
            raise FileNotFoundError(
                f"No supported images found under {self.root}. Add JPG/PNG images or check the mounted data path."
            )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(str(self.paths[index]))
        if self.transform:
            image = self.transform(image)
        return image, 0


def _rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _masked_cosine(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction.float(), dim=1)
    target = F.normalize(target.detach().float(), dim=1)
    per_pixel = 1.0 - (prediction * target).sum(dim=1)
    mask = F.interpolate(mask.float(), size=per_pixel.shape[-2:], mode="nearest").squeeze(1)
    return (per_pixel * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_rgb_loss(predictions, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pixel_mask = mask[:, 0].float()
    denominator = pixel_mask.sum().clamp_min(1.0)
    losses = []
    for prediction in predictions:
        per_pixel = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none").mean(dim=1)
        losses.append((per_pixel * pixel_mask).sum() / denominator)
    return torch.stack(losses).mean()


@torch.no_grad()
def _masked_rgb_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, mean, std):
    mean_t = target.new_tensor(mean).view(1, 3, 1, 1)
    std_t = target.new_tensor(std).view(1, 3, 1, 1)
    prediction_rgb = (prediction.detach() * std_t + mean_t).clamp(0, 1)
    target_rgb = (target * std_t + mean_t).clamp(0, 1)
    pixel_mask = mask.float()
    expanded_mask = pixel_mask.expand_as(target_rgb)
    denominator = expanded_mask.sum().clamp_min(1.0)
    difference = prediction_rgb - target_rgb
    mae = (difference.abs() * expanded_mask).sum() / denominator
    mse = (difference.square() * expanded_mask).sum() / denominator
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
    return mae.item(), psnr.item()


def _make_loader(args, generator):
    if args.dataset == "cifar100":
        transform = transforms.Compose([
            transforms.RandomCrop(args.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((.5071, .4867, .4408), (.2675, .2565, .2761)),
        ])
        dataset = datasets.CIFAR100(args.data, train=True, download=True, transform=transform)
        train_split, _ = random_split(dataset, [len(dataset) - 5000, 5000], generator=generator)
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        dataset = UnlabeledImageFolder(args.data, transform=transform)
        train_split = dataset
    return DataLoader(train_split, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.workers, generator=generator,
                      pin_memory=args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))


def _make_visual_batch(args):
    mean, std = ((.5071, .4867, .4408), (.2675, .2565, .2761)) if args.dataset == "cifar100" else ((.485, .456, .406), (.229, .224, .225))
    if args.dataset == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        dataset = datasets.CIFAR100(args.data, train=True, download=True, transform=transform)
        train_split, _ = random_split(dataset, [len(dataset) - 5000, 5000],
                                      generator=torch.Generator().manual_seed(args.seed))
        available = train_split.indices
    else:
        transform = transforms.Compose([
            transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        dataset = UnlabeledImageFolder(args.data, transform=transform)
        available = range(len(dataset))
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(list(available), size=min(args.visualize_samples, len(available)), replace=False)
    return torch.stack([dataset[int(index)][0] for index in selected]), mean, std


def _save_state(path, student, teacher, pixel_predictor, spatial_predictor, global_predictor, optimizer, scheduler,
                rgb_predictor, epoch, history, loader_generator, args, device):
    state = {
        "student": student.state_dict(), "teacher": teacher.state_dict(),
        "pixel_predictor": pixel_predictor.state_dict(), "spatial_predictor": spatial_predictor.state_dict(),
        "global_predictor": global_predictor.state_dict(),
        "rgb_predictor": rgb_predictor.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "epoch": epoch, "history": history, "loader_rng": loader_generator.get_state(),
        "rng": _rng_state(), "variant": args.variant, "config": vars(args), "device": str(device),
    }
    torch.save(state, path)


def pretrain_main():
    parser = argparse.ArgumentParser(description="PIXREL masked-context and RGB reconstruction pretraining")
    parser.add_argument("--data", default="./data")
    parser.add_argument("--output", default="./outputs/pretrain-cifar100")
    parser.add_argument("--dataset", choices=["cifar100", "imagefolder"], default="cifar100",
                        help="imagefolder expects class subdirectories; labels are ignored")
    parser.add_argument("--image-size", type=int, help="square input size (default: 32 for CIFAR-100, 128 for ImageFolder)")
    parser.add_argument("--variant", choices=["adaptive", "fixed", "conv"], default="adaptive")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, help="default: 128 for CIFAR-100, 16 for ImageFolder")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--global-weight", type=float, default=0.5)
    parser.add_argument("--spatial-weight", type=float, default=1.0)
    parser.add_argument("--rgb-weight", type=float, default=0.1,
                        help="weight for masked RGB reconstruction averaged across H0-H3")
    parser.add_argument("--ema-start", type=float, default=0.99)
    parser.add_argument("--ema-end", type=float, default=0.999)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--visualize-every", type=int, default=5,
                        help="save feature-path snapshots every N epochs (also saves first and last epoch)")
    parser.add_argument("--visualize-samples", type=int, default=2, help="fixed images shown in each snapshot")
    parser.add_argument("--resume", help="resume from a PIXREL pretraining last.pt checkpoint")
    _train(parser.parse_args())


def _train(args):
    if args.image_size is None:
        args.image_size = 32 if args.dataset == "cifar100" else 128
    if args.batch_size is None:
        args.batch_size = 128 if args.dataset == "cifar100" else 16
    if args.image_size < 8 or args.image_size % args.patch_size:
        raise ValueError("image_size must be >= 8 and divisible by patch_size")
    if args.visualize_every < 1 or args.visualize_samples < 1:
        raise ValueError("visualize_every and visualize_samples must be positive")
    if args.dataset == "cifar100" and args.image_size != 32:
        raise ValueError("CIFAR-100 pretraining uses 32x32 images; use --dataset imagefolder for other sizes")
    if (args.epochs < 1 or args.patch_size < 1 or args.global_weight < 0
            or args.spatial_weight < 0 or args.rgb_weight < 0):
        raise ValueError("epochs and patch_size must be positive; loss weights must be nonnegative")
    if not 0.0 <= args.ema_start <= args.ema_end < 1.0:
        raise ValueError("EMA momenta must satisfy 0 <= ema_start <= ema_end < 1")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = _make_loader(args, loader_generator)
    student = PixrelEncoder(variant=args.variant).to(device)
    teacher = deepcopy(student).to(device).eval()
    teacher.requires_grad_(False)
    pixel_predictor = PixelPredictor(student.spatial_dim // 2).to(device)
    spatial_predictor = PixelPredictor(student.spatial_dim).to(device)
    global_predictor = GlobalPredictor(student.embedding_dim).to(device)
    rgb_predictor = RGBReconstructionHead().to(device)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(pixel_predictor.parameters()) +
        list(spatial_predictor.parameters()) + list(global_predictor.parameters()) +
        list(rgb_predictor.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    visualization_dir = output / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    visual_images, visual_mean, visual_std = _make_visual_batch(args)
    history = []; start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["variant"] != args.variant:
            raise ValueError("resume checkpoint variant does not match --variant")
        if "rgb_predictor" not in state:
            raise ValueError("this checkpoint predates RGB reconstruction; start a new run or resume a reconstruction-enabled checkpoint")
        for key in ("dataset", "image_size", "seed", "epochs", "batch_size", "lr", "weight_decay", "patch_size", "mask_ratio",
                    "global_weight", "spatial_weight", "rgb_weight", "ema_start", "ema_end"):
            legacy_default = {"dataset": "cifar100", "image_size": 32}.get(key)
            saved_value = state["config"].get(key, legacy_default)
            if saved_value != getattr(args, key):
                raise ValueError(f"resume setting {key} differs from the checkpoint")
        student.load_state_dict(state["student"]); teacher.load_state_dict(state["teacher"])
        pixel_predictor.load_state_dict(state["pixel_predictor"])
        spatial_predictor.load_state_dict(state["spatial_predictor"])
        global_predictor.load_state_dict(state["global_predictor"])
        rgb_predictor.load_state_dict(state["rgb_predictor"])
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]; history = state["history"]
        loader_generator.set_state(state["loader_rng"]); _restore_rng(state["rng"])
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    steps_per_epoch = min(len(loader), args.limit_train_batches or len(loader))
    total_steps = max(steps_per_epoch * args.epochs, 1)
    dataset_name = "CIFAR-100 train split" if args.dataset == "cifar100" else "unlabeled ImageFolder"
    print(f"Unlabeled {dataset_name} images: {len(loader.dataset)}; image_size={args.image_size}; device={device_name}; batches/epoch={steps_per_epoch}", flush=True)
    began = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        student.train(); teacher.eval()
        epoch_start = time.perf_counter()
        total_sum = pixel_sum = spatial_sum = global_sum = rgb_sum = rgb_mae_sum = rgb_psnr_sum = teacher_std_sum = global_std_sum = 0.0
        batches = 0
        for batch_index, (images, _) in enumerate(loader):
            if args.limit_train_batches and batch_index >= args.limit_train_batches:
                break
            images = images.to(device, non_blocking=True)
            masked_images, mask = block_mask(images, args.patch_size, args.mask_ratio)
            student_out = student(masked_images, return_spatial=True)
            with torch.no_grad():
                teacher_out = teacher(images, return_spatial=True)
            pixel_prediction = pixel_predictor(student_out.stages[0])
            pixel_loss = _masked_cosine(pixel_prediction, teacher_out.stages[0], mask)
            spatial_prediction = spatial_predictor(student_out.spatial)
            spatial_loss = _masked_cosine(spatial_prediction, teacher_out.spatial, mask)
            global_prediction = global_predictor(student_out.embedding)
            global_loss = (1.0 - F.cosine_similarity(
                F.normalize(global_prediction.float(), dim=1),
                F.normalize(teacher_out.embedding.detach().float(), dim=1), dim=1,
            )).mean()
            rgb_predictions = rgb_predictor(student_out.stages, images.shape[-2:])
            rgb_loss = _masked_rgb_loss(rgb_predictions, images, mask)
            rgb_mean, rgb_psnr = _masked_rgb_metrics(
                rgb_predictions[-1], images, mask,
                (.5071, .4867, .4408) if args.dataset == "cifar100" else (.485, .456, .406),
                (.2675, .2565, .2761) if args.dataset == "cifar100" else (.229, .224, .225),
            )
            loss = (pixel_loss + args.spatial_weight * spatial_loss + args.global_weight * global_loss
                    + args.rgb_weight * rgb_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite pretraining loss at epoch {epoch + 1}, batch {batch_index + 1}")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            step = epoch * steps_per_epoch + batch_index + 1
            momentum = cosine_ema_momentum(step, total_steps, args.ema_start, args.ema_end)
            ema_update(teacher, student, momentum)
            total_sum += loss.item(); pixel_sum += pixel_loss.item()
            spatial_sum += spatial_loss.item(); global_sum += global_loss.item()
            rgb_sum += rgb_loss.item()
            rgb_mae_sum += rgb_mean; rgb_psnr_sum += rgb_psnr
            teacher_std_sum += teacher_out.stages[0].detach().float().std(unbiased=False).item()
            global_std_sum += teacher_out.embedding.detach().float().std(dim=0, unbiased=False).mean().item()
            batches += 1
            if batches % 50 == 0:
                print(f"  epoch {epoch+1}/{args.epochs} batch {batches}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "total_loss": total_sum / max(batches, 1),
            "local_pixel_cosine_loss": pixel_sum / max(batches, 1),
            "fused_spatial_cosine_loss": spatial_sum / max(batches, 1),
            "global_cosine_loss": global_sum / max(batches, 1),
            "masked_rgb_reconstruction_loss": rgb_sum / max(batches, 1),
            "masked_rgb_mae_01": rgb_mae_sum / max(batches, 1),
            "masked_rgb_psnr_db": rgb_psnr_sum / max(batches, 1),
            "teacher_feature_std": teacher_std_sum / max(batches, 1),
            "teacher_embedding_channel_std": global_std_sum / max(batches, 1),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row); print(json.dumps(row), flush=True)
        _save_state(output / "last.pt", student, teacher, pixel_predictor, spatial_predictor, global_predictor,
                    optimizer, scheduler, rgb_predictor, epoch + 1, history, loader_generator, args, device)
        torch.save(student.state_dict(), output / "encoder.pt")
        torch.save({
            "format_version": 1,
            "variant": args.variant,
            "widths": (32, 32, 48, 64),
            "high_resolution_fusion": True,
            "state_dict": student.state_dict(),
            "input_size": args.image_size,
            "normalization": {
                "mean": (.5071, .4867, .4408) if args.dataset == "cifar100" else (.485, .456, .406),
                "std": (.2675, .2565, .2761) if args.dataset == "cifar100" else (.229, .224, .225),
            },
            "pretraining_dataset": args.dataset,
            "pretraining_config": {
                "seed": args.seed,
                "epochs": args.epochs,
                "patch_size": args.patch_size,
                "mask_ratio": args.mask_ratio,
                "loss_weights": {
                    "local_pixel": 1.0,
                    "fused_spatial": args.spatial_weight,
                    "global": args.global_weight,
                    "masked_rgb": args.rgb_weight,
                },
            },
        }, output / "encoder_bundle.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2))
        completed_epoch = epoch + 1
        if completed_epoch == 1 or completed_epoch % args.visualize_every == 0 or completed_epoch == args.epochs:
            snapshot = visualization_dir / f"epoch_{completed_epoch:04d}.png"
            save_feature_snapshot(
                student, teacher, pixel_predictor, spatial_predictor, rgb_predictor, visual_images,
                visual_mean, visual_std, completed_epoch, snapshot, args.seed,
                patch_size=args.patch_size, mask_ratio=args.mask_ratio,
            )
            save_learning_curves(history, visualization_dir / "learning_curves.png")
            snapshots = sorted(visualization_dir.glob("epoch_*.png"))
            save_progress_gif(snapshots, visualization_dir / "feature_path_progress.gif")

    summary = {
        "variant": args.variant, "dataset": args.dataset, "image_size": args.image_size,
        "seed": args.seed, "epochs_completed": args.epochs,
        "encoder_parameters": sum(p.numel() for p in student.parameters()),
        "device": str(device), "device_name": device_name,
        "elapsed_seconds_this_invocation": time.perf_counter() - began,
        "configuration": vars(args), "warning": "Research pretraining checkpoint; transfer evidence is required before foundation-model claims.",
    }
    (output / "pretrain.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
