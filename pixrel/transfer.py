from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from pixrel.models.checkpoint import load_pretrained_encoder
from pixrel.models.encoder import PixrelEncoder


def _transforms(size: int, mean, std):
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    evaluation = transforms.Compose([
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train, evaluation


def _loaders(args, size, mean, std):
    train_transform, eval_transform = _transforms(size, mean, std)
    train_base = datasets.ImageFolder(args.train_dir, transform=train_transform)
    if args.val_dir:
        val_base = datasets.ImageFolder(args.val_dir, transform=eval_transform)
        if train_base.class_to_idx != val_base.class_to_idx:
            raise ValueError("train and validation folders must contain the same class names")
        train_data = train_base
        val_data = val_base
    else:
        val_base = datasets.ImageFolder(args.train_dir, transform=eval_transform)
        n_val = max(1, round(len(train_base) * args.val_fraction))
        if n_val >= len(train_base):
            raise ValueError("need at least two images to create train and validation splits")
        generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(len(train_base), generator=generator).tolist()
        val_indices, train_indices = permutation[:n_val], permutation[n_val:]
        train_data = Subset(train_base, train_indices)
        val_data = Subset(val_base, val_indices)
    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=generator, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory)
    return train_loader, val_loader, train_base.class_to_idx


def _evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = 0.0
    batches = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images, classify=True)
            total_loss += nn.functional.cross_entropy(logits, labels).item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
            batches += 1
    return {"loss": total_loss / max(batches, 1), "accuracy": correct / max(total, 1)}


def train_main():
    parser = argparse.ArgumentParser(description="Fine-tune or linear-probe a PIXREL encoder on ImageFolder classes")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="PIXREL encoder_bundle.pt or compatible checkpoint")
    source.add_argument("--scratch", action="store_true", help="initialize a same-architecture model from random weights")
    parser.add_argument("--train-dir", required=True, help="ImageFolder root with one subdirectory per class")
    parser.add_argument("--val-dir", help="optional separate ImageFolder validation root")
    parser.add_argument("--output", default="./outputs/transfer")
    parser.add_argument("--image-size", type=int, help="defaults to the encoder bundle input size")
    parser.add_argument("--variant", choices=["adaptive", "fixed", "conv"], default="adaptive",
                        help="encoder variant used with --scratch")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--freeze-encoder", action="store_true", help="train only the new classifier head")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    _train(args)


def _train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))

    if args.scratch:
        size = args.image_size or 128
        mean, std = (.485, .456, .406), (.229, .224, .225)
        preprocessing = {"input_size": size, "normalization": {"mean": mean, "std": std}}
    else:
        _, preprocessing = load_pretrained_encoder(args.checkpoint)
        size = args.image_size or preprocessing["input_size"]
        mean = tuple(preprocessing["normalization"]["mean"])
        std = tuple(preprocessing["normalization"]["std"])
    train_loader, val_loader, class_to_idx = _loaders(args, size, mean, std)
    if args.scratch:
        model = PixrelEncoder(num_classes=len(class_to_idx), variant=args.variant)
    else:
        model, _ = load_pretrained_encoder(args.checkpoint, num_classes=len(class_to_idx))
    model.to(device)
    if args.freeze_encoder:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("classifier."))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    history = []
    began = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images, classify=True), labels)
            loss.backward(); optimizer.step()
            loss_sum += loss.item(); batches += 1
        metrics = _evaluate(model, val_loader, device)
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": loss_sum / max(batches, 1),
               "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"],
               "lr": scheduler.get_last_lr()[0]}
        history.append(row); print(json.dumps(row), flush=True)
        state = {
            "model": model.state_dict(), "variant": model.variant, "seed": args.seed,
            "class_to_idx": class_to_idx, "preprocessing": {"input_size": size, "normalization": {"mean": mean, "std": std}},
            "config": vars(args), "epoch": epoch + 1, "source": "scratch" if args.scratch else str(args.checkpoint),
        }
        torch.save(state, output / "last.pt")
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            torch.save(state, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2))
    summary = {"seed": args.seed, "variant": model.variant,
               "source": "scratch" if args.scratch else str(args.checkpoint), "classes": len(class_to_idx),
               "best_val_accuracy": best_accuracy, "device": str(device),
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
               "elapsed_seconds": time.perf_counter() - began, "config": vars(args)}
    (output / "run.json").write_text(json.dumps(summary, indent=2))
    (output / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2))
    print(json.dumps(summary), flush=True)


def evaluate_main():
    parser = argparse.ArgumentParser(description="Evaluate a PIXREL ImageFolder transfer checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True, help="ImageFolder root with one subdirectory per class")
    parser.add_argument("--output", default="./outputs/transfer/eval.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_to_idx = state["class_to_idx"]
    model, _ = load_pretrained_encoder(args.checkpoint, num_classes=len(class_to_idx))
    model.load_state_dict(state["model"]); model.to(device)
    size = state["preprocessing"]["input_size"]
    norm = state["preprocessing"]["normalization"]
    _, eval_transform = _transforms(size, norm["mean"], norm["std"])
    dataset = datasets.ImageFolder(args.data_dir, transform=eval_transform)
    if dataset.class_to_idx != class_to_idx:
        raise ValueError("evaluation folder classes do not match the checkpoint class_to_idx")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    result = {**_evaluate(model, loader, device), "checkpoint": str(args.checkpoint),
              "examples": len(dataset), "classes": len(class_to_idx), "device": str(device)}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)); print(json.dumps(result))
