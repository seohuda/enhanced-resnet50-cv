import argparse
import csv
import json
import logging
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from config import load_config, merge_config
from model import build_model
from utils import SoftLabelCrossEntropyLoss, cutmix_data, set_seed


def setup_logging(output_dir: str) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ResNet50 variants on CIFAR-100")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file (CLI flags override YAML values)")
    parser.add_argument("--model", type=str, default="se_resnet50",
                        choices=["resnet50", "se_resnet50"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--cutmix", action="store_true", help="Enable CutMix augmentation")
    parser.add_argument("--cutmix-prob", type=float, default=0.5)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="runs/default")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of training data to use for validation")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser


def explicit_cli_keys() -> set:
    probe = build_parser()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    return set(vars(probe.parse_args()).keys())


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        yaml_dict = load_config(args.config)
        args = merge_config(args, explicit_cli_keys(), yaml_dict)
    return args


def get_dataloaders(
    args: argparse.Namespace, device: torch.device
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    full_train_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )

    num_train = len(full_train_dataset)
    num_val = int(num_train * args.val_split)
    num_train = num_train - num_val

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(full_train_dataset), generator=generator).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = Subset(full_train_dataset, train_indices)

    val_base_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=False, transform=test_transform
    )
    val_dataset = Subset(val_base_dataset, val_indices)

    test_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=args.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=args.num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=args.num_workers > 0
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    soft_criterion = SoftLabelCrossEntropyLoss()

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type, enabled=args.amp):
            r = np.random.rand(1)
            if args.cutmix and r < args.cutmix_prob:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, args.cutmix_alpha)
                outputs = model(inputs)
                loss = soft_criterion(outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                targets_a = targets
                targets_b = targets
                lam = 1.0

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a).sum().float()
                    + (1 - lam) * predicted.eq(targets_b).sum().float()).item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool = False
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_top1 += predicted.eq(targets).sum().item()

            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

    epoch_loss = running_loss / total
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    return epoch_loss, top1_acc, top5_acc


def save_checkpoint(state: dict, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model} | CutMix: {args.cutmix} | Seed: {args.seed}")

    train_loader, val_loader, test_loader = get_dataloaders(args, device)
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = build_model(args.model, num_classes=100).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {num_params:,}")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs]
    )

    scaler = torch.amp.GradScaler(device.type) if args.amp else None

    start_epoch = 1
    best_val_acc = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Resuming from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            if scaler is not None and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info(f"Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")
        else:
            logger.info(f"Checkpoint not found: {args.resume}")
            return

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    csv_exists = os.path.isfile(csv_path) and args.resume
    csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_top1", "val_top5", "lr", "time_s"])

    logger.info(f"\n{'Epoch':>5} {'TrLoss':>7} {'TrAcc':>6} {'VaLoss':>7} {'VaTop1':>6} {'VaTop5':>6} {'LR':>8}")
    logger.info("-" * 55)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, args, scaler)
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, device, args.amp)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        csv_writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{val_loss:.4f}", f"{val_top1:.2f}", f"{val_top5:.2f}",
                             f"{lr:.6f}", f"{elapsed:.1f}"])
        csv_file.flush()

        logger.info(f"{epoch:>5d} {train_loss:>7.4f} {train_acc:>6.2f} {val_loss:>7.4f} "
                    f"{val_top1:>6.2f} {val_top5:>6.2f} {lr:>8.6f}")

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config,
            "num_params": num_params,
        }
        if scaler is not None:
            checkpoint_state["scaler_state_dict"] = scaler.state_dict()

        save_checkpoint(checkpoint_state, os.path.join(args.output_dir, "last.pth"))

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            checkpoint_state["best_val_acc"] = best_val_acc
            save_checkpoint(checkpoint_state, os.path.join(args.output_dir, "best.pth"))

    csv_file.close()

    logger.info(f"\nBest Validation Top-1: {best_val_acc:.2f}%")
    logger.info("\nEvaluating on test set (final evaluation)...")
    test_loss, test_top1, test_top5 = evaluate(model, test_loader, device, args.amp)
    logger.info(f"Test Top-1: {test_top1:.2f}% | Test Top-5: {test_top5:.2f}%")

    best_ckpt = torch.load(os.path.join(args.output_dir, "best.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss_best, test_top1_best, test_top5_best = evaluate(model, test_loader, device, args.amp)
    logger.info(f"Test Top-1 (best ckpt): {test_top1_best:.2f}% | Test Top-5: {test_top5_best:.2f}%")

    summary = {
        "model": args.model,
        "use_cutmix": args.cutmix,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_val_top1": best_val_acc,
        "test_top1_last": test_top1,
        "test_top5_last": test_top5,
        "test_top1_best": test_top1_best,
        "test_top5_best": test_top5_best,
        "num_params": num_params,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
