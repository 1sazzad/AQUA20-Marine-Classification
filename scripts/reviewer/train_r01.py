"""Reviewer training entry point for R01.

R01 is the controlled ConvNeXt-Small reviewer branch on the original AQUA20
train/validation/test split. The only scientific variable is the loss:

- cross-entropy for seeds 42, 123, and 2026
- gentle class-balanced focal for seeds 123 and 2026
- seed 42 focal reuses the frozen F03 checkpoint rather than retraining

The script writes one run directory per seed and loss under the configured
output root:

```
<AQUA20_OUTPUT_ROOT>/R01/<loss>/seed_<seed>/
```

Required artifacts are created under that run directory:

- weights/best_model.pth
- weights/last_checkpoint.pth
- tables/training_history.csv
- tables/validation_metrics.csv
- tables/test_metrics.csv
- run_config.json

The implementation is deterministic, resume-aware, AMP-enabled, and uses
validation Macro F1 for checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ConvNeXt_Small_Weights, convnext_small


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_SPLITS = {"train": 5247, "val": 1312, "test": 1612}
ALLOWED_SEEDS = {42, 123, 2026}
DEFAULT_MAX_EPOCHS = 40
DEFAULT_PATIENCE = 8
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_LR = 1e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPS = 1e-8
DEFAULT_IMAGE_SIZE = 224
DEFAULT_ETA_MIN = 0.0
DEFAULT_BETA = 0.999
DEFAULT_GAMMA = 1.0
DEFAULT_MAX_GRAD_NORM = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reviewer R01 runs.")
    parser.add_argument("--loss", choices=("ce", "focal"), required=True)
    parser.add_argument("--seed", type=int, choices=sorted(ALLOWED_SEEDS), required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--f03-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from last_checkpoint.pth if present.")
    parser.add_argument("--sanity-check", action="store_true", help="Run dataset/model/checkpoint sanity checks and exit.")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--eta-min", type=float, default=DEFAULT_ETA_MIN)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    return parser.parse_args()


def resolve_path(value: Path | None, env_name: str, *, must_exist: bool = True, create: bool = False) -> Path:
    raw_value = value if value is not None else os.getenv(env_name)
    if not raw_value:
        raise RuntimeError(f"Missing required path or environment variable: {env_name}")

    path = Path(raw_value).expanduser()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif must_exist and not path.exists():
        raise FileNotFoundError(f"{env_name} points to a missing path: {path}")
    return path


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                10,
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def load_split_datasets(data_root: Path, image_size: int) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    train_transform, eval_transform = build_transforms(image_size)
    train_ds = datasets.ImageFolder(data_root / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_transform)
    return train_ds, val_ds, test_ds


def validate_dataset_splits(train_ds: datasets.ImageFolder, val_ds: datasets.ImageFolder, test_ds: datasets.ImageFolder) -> None:
    split_lengths = {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    }
    for split_name, expected in EXPECTED_SPLITS.items():
        actual = split_lengths[split_name]
        if actual != expected:
            raise ValueError(f"{split_name} count mismatch: expected {expected}, found {actual}")

    train_classes = list(train_ds.classes)
    if len(train_classes) != 20:
        raise ValueError(f"Expected 20 classes in train split, found {len(train_classes)}")

    if val_ds.classes != train_classes:
        raise ValueError("Validation class order does not match train class order")
    if test_ds.classes != train_classes:
        raise ValueError("Test class order does not match train class order")


def compute_class_weights(train_ds: datasets.ImageFolder, beta: float) -> torch.Tensor:
    counts = Counter(train_ds.targets)
    class_counts = np.asarray([counts[index] for index in range(len(train_ds.classes))], dtype=np.float64)
    effective_num = 1.0 - np.power(beta, class_counts)
    raw_weights = (1.0 - beta) / np.clip(effective_num, 1e-12, None)
    normalized_weights = raw_weights / raw_weights.mean()
    return torch.tensor(normalized_weights, dtype=torch.float32)


def build_model(num_classes: int) -> nn.Module:
    model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, *, class_weights: torch.Tensor | None, gamma: float) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None, persistent=False)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal_factor = (1.0 - pt).pow(self.gamma)
        if self.class_weights is None:
            weights = 1.0
        else:
            weights = self.class_weights[targets]
        return (weights * focal_factor * ce).mean()


def compute_topk_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probabilities = torch.softmax(logits, dim=1)
    predicted = probabilities.argmax(dim=1)
    topk_indices = torch.topk(probabilities, k=5, dim=1).indices

    targets_np = targets.cpu().numpy()
    predicted_np = predicted.cpu().numpy()
    topk_np = topk_indices.cpu().numpy()

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        targets_np,
        predicted_np,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = precision_recall_fscore_support(
        targets_np,
        predicted_np,
        average="weighted",
        zero_division=0,
    )[2]

    return {
        "top1": float((predicted_np == targets_np).mean()),
        "top2": float(np.mean(np.any(topk_np[:, :2] == targets_np[:, None], axis=1))),
        "top3": float(np.mean(np.any(topk_np[:, :3] == targets_np[:, None], axis=1))),
        "top5": float(np.mean(np.any(topk_np[:, :5] == targets_np[:, None], axis=1))),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_batches = []
    target_batches = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            batch_loss = loss_fn(logits, targets)
            batch_size = targets.size(0)

            total_loss += float(batch_loss.item()) * batch_size
            total_examples += batch_size
            logits_batches.append(logits.detach().float().cpu())
            target_batches.append(targets.detach().cpu())

    logits = torch.cat(logits_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    metrics = compute_topk_metrics(logits, targets)
    average_loss = total_loss / max(total_examples, 1)
    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    return metrics, probabilities, targets.numpy(), predictions, average_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    max_grad_norm: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total_examples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        amp_context = (
            torch.amp.autocast(device_type=device.type, enabled=True)
            if device.type == "cuda"
            else nullcontext()
        )

        with amp_context:
            logits = model(images)
            loss = loss_fn(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.detach().argmax(dim=1) == targets).sum().item())
        total_examples += batch_size

    return total_loss / max(total_examples, 1), correct / max(total_examples, 1)


def checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    best_val_macro_f1: float,
    best_epoch: int,
    early_stop_counter: int,
    seed: int,
    loss_name: str,
    config: dict[str, Any],
    train_metrics: dict[str, float] | None = None,
    val_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
        "early_stop_counter": early_stop_counter,
        "seed": seed,
        "loss": loss_name,
        "config": config,
    }
    if train_metrics is not None:
        payload["train_metrics"] = train_metrics
    if val_metrics is not None:
        payload["validation_metrics"] = val_metrics
    return payload


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint format")


def write_csv(dataframe_rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(dataframe_rows).to_csv(path, index=False)


def build_loaders(
    train_ds: datasets.ImageFolder,
    val_ds: datasets.ImageFolder,
    test_ds: datasets.ImageFolder,
    *,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    eval_loader_kwargs = dict(loader_kwargs)
    eval_loader_kwargs.pop("worker_init_fn", None)
    eval_loader_kwargs.pop("persistent_workers", None)
    eval_loader_kwargs.pop("prefetch_factor", None)

    val_loader = DataLoader(val_ds, shuffle=False, **eval_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **eval_loader_kwargs)
    return train_loader, val_loader, test_loader


def sanity_check(
    *,
    run_root: Path,
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    checkpoint_config: dict[str, Any],
) -> None:
    probe_dir = run_root / "_sanity_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    images, targets = next(iter(train_loader))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    amp_context = (
        torch.amp.autocast(device_type=device.type, enabled=True)
        if device.type == "cuda"
        else nullcontext()
    )
    with amp_context:
        logits = model(images)
        loss = loss_fn(logits, targets)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_MAX_GRAD_NORM)
    scaler.step(optimizer)
    scaler.update()

    probe_checkpoint = checkpoint_payload(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        scaler=scaler,
        best_val_macro_f1=0.0,
        best_epoch=0,
        early_stop_counter=0,
        seed=int(checkpoint_config["seed"]),
        loss_name=str(checkpoint_config["loss"]),
        config=checkpoint_config,
        train_metrics={"sanity_check_loss": float(loss.item())},
    )
    probe_path = probe_dir / "probe_checkpoint.pth"
    torch.save(probe_checkpoint, probe_path)
    _ = torch.load(probe_path, map_location="cpu")
    shutil.rmtree(probe_dir)


def reuse_frozen_f03(
    *,
    f03_root: Path,
    run_root: Path,
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    config: dict[str, Any],
) -> None:
    source_checkpoint = f03_root / "weights" / "best_model.pth"
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Frozen F03 checkpoint missing: {source_checkpoint}")

    checkpoint = torch.load(source_checkpoint, map_location="cpu")
    model.load_state_dict(extract_state_dict(checkpoint))
    model = model.to(device)

    best_model_path = run_root / "weights" / "best_model.pth"
    last_checkpoint_path = run_root / "weights" / "last_checkpoint.pth"
    shutil.copy2(source_checkpoint, best_model_path)

    best_val_metrics, _, _, _, best_val_loss = run_inference(model, val_loader, loss_fn, device)
    test_metrics, test_probabilities, test_targets, test_predictions, test_loss = run_inference(model, test_loader, loss_fn, device)

    checkpoint_metadata = checkpoint_payload(
        epoch=int(checkpoint.get("epoch", checkpoint.get("best_epoch", 0))),
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], eps=config["eps"]),
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], eps=config["eps"]), T_max=config["max_epochs"], eta_min=config["eta_min"]),
        scaler=torch.cuda.amp.GradScaler(enabled=device.type == "cuda"),
        best_val_macro_f1=float(best_val_metrics["macro_f1"]),
        best_epoch=int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0))),
        early_stop_counter=0,
        seed=config["seed"],
        loss_name=config["loss"],
        config=config,
        val_metrics=best_val_metrics,
    )
    torch.save(checkpoint_metadata, last_checkpoint_path)

    history_rows = [
        {
            "epoch": int(checkpoint.get("epoch", checkpoint.get("best_epoch", 0))),
            "train_loss": math.nan,
            "train_top1": math.nan,
            "val_loss": float(best_val_loss),
            "val_top1": best_val_metrics["top1"],
            "val_top2": best_val_metrics["top2"],
            "val_top3": best_val_metrics["top3"],
            "val_top5": best_val_metrics["top5"],
            "val_macro_precision": best_val_metrics["macro_precision"],
            "val_macro_recall": best_val_metrics["macro_recall"],
            "val_macro_f1": best_val_metrics["macro_f1"],
            "val_weighted_f1": best_val_metrics["weighted_f1"],
            "lr": config["lr"],
            "best": True,
            "resumed": False,
            "reused_f03": True,
        }
    ]
    write_csv(history_rows, run_root / "tables" / "training_history.csv")
    write_csv(
        [
            {
                "epoch": row["epoch"],
                "val_loss": row["val_loss"],
                "val_top1": row["val_top1"],
                "val_top2": row["val_top2"],
                "val_top3": row["val_top3"],
                "val_top5": row["val_top5"],
                "val_macro_precision": row["val_macro_precision"],
                "val_macro_recall": row["val_macro_recall"],
                "val_macro_f1": row["val_macro_f1"],
                "val_weighted_f1": row["val_weighted_f1"],
                "best": True,
                "reused_f03": True,
            }
        ],
        run_root / "tables" / "validation_metrics.csv",
    )

    pd.DataFrame(
        [
            {
                "split": "test",
                "epoch": int(checkpoint.get("epoch", checkpoint.get("best_epoch", 0))),
                "loss": test_loss,
                **test_metrics,
                "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
                "best_epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0))),
                "reused_f03": True,
            }
        ]
    ).to_csv(run_root / "tables" / "test_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "loss": config["loss"],
                "seed": config["seed"],
                "reused_frozen_f03": True,
                "source_checkpoint": str(source_checkpoint),
                "best_epoch": int(checkpoint.get("epoch", checkpoint.get("best_epoch", 0))),
                "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
            }
        ]
    ).to_csv(run_root / "tables" / "frozen_reuse_summary.csv", index=False)

    with open(run_root / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def save_run_tables(
    *,
    run_root: Path,
    config: dict[str, Any],
    history_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_metrics_row: dict[str, Any],
) -> None:
    write_csv(history_rows, run_root / "tables" / "training_history.csv")
    write_csv(validation_rows, run_root / "tables" / "validation_metrics.csv")
    pd.DataFrame([test_metrics_row]).to_csv(run_root / "tables" / "test_metrics.csv", index=False)
    with open(run_root / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def main() -> int:
    args = parse_args()
    set_deterministic_seed(args.seed)

    data_root = resolve_path(args.data_root, "AQUA20_DATA_ROOT")
    f03_root = resolve_path(args.f03_root, "AQUA20_F03_ROOT")
    output_root = resolve_path(args.output_root, "AQUA20_OUTPUT_ROOT", must_exist=False, create=True)
    run_root = output_root / "R01" / args.loss / f"seed_{args.seed}"
    weights_dir = run_root / "weights"
    tables_dir = run_root / "tables"
    weights_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = weights_dir / "last_checkpoint.pth"
    best_model_path = weights_dir / "best_model.pth"

    train_ds, val_ds, test_ds = load_split_datasets(data_root, args.image_size)
    validate_dataset_splits(train_ds, val_ds, test_ds)
    train_loader, val_loader, test_loader = build_loaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(train_ds.classes)).to(device)

    if args.loss == "ce":
        class_weights = None
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    else:
        class_weights = compute_class_weights(train_ds, args.beta).to(device)
        loss_fn = FocalCrossEntropyLoss(class_weights=class_weights, gamma=args.gamma)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.eps,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=args.eta_min,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    run_config: dict[str, Any] = {
        "experiment": "R01",
        "loss": args.loss,
        "seed": args.seed,
        "dataset_root": str(data_root),
        "f03_root": str(f03_root),
        "output_root": str(output_root),
        "run_root": str(run_root),
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "prefetch_factor": args.prefetch_factor,
        "image_size": args.image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "eps": args.eps,
        "eta_min": args.eta_min,
        "beta": args.beta,
        "gamma": args.gamma,
        "max_grad_norm": args.max_grad_norm,
        "checkpoint_selection": "validation_macro_f1",
        "amp_enabled": device.type == "cuda",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "resume_requested": args.resume,
        "sanity_check_requested": args.sanity_check,
        "class_names": list(train_ds.classes),
        "class_counts": {name: int(Counter(train_ds.targets)[index]) for index, name in enumerate(train_ds.classes)},
        "frozen_f03_reuse": args.loss == "focal" and args.seed == 42,
    }

    if args.sanity_check:
        sanity_check(
            run_root=run_root,
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            checkpoint_config=run_config,
        )
        with open(run_root / "run_config.json", "w", encoding="utf-8") as file:
            json.dump(run_config, file, indent=2)
        print("Sanity check completed successfully.")
        print(f"Run root: {run_root}")
        return 0

    if args.loss == "focal" and args.seed == 42:
        reuse_frozen_f03(
            f03_root=f03_root,
            run_root=run_root,
            model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            loss_fn=loss_fn,
            config=run_config,
        )
        print(f"Frozen F03 checkpoint reused for focal seed 42 in {run_root}")
        return 0

    if last_checkpoint_path.exists() and not args.resume:
        raise FileExistsError(
            f"Existing run detected at {run_root}. Re-run with --resume to continue."
        )

    history_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    best_val_macro_f1 = float("-inf")
    best_epoch = 0
    early_stop_counter = 0
    start_epoch = 1

    if args.resume:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume was requested but no checkpoint exists at {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        if checkpoint.get("seed") != args.seed or checkpoint.get("loss") != args.loss:
            raise ValueError("Resume checkpoint does not match the requested seed/loss.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", scaler.state_dict()))
        best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", best_val_macro_f1))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        early_stop_counter = int(checkpoint.get("early_stop_counter", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

        history_csv = run_root / "tables" / "training_history.csv"
        if history_csv.is_file():
            history_rows = pd.read_csv(history_csv).to_dict(orient="records")

        validation_csv = run_root / "tables" / "validation_metrics.csv"
        if validation_csv.is_file():
            validation_rows = pd.read_csv(validation_csv).to_dict(orient="records")

    if start_epoch > args.max_epochs:
        print("Training already completed; evaluating the frozen best checkpoint.")
    else:
        for epoch in range(start_epoch, args.max_epochs + 1):
            train_loss, train_top1 = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                scaler,
                device,
                args.max_grad_norm,
            )
            scheduler.step()

            val_metrics, _, _, _, val_loss = run_inference(model, val_loader, loss_fn, device)
            current_lr = optimizer.param_groups[0]["lr"]

            improved = val_metrics["macro_f1"] > best_val_macro_f1
            if improved:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                early_stop_counter = 0
                best_payload = checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_macro_f1=best_val_macro_f1,
                    best_epoch=best_epoch,
                    early_stop_counter=early_stop_counter,
                    seed=args.seed,
                    loss_name=args.loss,
                    config=run_config,
                    train_metrics={"train_loss": train_loss, "train_top1": train_top1},
                    val_metrics=val_metrics,
                )
                torch.save(best_payload, best_model_path)
            else:
                early_stop_counter += 1

            last_payload = checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_macro_f1=best_val_macro_f1,
                best_epoch=best_epoch,
                early_stop_counter=early_stop_counter,
                seed=args.seed,
                loss_name=args.loss,
                config=run_config,
                train_metrics={"train_loss": train_loss, "train_top1": train_top1},
                val_metrics=val_metrics,
            )
            torch.save(last_payload, last_checkpoint_path)

            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "val_loss": val_loss,
                "val_top1": val_metrics["top1"],
                "val_top2": val_metrics["top2"],
                "val_top3": val_metrics["top3"],
                "val_top5": val_metrics["top5"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "lr": current_lr,
                "best": improved,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_macro_f1,
                "early_stop_counter": early_stop_counter,
                "resumed": args.resume,
            }
            history_rows.append(history_row)
            validation_rows.append(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_top1": val_metrics["top1"],
                    "val_top2": val_metrics["top2"],
                    "val_top3": val_metrics["top3"],
                    "val_top5": val_metrics["top5"],
                    "val_macro_precision": val_metrics["macro_precision"],
                    "val_macro_recall": val_metrics["macro_recall"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_weighted_f1": val_metrics["weighted_f1"],
                    "best": improved,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                }
            )

            save_run_tables(
                run_root=run_root,
                config=run_config,
                history_rows=history_rows,
                validation_rows=validation_rows,
                test_metrics_row={
                    "split": "test",
                    "status": "pending",
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                },
            )

            print(
                f"Epoch {epoch:02d}/{args.max_epochs} | "
                f"train_loss={train_loss:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f} | "
                f"best={best_val_macro_f1:.4f} | patience={early_stop_counter}/{args.patience}"
            )

            if early_stop_counter >= args.patience:
                print("Early stopping triggered.")
                break

    if not best_model_path.is_file():
        raise FileNotFoundError("best_model.pth was not created during training.")

    best_checkpoint = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(extract_state_dict(best_checkpoint))
    model = model.to(device)

    val_metrics, val_probabilities, val_targets, val_predictions, val_loss = run_inference(model, val_loader, loss_fn, device)
    test_metrics, test_probabilities, test_targets, test_predictions, test_loss = run_inference(model, test_loader, loss_fn, device)

    history_df = pd.DataFrame(history_rows)
    validation_df = pd.DataFrame(validation_rows)

    history_df.to_csv(run_root / "tables" / "training_history.csv", index=False)
    validation_df.to_csv(run_root / "tables" / "validation_metrics.csv", index=False)

    test_metrics_row = {
        "split": "test",
        "epoch": int(best_checkpoint.get("epoch", best_epoch)),
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_macro_f1),
        "loss": test_loss,
        **test_metrics,
        "checkpoint_path": str(best_model_path),
        "loss_name": args.loss,
        "seed": args.seed,
    }
    pd.DataFrame([test_metrics_row]).to_csv(run_root / "tables" / "test_metrics.csv", index=False)

    with open(run_root / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    print("Training complete.")
    print(f"Run root: {run_root}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro F1: {best_val_macro_f1:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
