"""Reviewer R02 — seed-matched HR384 continuation.

Current locked scope:
- seed 123 only
- initialize from R01 Gentle CB Focal seed123 best checkpoint
- original AQUA20 distribution only
- ConvNeXt-Small, 384x384
- Gentle Class-Balanced Focal
- fixed 9 completed HR epochs
- scheduler horizon remains T_max=15
- best checkpoint selected strictly by validation Macro F1
- official test evaluated only after the best validation checkpoint is locked
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torchvision import datasets, transforms

import train_r01 as r01


SEED = 123
EXPECTED_SOURCE_SHA256 = (
    "e2c9a532dca693ec69ae97b6234b4e8952ab21103642dadc59c5ffcfef8df118"
)

HR_IMAGE_SIZE = 384
HR_RESIZE_SIZE = 440
BATCH_SIZE = 8
WORKERS = 0
LR = 5e-6
WEIGHT_DECAY = 1e-4
EPS = 1e-8
BETA = 0.999
GAMMA = 1.0
MAX_GRAD_NORM = 1.0

COMPLETED_HR_EPOCHS = 9
SCHEDULER_T_MAX = 15
ETA_MIN = 0.0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AQUA20 reviewer R02 HR384 continuation — seed123 only."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume seed123 from R02 last_checkpoint.pth.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Strict source/dataset/GPU + forward-only check, then exit.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_info() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
    }

    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        result["gpu_total_memory_bytes"] = int(props.total_memory)

        try:
            line = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            result["nvidia_smi"] = line
        except Exception as exc:
            result["nvidia_smi_error"] = str(exc)

    return result


def build_hr_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (HR_RESIZE_SIZE, HR_RESIZE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomResizedCrop(
                HR_IMAGE_SIZE,
                scale=(0.80, 1.00),
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
            transforms.ColorJitter(
                brightness=(0.88, 1.12),
                contrast=(0.88, 1.12),
                saturation=(0.88, 1.12),
                hue=(-0.03, 0.03),
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (HR_IMAGE_SIZE, HR_IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def load_datasets(data_root: Path):
    train_transform, eval_transform = build_hr_transforms()

    train_ds = datasets.ImageFolder(
        data_root / "train",
        transform=train_transform,
    )
    val_ds = datasets.ImageFolder(
        data_root / "val",
        transform=eval_transform,
    )
    test_ds = datasets.ImageFolder(
        data_root / "test",
        transform=eval_transform,
    )

    r01.validate_dataset_splits(train_ds, val_ds, test_ds)

    assert len(train_ds) == 5247
    assert len(val_ds) == 1312
    assert len(test_ds) == 1612
    assert len(train_ds.classes) == 20

    return train_ds, val_ds, test_ds


def verify_source_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
) -> tuple[dict[str, Any], str]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing source checkpoint: {checkpoint_path}"
        )

    digest = sha256_file(checkpoint_path)

    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "R01 Focal seed123 source SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_SOURCE_SHA256}\n"
            f"Actual:   {digest}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if int(checkpoint.get("seed", -1)) != SEED:
        raise RuntimeError(
            f"Source checkpoint seed mismatch: {checkpoint.get('seed')}"
        )

    if str(checkpoint.get("loss", "")).lower() != "focal":
        raise RuntimeError(
            f"Source checkpoint is not focal: {checkpoint.get('loss')}"
        )

    source_epoch = int(
        checkpoint.get(
            "best_epoch",
            checkpoint.get("epoch", -1),
        )
    )
    if source_epoch != 24:
        raise RuntimeError(
            f"Expected seed123 source best epoch 24; got {source_epoch}"
        )

    state = r01.extract_state_dict(checkpoint)

    if tuple(state["classifier.2.weight"].shape) != (20, 768):
        raise RuntimeError(
            "Unexpected classifier shape in source checkpoint."
        )

    model.load_state_dict(state, strict=True)

    return checkpoint, digest


def save_prediction_table(
    *,
    path: Path,
    dataset: datasets.ImageFolder,
    probabilities,
    targets,
    predictions,
) -> None:
    rows = []

    for index, (sample_path, _) in enumerate(dataset.samples):
        target = int(targets[index])
        pred = int(predictions[index])

        row = {
            "sample_index": index,
            "sample_path": sample_path,
            "target_index": target,
            "target_class": dataset.classes[target],
            "pred_index": pred,
            "pred_class": dataset.classes[pred],
            "confidence": float(probabilities[index][pred]),
        }

        for class_index, class_name in enumerate(dataset.classes):
            row[f"prob_{class_index:02d}_{class_name}"] = float(
                probabilities[index][class_index]
            )

        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    args = parse_args()

    r01.set_deterministic_seed(SEED)

    data_root = args.data_root.expanduser().resolve()
    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root missing: {data_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    run_root = output_root / "R02" / f"seed_{SEED}"
    weights_dir = run_root / "weights"
    tables_dir = run_root / "tables"
    logs_dir = run_root / "logs"

    weights_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = weights_dir / "best_model.pth"
    last_checkpoint_path = weights_dir / "last_checkpoint.pth"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_ds, val_ds, test_ds = load_datasets(data_root)

    train_loader, val_loader, test_loader = r01.build_loaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        prefetch_factor=2,
        seed=SEED,
        device=device,
    )

    model = r01.build_model(len(train_ds.classes))

    source_metadata, source_sha256 = verify_source_checkpoint(
        source_checkpoint,
        model,
    )

    model = model.to(device)

    class_weights = r01.compute_class_weights(
        train_ds,
        BETA,
    ).to(device)

    loss_fn = r01.FocalCrossEntropyLoss(
        class_weights=class_weights,
        gamma=GAMMA,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        eps=EPS,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SCHEDULER_T_MAX,
        eta_min=ETA_MIN,
    )

    scaler = torch.amp.GradScaler("cuda",
        enabled=device.type == "cuda"
    )

    class_counts = Counter(train_ds.targets)

    run_config: dict[str, Any] = {
        "experiment": "R02",
        "experiment_role": "reviewer_hr384_multiseed",
        "seed": SEED,
        "loss": "focal",
        "dataset_root": str(data_root),
        "output_root": str(output_root),
        "run_root": str(run_root),

        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "source_seed": int(source_metadata.get("seed", SEED)),
        "source_best_epoch": int(
            source_metadata.get(
                "best_epoch",
                source_metadata.get("epoch", 24),
            )
        ),
        "source_role": "R01 Gentle CB Focal seed123 224x224 best checkpoint",

        "image_size": HR_IMAGE_SIZE,
        "pre_resize": HR_RESIZE_SIZE,
        "batch_size": BATCH_SIZE,
        "workers": WORKERS,

        "completed_hr_epoch_budget": COMPLETED_HR_EPOCHS,
        "scheduler_t_max": SCHEDULER_T_MAX,

        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "eps": EPS,
        "eta_min": ETA_MIN,
        "beta": BETA,
        "gamma": GAMMA,
        "max_grad_norm": MAX_GRAD_NORM,

        "checkpoint_selection": "strict_highest_validation_macro_f1",
        "official_test_policy": (
            "test evaluated only after best validation checkpoint locked"
        ),

        "amp_enabled": device.type == "cuda",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,

        "resume_requested": args.resume,
        "sanity_check_requested": args.sanity_check,

        "train_transform": {
            "resize": [440, 440],
            "random_resized_crop": {
                "size": 384,
                "scale": [0.80, 1.00],
                "ratio": [0.75, 1.3333333333333333],
            },
            "horizontal_flip_p": 0.5,
            "rotation_degrees": 10,
            "color_jitter": {
                "brightness": [0.88, 1.12],
                "contrast": [0.88, 1.12],
                "saturation": [0.88, 1.12],
                "hue": [-0.03, 0.03],
            },
            "normalization": "ImageNet",
        },

        "eval_transform": {
            "resize": [384, 384],
            "normalization": "ImageNet",
        },

        "class_names": list(train_ds.classes),
        "class_counts": {
            name: int(class_counts[index])
            for index, name in enumerate(train_ds.classes)
        },

        "dataset_counts": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
            "classes": len(train_ds.classes),
        },

        "hardware_software": gpu_info(),
    }

    with (run_root / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_config, f, indent=2)

    initialization = {
        "verified_utc": utc_now(),
        "source_checkpoint": str(source_checkpoint),
        "sha256": source_sha256,
        "seed": SEED,
        "source_best_epoch": run_config["source_best_epoch"],
        "strict_state_dict_load": True,
        "classifier_shape": [20, 768],
    }

    with (run_root / "initialization.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(initialization, f, indent=2)

    print("=" * 78)
    print("AQUA20 REVIEWER R02 — HR384 SEED123")
    print("=" * 78)
    print(f"Device              : {device}")
    if torch.cuda.is_available():
        print(f"GPU                 : {torch.cuda.get_device_name(0)}")
    print(f"Train / Val / Test  : {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
    print(f"Classes             : {len(train_ds.classes)}")
    print(f"Source              : {source_checkpoint}")
    print(f"Source SHA-256      : {source_sha256}")
    print(f"Source best epoch   : {run_config['source_best_epoch']}")
    print(f"HR image size       : {HR_IMAGE_SIZE}")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"LR                  : {LR}")
    print(f"HR epoch budget     : {COMPLETED_HR_EPOCHS}")
    print(f"Scheduler T_max     : {SCHEDULER_T_MAX}")
    print("=" * 78)

    if args.sanity_check:
        model.eval()
        images, targets = next(iter(train_loader))
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.inference_mode():
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = loss_fn(logits, targets)

        if tuple(logits.shape) != (images.shape[0], 20):
            raise RuntimeError(
                f"Unexpected logits shape: {tuple(logits.shape)}"
            )

        print()
        print("R02 SEED123 SANITY CHECK: PASS")
        print(f"Batch shape          : {tuple(images.shape)}")
        print(f"Logits shape         : {tuple(logits.shape)}")
        print(f"Forward loss         : {float(loss.item()):.6f}")
        print("Optimizer step       : NOT PERFORMED")
        print("Training             : NOT STARTED")
        return 0

    if best_model_path.exists() and not args.resume:
        raise FileExistsError(
            f"Existing best checkpoint detected: {best_model_path}. "
            "Use --resume only for an interrupted existing R02 seed123 run."
        )

    if last_checkpoint_path.exists() and not args.resume:
        raise FileExistsError(
            f"Existing last checkpoint detected: {last_checkpoint_path}. "
            "Use --resume only for an interrupted existing R02 seed123 run."
        )

    history_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    best_val_macro_f1 = float("-inf")
    best_epoch = 0
    start_epoch = 1

    if args.resume:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"--resume requested but missing: {last_checkpoint_path}"
            )

        checkpoint = torch.load(
            last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if int(checkpoint.get("seed", -1)) != SEED:
            raise RuntimeError("Resume seed mismatch.")

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
        scaler.load_state_dict(
            checkpoint.get(
                "scaler_state_dict",
                scaler.state_dict(),
            )
        )

        best_val_macro_f1 = float(
            checkpoint.get(
                "best_val_macro_f1",
                best_val_macro_f1,
            )
        )
        best_epoch = int(
            checkpoint.get("best_epoch", 0)
        )
        start_epoch = int(
            checkpoint.get("epoch", 0)
        ) + 1

        history_csv = tables_dir / "training_history.csv"
        if history_csv.is_file():
            history_rows = pd.read_csv(
                history_csv
            ).to_dict(orient="records")

        validation_csv = tables_dir / "validation_metrics.csv"
        if validation_csv.is_file():
            validation_rows = pd.read_csv(
                validation_csv
            ).to_dict(orient="records")

        resume_record = {
            "resume_utc": utc_now(),
            "resume_from_epoch": start_epoch - 1,
            "next_epoch": start_epoch,
            "last_checkpoint": str(last_checkpoint_path),
            "last_checkpoint_sha256": sha256_file(
                last_checkpoint_path
            ),
        }

        with (run_root / "resume_evidence.jsonl").open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(resume_record) + "\n")

    run_started_utc = utc_now()
    run_started_perf = time.perf_counter()

    run_config["run_started_utc"] = run_started_utc
    with (run_root / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_config, f, indent=2)

    if start_epoch <= COMPLETED_HR_EPOCHS:
        for epoch in range(
            start_epoch,
            COMPLETED_HR_EPOCHS + 1,
        ):
            epoch_started = time.perf_counter()

            train_loss, train_top1 = r01.train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                scaler,
                device,
                MAX_GRAD_NORM,
            )

            scheduler.step()

            (
                val_metrics,
                _,
                _,
                _,
                val_loss,
            ) = r01.run_inference(
                model,
                val_loader,
                loss_fn,
                device,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            improved = (
                val_metrics["macro_f1"]
                > best_val_macro_f1
            )

            if improved:
                best_val_macro_f1 = float(
                    val_metrics["macro_f1"]
                )
                best_epoch = epoch

                best_payload = r01.checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_macro_f1=best_val_macro_f1,
                    best_epoch=best_epoch,
                    early_stop_counter=0,
                    seed=SEED,
                    loss_name="focal",
                    config=run_config,
                    train_metrics={
                        "train_loss": train_loss,
                        "train_top1": train_top1,
                    },
                    val_metrics=val_metrics,
                )
                torch.save(
                    best_payload,
                    best_model_path,
                )

            last_payload = r01.checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_macro_f1=best_val_macro_f1,
                best_epoch=best_epoch,
                early_stop_counter=0,
                seed=SEED,
                loss_name="focal",
                config=run_config,
                train_metrics={
                    "train_loss": train_loss,
                    "train_top1": train_top1,
                },
                val_metrics=val_metrics,
            )

            torch.save(
                last_payload,
                last_checkpoint_path,
            )

            epoch_runtime = (
                time.perf_counter() - epoch_started
            )

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
                "epoch_runtime_seconds": epoch_runtime,
                "resumed": args.resume,
            }

            validation_row = {
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

            history_rows.append(history_row)
            validation_rows.append(validation_row)

            pd.DataFrame(history_rows).to_csv(
                tables_dir / "training_history.csv",
                index=False,
            )
            pd.DataFrame(validation_rows).to_csv(
                tables_dir / "validation_metrics.csv",
                index=False,
            )

            pd.DataFrame(
                [
                    {
                        "split": "test",
                        "status": "pending_until_best_checkpoint_locked",
                        "best_epoch": best_epoch,
                        "best_val_macro_f1": best_val_macro_f1,
                    }
                ]
            ).to_csv(
                tables_dir / "test_metrics.csv",
                index=False,
            )

            print(
                f"Epoch {epoch:02d}/{COMPLETED_HR_EPOCHS} | "
                f"train_loss={train_loss:.4f} | "
                f"val_top1={val_metrics['top1']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
                f"best={best_val_macro_f1:.4f} "
                f"(epoch {best_epoch}) | "
                f"lr={current_lr:.8f} | "
                f"{epoch_runtime:.1f}s"
            )

    if not best_model_path.is_file():
        raise FileNotFoundError(
            "best_model.pth was not created."
        )

    best_checkpoint = torch.load(
        best_model_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        r01.extract_state_dict(best_checkpoint),
        strict=True,
    )
    model = model.to(device)

    (
        val_metrics,
        val_probabilities,
        val_targets,
        val_predictions,
        val_loss,
    ) = r01.run_inference(
        model,
        val_loader,
        loss_fn,
        device,
    )

    (
        test_metrics,
        test_probabilities,
        test_targets,
        test_predictions,
        test_loss,
    ) = r01.run_inference(
        model,
        test_loader,
        loss_fn,
        device,
    )

    pd.DataFrame(
        [
            {
                "split": "validation",
                "epoch": int(best_epoch),
                "loss": val_loss,
                **val_metrics,
                "seed": SEED,
            }
        ]
    ).to_csv(
        tables_dir / "best_validation_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "split": "test",
                "epoch": int(best_epoch),
                "best_epoch": int(best_epoch),
                "best_val_macro_f1": float(
                    best_val_macro_f1
                ),
                "loss": test_loss,
                **test_metrics,
                "checkpoint_path": str(best_model_path),
                "loss_name": "focal",
                "seed": SEED,
            }
        ]
    ).to_csv(
        tables_dir / "test_metrics.csv",
        index=False,
    )

    save_prediction_table(
        path=tables_dir / "validation_predictions.csv",
        dataset=val_ds,
        probabilities=val_probabilities,
        targets=val_targets,
        predictions=val_predictions,
    )

    save_prediction_table(
        path=tables_dir / "test_predictions.csv",
        dataset=test_ds,
        probabilities=test_probabilities,
        targets=test_targets,
        predictions=test_predictions,
    )

    run_finished_utc = utc_now()
    runtime_seconds = (
        time.perf_counter() - run_started_perf
    )

    checkpoint_hashes = [
        {
            "file": "best_model.pth",
            "sha256": sha256_file(best_model_path),
        },
        {
            "file": "last_checkpoint.pth",
            "sha256": sha256_file(last_checkpoint_path),
        },
    ]

    pd.DataFrame(checkpoint_hashes).to_csv(
        tables_dir / "checkpoint_sha256.csv",
        index=False,
    )

    run_config["run_finished_utc"] = run_finished_utc
    run_config["runtime_seconds_this_process"] = runtime_seconds
    run_config["best_epoch"] = int(best_epoch)
    run_config["best_validation_macro_f1"] = float(
        best_val_macro_f1
    )
    run_config["test_macro_f1"] = float(
        test_metrics["macro_f1"]
    )
    run_config["test_top1"] = float(
        test_metrics["top1"]
    )

    with (run_root / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_config, f, indent=2)

    print()
    print("=" * 78)
    print("R02 SEED123 COMPLETE")
    print("=" * 78)
    print(f"Run root              : {run_root}")
    print(f"Best epoch            : {best_epoch}")
    print(
        f"Best Val Macro F1     : "
        f"{best_val_macro_f1:.6f}"
    )
    print(
        f"Test Top-1            : "
        f"{test_metrics['top1']:.6f}"
    )
    print(
        f"Test Macro F1         : "
        f"{test_metrics['macro_f1']:.6f}"
    )
    print(
        f"Runtime this process  : "
        f"{runtime_seconds:.1f} sec"
    )
    print(
        f"best_model SHA-256    : "
        f"{checkpoint_hashes[0]['sha256']}"
    )
    print(
        f"last_checkpoint SHA   : "
        f"{checkpoint_hashes[1]['sha256']}"
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
