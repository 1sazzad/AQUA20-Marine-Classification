#!/usr/bin/env python3
"""Validate the reviewer environment setup without running training."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


REQUIRED_ENV = (
    "AQUA20_DATA_ROOT",
    "AQUA20_F03_ROOT",
    "AQUA20_OUTPUT_ROOT",
)

TRAIN_COUNTS = {"train": 5247, "val": 1312, "test": 1612}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def require_env(name: str) -> Path:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} points to a missing path: {path}")
    return path


def verify_split_dir(data_root: Path) -> None:
    split_dirs = {}

    for split_name in TRAIN_COUNTS:
        split_dir = data_root / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing {split_name} directory: {split_dir}")
        split_dirs[split_name] = split_dir

    for split_name, expected_count in TRAIN_COUNTS.items():
        split_path = split_dirs[split_name]
        actual_count = sum(
            1
            for file_path in split_path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES
        )
        if actual_count != expected_count:
            raise ValueError(
                f"{split_name} file count mismatch: expected {expected_count}, found {actual_count} in {split_path}"
            )

    train_class_names = sorted(p.name for p in split_dirs["train"].iterdir() if p.is_dir())
    if len(train_class_names) != 20:
        raise ValueError(f"Expected 20 classes in train split, found {len(train_class_names)}")

    for split_name in ("train", "val", "test"):
        split_path = split_dirs[split_name]
        class_names = sorted(p.name for p in split_path.iterdir() if p.is_dir())
        if len(class_names) != 20:
            raise ValueError(f"Expected 20 classes in {split_name}, found {len(class_names)}")
        if class_names != train_class_names:
            raise ValueError(f"Class folder mismatch in {split_name}")


def verify_f03_root(f03_root: Path) -> None:
    best_model = f03_root / "weights" / "best_model.pth"
    summary_csv = f03_root / "tables" / "F03_test_metrics_summary.csv"
    config_json = f03_root / "tables" / "F03_training_config.json"

    if not best_model.is_file():
        raise FileNotFoundError(f"Missing frozen F03 best_model.pth: {best_model}")
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Missing frozen F03 summary CSV: {summary_csv}")
    if not config_json.is_file():
        raise FileNotFoundError(f"Missing frozen F03 config JSON: {config_json}")


def create_reviewer_output_root(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    probe = output_root / ".write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()


def main() -> int:
    print("Checking reviewer environment ...")

    env_values = {name: require_env(name) for name in REQUIRED_ENV}

    data_root = env_values["AQUA20_DATA_ROOT"]
    f03_root = env_values["AQUA20_F03_ROOT"]
    output_root = env_values["AQUA20_OUTPUT_ROOT"]

    verify_split_dir(data_root)
    verify_f03_root(f03_root)
    create_reviewer_output_root(output_root)

    print("AQUA20 data root:", data_root)
    print("AQUA20 F03 root:", f03_root)
    print("AQUA20 reviewer output root:", output_root)
    print("All reviewer environment checks passed.")
    print("No training was run.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - kept for user-friendly CLI validation
        print(f"Reviewer environment check failed: {exc}")
        raise SystemExit(1)
