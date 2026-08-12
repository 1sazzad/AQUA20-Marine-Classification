#!/usr/bin/env python3
"""Print the local CUDA and GPU environment summary for Azure GPU execution."""

from __future__ import annotations

import platform


def format_mib(value_bytes: int) -> str:
    return f"{value_bytes / (1024 ** 2):.1f} MiB"


def main() -> int:
    print(f"Python version: {platform.python_version()}")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        print("PyTorch version: unavailable")
        print("torchvision version: unavailable")
        print("CUDA availability: unavailable")
        print("CUDA runtime version: unavailable")
        print("cuDNN version: unavailable")
        print("GPU name: unavailable")
        print("GPU memory: unavailable")
        print(f"Import error: {exc}")
        return 1

    print(f"PyTorch version: {torch.__version__}")

    try:
        import torchvision

        print(f"torchvision version: {torchvision.__version__}")
    except Exception as exc:
        print("torchvision version: unavailable")
        print(f"Import error: {exc}")
        return 1

    cuda_available = torch.cuda.is_available()
    print(f"CUDA availability: {cuda_available}")
    print(f"CUDA runtime version: {torch.version.cuda if cuda_available else 'N/A'}")

    if not cuda_available:
        print("GPU name: N/A")
        print("GPU memory: N/A")
        return 1

    try:
        cudnn_version = torch.backends.cudnn.version()
        print(f"cuDNN version: {cudnn_version}")
    except Exception:
        print("cuDNN version: N/A")

    if torch.cuda.device_count() > 0:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU name: {properties.name}")
        print(f"GPU memory: {format_mib(properties.total_memory)}")

        try:
            from torchvision.models import convnext_small

            model = convnext_small(weights=None).to("cuda")
            model.eval()
            sample = torch.randn(1, 3, 224, 224, device="cuda")
            with torch.inference_mode():
                output = model(sample)
            print(f"torchvision model construction: ok ({tuple(output.shape)})")
        except Exception as exc:
            print("torchvision model construction: failed")
            print(f"Model error: {exc}")
            return 1
    else:
        print("GPU name: N/A")
        print("GPU memory: N/A")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
