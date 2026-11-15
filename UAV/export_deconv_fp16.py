"""Fuse trained DEConv branches and save a compact FP16 inference checkpoint."""

import argparse
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules import DEConv


def find_latest_best_map50() -> Path:
    """Find the newest mAP50-best checkpoint in the project's training runs."""
    run_root = ROOT / "UAV/UAV2_3/runs/Yolov11n/Yolov11n-FPN-DEConv-FocalEIoULoss"
    candidates = list(run_root.glob("train*/weights/best_mAP50.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best_mAP50.pt was found under {run_root}. Pass a trained checkpoint path explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def export_fp16(weights: Path, output: Path) -> None:
    checkpoint = torch.load(weights, map_location="cpu")
    model = checkpoint.get("ema") or checkpoint["model"]
    model = model.float().eval().fuse()
    model.half()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    compact = {
        "model": model,
        "ema": None,
        "train_args": checkpoint.get("train_args", {}),
        "epoch": checkpoint.get("epoch", -1),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, output)
    print(f"Source: {weights} ({weights.stat().st_size / 1e6:.1f} MB)")
    print(f"FP16 fused: {output} ({output.stat().st_size / 1e6:.1f} MB)")
    print(f"Remaining DEConv branches: {sum(isinstance(m, DEConv) and not m.deploy for m in model.modules())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "weights",
        type=Path,
        nargs="?",
        help="Trained DEConv checkpoint, preferably best_mAP50.pt; defaults to the newest project checkpoint",
    )
    parser.add_argument("--output", type=Path, help="Output .pt path")
    args = parser.parse_args()
    weights = args.weights or find_latest_best_map50()
    destination = args.output or weights.with_name(f"{weights.stem}_fused_fp16.pt")
    print(f"Using checkpoint: {weights}")
    export_fp16(weights, destination)
