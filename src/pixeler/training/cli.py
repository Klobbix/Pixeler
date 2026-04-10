"""
CLI entry points for the Pixeler training subsystem.

Commands
--------
pixeler-capture
    Launch an interactive annotation session against a live game window.

    Usage::

        pixeler-capture --game "RuneScape" --dataset datasets/runescape --classes enemy,loot,npc
        pixeler-capture --game "MyGame" --dataset datasets/mygame --classes enemy --fps 0.5

    Options:
        --game      Substring of the game window title (required).
        --dataset   Path to the dataset directory (created if absent).
        --classes   Comma-separated list of class names (required).
        --fps       Frames per second for the Space-key auto-advance (default: 1.0).
        --backend   Window backend: "win32" (default) or "window".

pixeler-train
    Export the dataset to a YOLO-compatible layout and train a YOLOv8 model.
    Requires the [train] extra: ``uv sync --extra train``

    Usage::

        pixeler-train --dataset datasets/runescape --out models/runescape.onnx
        pixeler-train --dataset datasets/runescape --out models/runescape.onnx --epochs 100 --device cuda:0

    Options:
        --dataset   Path to the dataset directory (required).
        --out       Output path for the exported .onnx model (required).
        --epochs    Training epochs (default: 50).
        --imgsz     Input image size for YOLO (default: 640).
        --model     YOLOv8 model size: n/s/m/l/x (default: n).
        --device    Training device: cpu / cuda:0 / mps (default: cpu).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# pixeler-capture
# ---------------------------------------------------------------------------

def capture_main() -> None:
    parser = argparse.ArgumentParser(
        prog="pixeler-capture",
        description="Interactive annotation session for a live game window.",
    )
    parser.add_argument("--game",     required=True,
                        help="Substring of the game window title.")
    parser.add_argument("--dataset",  required=True,
                        help="Path to the dataset directory.")
    parser.add_argument("--classes",  required=True,
                        help="Comma-separated class names, e.g. 'enemy,loot,npc'.")
    parser.add_argument("--fps",      type=float, default=1.0,
                        help="Auto-advance frame rate when pressing Space (default: 1.0).")
    parser.add_argument("--backend",  choices=["win32", "window"], default="win32",
                        help="Window backend (default: win32).")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        print("Error: --classes must contain at least one class name.", file=sys.stderr)
        sys.exit(1)

    # Resolve window
    window = _open_window(args.game, args.backend)

    # Resolve dataset
    from pixeler.training.dataset import Dataset
    ds = Dataset(root=Path(args.dataset), classes=classes)
    print(f"Dataset: {ds}")

    # Run session
    from pixeler.training.capturer import CaptureSession
    session = CaptureSession(window=window, dataset=ds)
    session.run(fps=args.fps)


# ---------------------------------------------------------------------------
# pixeler-train
# ---------------------------------------------------------------------------

def train_main() -> None:
    parser = argparse.ArgumentParser(
        prog="pixeler-train",
        description="Train a YOLOv8 model on a Pixeler dataset. Requires uv sync --extra train.",
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to the dataset directory.")
    parser.add_argument("--out",     required=True,
                        help="Output path for the .onnx model file.")
    parser.add_argument("--epochs",  type=int, default=50,
                        help="Number of training epochs (default: 50).")
    parser.add_argument("--patience",  type=int, default=50,
                        help="Number of epochs to wait for improvement before early stopping (default: 50).")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="YOLO input image size (default: 640).")
    parser.add_argument("--model",   default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size (default: n = nano).")
    parser.add_argument("--device",  default="cpu",
                        help="Training device: cpu / cuda:0 / mps (default: cpu).")
    args = parser.parse_args()

    # Lazy import — only available with [train] extra
    try:
        from pixeler.training.trainer import YOLOTrainer
    except ImportError:
        print(
            "Error: Ultralytics is not installed.\n"
            "Run:  uv sync --extra train",
            file=sys.stderr,
        )
        sys.exit(1)

    from pixeler.training.dataset import Dataset
    ds = Dataset.load_existing(Path(args.dataset))
    print(f"Dataset: {ds}")

    trainer = YOLOTrainer(dataset=ds, model_size=f"yolov8{args.model}")
    onnx_path = trainer.train(
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        device=args.device,
        out_path=Path(args.out),
    )
    print(f"\nModel exported to: {onnx_path}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _open_window(title_substring: str, backend: str):
    """Return an AbstractWindow for the given game title substring."""
    if backend == "win32":
        from pixeler.window.win32_window import Win32Window
        try:
            return Win32Window(title_substring)
        except Exception as exc:
            print(f"Error: Could not find window '{title_substring}': {exc}",
                  file=sys.stderr)
            sys.exit(1)
    else:
        from pixeler.window.window import Window
        try:
            return Window(title_substring)
        except Exception as exc:
            print(f"Error: Could not find window '{title_substring}': {exc}",
                  file=sys.stderr)
            sys.exit(1)
