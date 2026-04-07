"""
Pixeler training subsystem — dataset capture and model training utilities.

Quick reference::

    from pixeler.training import Dataset, BoundingBox, CaptureSession, YOLOTrainer

    # 1. Capture labeled screenshots
    ds = Dataset(Path("datasets/mygame"), classes=["enemy", "loot"])
    session = CaptureSession(window=Win32Window("MyGame"), dataset=ds)
    session.run(fps=0.5)

    # 2. Train (requires: uv sync --extra train)
    trainer = YOLOTrainer(dataset=ds)
    onnx = trainer.train(epochs=50, out_path=Path("models/mygame.onnx"))

    # 3. ORB sprite library (no training needed)
    from pixeler.training import ORBTrainer
    orb = ORBTrainer(Path("sprites/mygame"))
    orb.add_reference("health_orb", crop_image)
    matcher = orb.load_matcher("health_orb")
"""

from pixeler.training.dataset import BoundingBox, Dataset, LabeledImage
from pixeler.training.capturer import CaptureSession
from pixeler.training.trainer import YOLOTrainer, ORBTrainer

__all__ = [
    "BoundingBox",
    "Dataset",
    "LabeledImage",
    "CaptureSession",
    "YOLOTrainer",
    "ORBTrainer",
]
