"""
Training pipeline for game object detectors.

Two trainers are provided:

YOLOTrainer
    Wraps Ultralytics YOLOv8 to train a multi-class detector and export it to
    ONNX for use with ``YOLOClassifier``.  Requires the optional ``[train]``
    dependency group::

        uv sync --extra train

ORBTrainer
    No gradient training required.  Manages a library of reference sprite images
    on disk.  Each saved image can be loaded directly into an ``ORBMatcher``
    for feature-based detection without any model training.

Typical flow::

    # --- one-time training ---
    from pathlib import Path
    from pixeler.training.dataset import Dataset
    from pixeler.training.trainer import YOLOTrainer

    ds = Dataset.load_existing(Path("datasets/mygame"))
    trainer = YOLOTrainer(dataset=ds)
    onnx = trainer.train(epochs=100, device="cuda:0", out_path=Path("models/mygame.onnx"))

    # --- at bot runtime ---
    from pixeler.vision.classifier import YOLOClassifier
    yolo = YOLOClassifier(onnx, class_names=ds.classes)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

import cv2

from pixeler.training.dataset import Dataset


# ---------------------------------------------------------------------------
# YOLO trainer (requires ultralytics extra)
# ---------------------------------------------------------------------------

class YOLOTrainer:
    """
    Train a YOLOv8 model on a Pixeler ``Dataset`` and export it to ONNX.

    Training artifacts (checkpoints, curves, confusion matrices) are written
    to ``training_runs/<dataset_name>/`` in the current working directory so
    you can inspect them after the run.

    :param dataset:     A ``Dataset`` instance (must have at least a few images).
    :param model_size:  YOLOv8 variant — one of ``yolov8n``, ``yolov8s``,
                        ``yolov8m``, ``yolov8l``, ``yolov8x``.
                        **n** (nano) is recommended for game bots: fast
                        inference, small enough for CPU, good accuracy on
                        simple sprite classes.
    """

    def __init__(self, dataset: Dataset, model_size: str = "yolov8n"):
        self._dataset = dataset
        self._model_size = model_size

    def train(
        self,
        epochs: int = 50,
        imgsz: int = 640,
        device: str = "cpu",
        out_path: Optional[Path] = None,
    ) -> Path:
        """
        Export the dataset, train YOLOv8, and save the resulting ONNX model.

        Training artifacts land in ``training_runs/<dataset_name>/``.
        The final ``.onnx`` file is copied to ``out_path`` (or returned in-place
        if ``out_path`` is None).

        :param epochs:   Number of training epochs.
        :param imgsz:    Square input resolution (pixels) for YOLO.
        :param device:   ``"cpu"``, ``"cuda:0"``, ``"mps"``, etc.
        :param out_path: Where to write the exported ``.onnx`` file.
                         If None the file is left alongside ``best.pt`` in the
                         training run directory and its path is returned.
        :returns: Path to the final ``.onnx`` model file.
        :raises ImportError: if Ultralytics is not installed.
        """
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            raise ImportError(
                "Ultralytics is not installed.\n"
                "Run:  uv sync --extra train"
            )

        run_root = Path("training_runs") / self._dataset.root.name

        # 1. Export dataset to YOLO folder layout + data.yaml
        data_dir = run_root / "data"
        print(f"[YOLOTrainer] Exporting dataset to {data_dir} …")
        yaml_path = self._dataset.export_yolo_structure(data_dir)
        print(f"[YOLOTrainer] data.yaml written: {yaml_path}")

        # 2. Train
        yolo_project = run_root / "yolo"
        print(
            f"[YOLOTrainer] Training {self._model_size}  "
            f"epochs={epochs}  imgsz={imgsz}  device={device}"
        )
        model = YOLO(f"{self._model_size}.pt")
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            project=str(yolo_project),
            name="train",
            exist_ok=True,   # overwrite previous run in the same directory
            verbose=True,
        )

        # 3. Locate best checkpoint
        save_dir = Path(results.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(
                f"Training completed but best.pt not found at {best_pt}."
            )
        print(f"[YOLOTrainer] Best checkpoint: {best_pt}")

        # 4. Export to ONNX
        return self.export_to_onnx(best_pt, out_path)

    @staticmethod
    def export_to_onnx(pt_path: Path, out_path: Optional[Path] = None) -> Path:
        """
        Export an existing ``.pt`` checkpoint to ONNX without retraining.

        :param pt_path:  Path to a YOLOv8 ``.pt`` weights file.
        :param out_path: Destination for the ``.onnx`` file.  If None the
                         file is placed alongside ``pt_path``.
        :returns: Path to the exported ``.onnx`` file.
        :raises ImportError: if Ultralytics is not installed.
        """
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            raise ImportError(
                "Ultralytics is not installed.\n"
                "Run:  uv sync --extra train"
            )

        pt_path = Path(pt_path)
        print(f"[YOLOTrainer] Exporting {pt_path.name} → ONNX …")
        model = YOLO(str(pt_path))
        model.export(format="onnx")

        # Ultralytics writes the ONNX next to the .pt file
        onnx_src = pt_path.with_suffix(".onnx")
        if not onnx_src.exists():
            raise FileNotFoundError(
                f"ONNX export succeeded but file not found at {onnx_src}."
            )

        if out_path is None:
            print(f"[YOLOTrainer] ONNX saved: {onnx_src}")
            return onnx_src

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_src, out_path)
        print(f"[YOLOTrainer] ONNX copied to: {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# ORB sprite library (no training required)
# ---------------------------------------------------------------------------

class ORBTrainer:
    """
    Manages a directory of reference sprite images for use with ``ORBMatcher``.

    ORB is feature-based — no gradient training is needed.  You simply save a
    clean crop of the target object (an icon, UI element, or sprite), then load
    it at runtime as the matcher's reference.

    Sprites are stored as ``<sprite_dir>/<name>.png``.

    Usage::

        orb_lib = ORBTrainer(Path("sprites/mygame"))

        # During capture / setup:
        orb_lib.add_reference("health_orb", health_orb_crop)

        # At bot runtime:
        matcher = orb_lib.load_matcher("health_orb")
        detection = matcher.find(screenshot)
    """

    def __init__(self, sprite_dir: Path):
        """
        :param sprite_dir: Directory that holds (or will hold) reference images.
                           Created automatically if it does not exist.
        """
        self.sprite_dir = Path(sprite_dir)
        self.sprite_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Reference management
    # ------------------------------------------------------------------

    def add_reference(self, name: str, image: cv2.Mat) -> Path:
        """
        Save a reference image to the sprite library.

        The image should be a tight crop of the object you want to detect —
        the cleaner and more distinctive the texture, the better ORB performs.

        :param name:  Logical name for this sprite (e.g. ``"health_orb"``).
                      Used as the filename stem.
        :param image: BGR cv2.Mat of the reference crop.
        :returns:     Path to the saved ``.png`` file.
        """
        path = self._path_for(name)
        cv2.imwrite(str(path), image)
        print(f"[ORBTrainer] Saved reference '{name}' → {path}")
        return path

    def add_reference_from_file(self, name: str, src_path: Path) -> Path:
        """
        Copy an existing image file into the sprite library under *name*.

        :param name:     Logical name for this sprite.
        :param src_path: Source image file to copy.
        :returns:        Path to the saved ``.png`` file.
        """
        src_path = Path(src_path)
        img = cv2.imread(str(src_path))
        if img is None:
            raise FileNotFoundError(f"Source image not found: {src_path}")
        return self.add_reference(name, img)

    def remove_reference(self, name: str) -> None:
        """Delete a named reference from the sprite library."""
        path = self._path_for(name)
        if path.exists():
            path.unlink()
            print(f"[ORBTrainer] Removed reference '{name}'.")
        else:
            print(f"[ORBTrainer] Reference '{name}' not found — nothing removed.")

    def list_references(self) -> List[str]:
        """Return a sorted list of all sprite names in the library."""
        return sorted(p.stem for p in self.sprite_dir.glob("*.png"))

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    def load_matcher(
        self,
        name: str,
        min_matches: int = 12,
        max_features: int = 500,
    ):
        """
        Return an ``ORBMatcher`` pre-loaded with the named reference image.

        :param name:         Name of the reference sprite to load.
        :param min_matches:  Passed through to ``ORBMatcher.__init__``.
        :param max_features: Passed through to ``ORBMatcher.__init__``.
        :returns: Ready-to-use ``ORBMatcher`` instance.
        :raises FileNotFoundError: if the named sprite does not exist.
        """
        from pixeler.vision.classifier import ORBMatcher

        path = self._path_for(name)
        if not path.exists():
            raise FileNotFoundError(
                f"No reference named '{name}' in {self.sprite_dir}. "
                f"Available: {self.list_references()}"
            )
        matcher = ORBMatcher(min_matches=min_matches, max_features=max_features)
        matcher.load_reference_from_file(path)
        return matcher

    def load_all_matchers(
        self,
        min_matches: int = 12,
        max_features: int = 500,
    ) -> dict:
        """
        Load every sprite in the library into its own ``ORBMatcher``.

        :returns: ``{name: ORBMatcher}`` dict for all sprites in the library.
        """
        return {
            name: self.load_matcher(name, min_matches, max_features)
            for name in self.list_references()
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path_for(self, name: str) -> Path:
        return self.sprite_dir / f"{name}.png"

    def __repr__(self) -> str:
        sprites = self.list_references()
        return f"ORBTrainer(sprite_dir='{self.sprite_dir}', sprites={sprites})"
