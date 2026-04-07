"""
Dataset management for training game object detectors.

Disk layout::

    datasets/
      my_game/
        classes.json          # ["enemy", "health_bar", ...]
        images/
          000001.png
          000002.png
        labels/               # YOLO format: class_id cx cy w h (normalized)
          000001.txt
          000002.txt
        metadata.json         # window title, capture settings, created_at

Coordinate convention
---------------------
YOLO labels use **center-x, center-y, width, height** — all normalized to
[0, 1] relative to the image dimensions.  ``BoundingBox`` stores this format
and provides helpers to convert to/from pixel (top-left x, y, w, h) space.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """
    A labeled bounding box in normalized YOLO coordinates.

    All spatial values (cx, cy, w, h) are in the range [0, 1] relative to
    the image dimensions — NOT pixel values.
    """
    class_id: int
    class_name: str
    cx: float   # center-x normalized
    cy: float   # center-y normalized
    w: float    # width normalized
    h: float    # height normalized

    def to_yolo_line(self) -> str:
        """Serialize to a single YOLO label file line."""
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    def to_pixel_rect(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """
        Convert to pixel (x, y, w, h) top-left format for OpenCV drawing.

        :returns: (x, y, w, h) in pixel coordinates.
        """
        pw = int(self.w * img_w)
        ph = int(self.h * img_h)
        px = int(self.cx * img_w - pw / 2)
        py = int(self.cy * img_h - ph / 2)
        return px, py, pw, ph

    @classmethod
    def from_pixel_rect(
        cls,
        class_id: int,
        class_name: str,
        x: int,
        y: int,
        w: int,
        h: int,
        img_w: int,
        img_h: int,
    ) -> BoundingBox:
        """
        Build a BoundingBox from pixel top-left coordinates (x, y, w, h),
        normalizing against the image dimensions.
        """
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        return cls(class_id=class_id, class_name=class_name, cx=cx, cy=cy, w=nw, h=nh)


@dataclass
class LabeledImage:
    """A captured screenshot paired with its annotation file."""
    image_path: Path
    label_path: Path
    boxes: List[BoundingBox] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset:
    """
    Manages a directory of labeled game screenshots for YOLO training.

    Typical flow::

        ds = Dataset(Path("datasets/mygame"), classes=["enemy", "npc"])
        ds.add_sample(screenshot_mat, [box1, box2])
        yaml_path = ds.export_yolo_structure(Path("yolo_export/mygame"))

    :param root:    Directory that will hold images/, labels/, and classes.json.
    :param classes: Ordered list of class names.  Index = class_id in labels.
    """

    def __init__(self, root: Path, classes: List[str]):
        self.root = Path(root)
        self.classes = classes
        self._images_dir = self.root / "images"
        self._labels_dir = self.root / "labels"
        self._classes_file = self.root / "classes.json"
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._labels_dir.mkdir(parents=True, exist_ok=True)
        # Write / overwrite classes.json so it stays in sync with self.classes
        self._classes_file.write_text(json.dumps(self.classes, indent=2))

    @classmethod
    def load_existing(cls, root: Path) -> Dataset:
        """
        Load a Dataset from an existing directory.
        Reads classes from classes.json — no need to pass them again.

        :raises FileNotFoundError: if root/classes.json does not exist.
        """
        root = Path(root)
        classes_file = root / "classes.json"
        if not classes_file.exists():
            raise FileNotFoundError(f"No classes.json found in {root}")
        classes = json.loads(classes_file.read_text())
        return cls(root=root, classes=classes)

    # ------------------------------------------------------------------
    # Sample management
    # ------------------------------------------------------------------

    def _next_id(self) -> str:
        existing = list(self._images_dir.glob("*.png"))
        return f"{len(existing) + 1:06d}"

    def add_sample(self, image: cv2.Mat, boxes: List[BoundingBox]) -> LabeledImage:
        """
        Save a screenshot and its bounding box annotations to disk.

        :param image: BGR cv2.Mat from window.screenshot().
        :param boxes: Bounding boxes annotated on this frame.
        :returns:     The LabeledImage record pointing to the saved files.
        """
        sample_id = self._next_id()
        img_path = self._images_dir / f"{sample_id}.png"
        lbl_path = self._labels_dir / f"{sample_id}.txt"

        cv2.imwrite(str(img_path), image)
        lbl_path.write_text("\n".join(b.to_yolo_line() for b in boxes))

        return LabeledImage(image_path=img_path, label_path=lbl_path, boxes=list(boxes))

    def load(self) -> List[LabeledImage]:
        """
        Read all saved samples from disk into LabeledImage records.

        Boxes are parsed from the YOLO label files.  Missing label files
        result in an empty box list for that sample.
        """
        class_map = {i: name for i, name in enumerate(self.classes)}
        samples: List[LabeledImage] = []

        for img_path in sorted(self._images_dir.glob("*.png")):
            lbl_path = self._labels_dir / f"{img_path.stem}.txt"
            boxes: List[BoundingBox] = []
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        boxes.append(BoundingBox(
                            class_id=cid,
                            class_name=class_map.get(cid, str(cid)),
                            cx=float(parts[1]),
                            cy=float(parts[2]),
                            w=float(parts[3]),
                            h=float(parts[4]),
                        ))
            samples.append(LabeledImage(img_path, lbl_path, boxes))

        return samples

    # ------------------------------------------------------------------
    # Splitting and export
    # ------------------------------------------------------------------

    def split(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        seed: int = 42,
    ) -> tuple[List[LabeledImage], List[LabeledImage], List[LabeledImage]]:
        """
        Shuffle and split samples into (train, val, test) lists.

        Does **not** copy any files — returns lists of existing LabeledImage
        records.  Pass these to ``export_yolo_structure`` to copy them.

        :returns: (train, val, test) lists of LabeledImage.
        """
        samples = self.load()
        rng = random.Random(seed)
        rng.shuffle(samples)

        n = len(samples)
        n_test = max(1, int(n * test_ratio)) if n > 2 else 0
        n_val  = max(1, int(n * val_ratio))  if n > 1 else 0

        test  = samples[:n_test]
        val   = samples[n_test:n_test + n_val]
        train = samples[n_test + n_val:]
        return train, val, test

    def export_yolo_structure(self, out_dir: Path) -> Path:
        """
        Export a train/val/test split in the Ultralytics-compatible layout::

            out_dir/
              images/
                train/  val/  test/
              labels/
                train/  val/  test/
              data.yaml

        Existing files in out_dir are overwritten.

        :param out_dir: Target directory (created if it does not exist).
        :returns:       Path to the generated data.yaml file.
        """
        out_dir = Path(out_dir)
        train, val, test = self.split()

        for split_name, samples in (("train", train), ("val", val), ("test", test)):
            img_out = out_dir / "images" / split_name
            lbl_out = out_dir / "labels" / split_name
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for s in samples:
                shutil.copy2(s.image_path, img_out / s.image_path.name)
                shutil.copy2(s.label_path, lbl_out / s.label_path.name)

        yaml_path = out_dir / "data.yaml"
        yaml_path.write_text(
            f"path: {out_dir.resolve()}\n"
            f"train: images/train\n"
            f"val:   images/val\n"
            f"test:  images/test\n"
            f"nc: {len(self.classes)}\n"
            f"names: {self.classes}\n"
        )
        return yaml_path

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(list(self._images_dir.glob("*.png")))

    def __repr__(self) -> str:
        return (
            f"Dataset(root='{self.root}', classes={self.classes}, "
            f"samples={len(self)})"
        )
