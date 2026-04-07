"""
ML-based vision: feature matching and YOLO object detection.

Two approaches:
- **ORBMatcher** — finds a reference sprite/icon in a screenshot using ORB
  keypoints. Tolerates small rotations and scale changes. Zero training needed;
  just supply a reference image.
- **YOLOClassifier** — runs a YOLO ONNX model (e.g. exported from YOLOv8) to
  detect and classify multiple object types in one pass. Requires a trained
  model file, but detects arbitrary game objects once trained.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from pixeler.math.point import Point
from pixeler.math.rectangle import Rectangle


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A detected object returned by either ORBMatcher or YOLOClassifier."""
    class_id: int
    class_name: str
    confidence: float
    box: Rectangle         # bounding box in image-local coordinates

    @property
    def center(self) -> Point:
        return self.box.get_center()

    def __repr__(self) -> str:
        return (f"Detection({self.class_name!r}, conf={self.confidence:.2f}, "
                f"box={self.box})")


# ---------------------------------------------------------------------------
# ORB feature matcher
# ---------------------------------------------------------------------------

class ORBMatcher:
    """
    Finds a reference image (sprite, icon, UI element) inside a screenshot
    using ORB keypoint matching and homography estimation.

    Unlike pixel-level template matching, ORB tolerates moderate rotation and
    scale changes, making it suitable for game objects that may vary slightly
    in size or angle.

    Usage::

        matcher = ORBMatcher()
        matcher.load_reference(cv2.imread("sword_icon.png"))

        match = matcher.find(screenshot)
        if match:
            print(match.center)   # Point where the icon is on screen
    """

    def __init__(self,
                 min_matches: int = 12,
                 max_features: int = 500):
        """
        :param min_matches:  Minimum number of good keypoint matches required
                             to report a detection. Lower = more false positives.
        :param max_features: Maximum ORB keypoints to detect per image.
        """
        self.min_matches = min_matches
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._ref_kp = None
        self._ref_desc = None
        self._ref_shape: tuple[int, int] = (0, 0)  # (h, w)

    def load_reference(self, image: cv2.Mat) -> None:
        """
        Compute and cache keypoints for the reference object image.

        Call once before running find(). Call again to switch reference.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        self._ref_shape = gray.shape[:2]
        self._ref_kp, self._ref_desc = self._orb.detectAndCompute(gray, None)
        if self._ref_desc is None or len(self._ref_kp) == 0:
            raise ValueError("No keypoints found in reference image. "
                             "Ensure the image has enough texture/detail.")

    def load_reference_from_file(self, path: str | Path) -> None:
        """Load and cache the reference image from a file path."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Reference image not found: {path}")
        self.load_reference(img)

    def find(self, scene: cv2.Mat) -> Detection | None:
        """
        Search for the reference object in *scene*.

        :param scene: BGR screenshot to search within.
        :returns: Detection with bounding box, or None if not found.
        """
        if self._ref_desc is None:
            raise RuntimeError("No reference loaded. Call load_reference() first.")

        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY) if len(scene.shape) == 3 else scene
        kp, desc = self._orb.detectAndCompute(gray, None)

        if desc is None or len(kp) < self.min_matches:
            return None

        matches = self._matcher.match(self._ref_desc, desc)
        matches = sorted(matches, key=lambda m: m.distance)
        good = [m for m in matches if m.distance < 64]

        if len(good) < self.min_matches:
            return None

        # Estimate homography to get the object's location in the scene
        src_pts = np.float32([self._ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.min_matches:
            return None

        # Project reference corners into scene to get the bounding box
        ref_h, ref_w = self._ref_shape
        corners = np.float32([[0, 0], [ref_w, 0],
                               [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)
        scene_corners = cv2.perspectiveTransform(corners, H)
        pts = scene_corners.reshape(-1, 2)

        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        x_max = int(np.max(pts[:, 0]))
        y_max = int(np.max(pts[:, 1]))

        confidence = inliers / max(len(self._ref_kp), 1)

        return Detection(
            class_id=0,
            class_name="object",
            confidence=min(1.0, confidence),
            box=Rectangle(x_min, y_min, x_max - x_min, y_max - y_min),
        )


# ---------------------------------------------------------------------------
# YOLO classifier (cv2.dnn, ONNX)
# ---------------------------------------------------------------------------

class YOLOClassifier:
    """
    Runs a YOLO ONNX model for real-time multi-class object detection.

    Designed for YOLOv8 models exported to ONNX (``model.export(format='onnx')``),
    but compatible with any YOLO variant that produces output in the shape
    ``(1, num_classes + 4, num_proposals)``.

    Typical workflow:
    1. Annotate game screenshots with LabelImg or Roboflow.
    2. Train YOLOv8 (``yolo train data=data.yaml model=yolov8n.pt``).
    3. Export to ONNX (``yolo export model=best.pt format=onnx``).
    4. Load here and call detect() every bot step.

    Usage::

        yolo = YOLOClassifier(
            model_path="models/game_objects.onnx",
            class_names=["enemy", "health_orb", "chest"],
        )
        detections = yolo.detect(screenshot)
        enemies = [d for d in detections if d.class_name == "enemy"]
    """

    def __init__(self,
                 model_path: str | Path,
                 class_names: List[str],
                 conf_threshold: float = 0.50,
                 nms_threshold: float = 0.45,
                 input_size: tuple[int, int] = (640, 640)):
        """
        :param model_path:      Path to the ONNX model file.
        :param class_names:     Ordered list of class name strings matching
                                the model's training labels.
        :param conf_threshold:  Minimum confidence to report a detection.
        :param nms_threshold:   IoU threshold for non-maximum suppression.
        :param input_size:      (width, height) the model was trained on.
        """
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_w, self.input_h = input_size

        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        # Prefer GPU if available; fall back to CPU silently
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def use_cuda(self) -> None:
        """Switch inference to CUDA if an NVIDIA GPU is available."""
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(self, image: cv2.Mat) -> List[Detection]:
        """
        Run inference on *image* and return all detections above the
        confidence threshold, with non-maximum suppression applied.

        :param image: BGR screenshot (any size — resized internally).
        :returns: List of Detection objects.
        """
        orig_h, orig_w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 255.0,
            size=(self.input_w, self.input_h),
            swapRB=True, crop=False,
        )
        self._net.setInput(blob)
        raw = self._net.forward()  # shape: (1, num_classes+4, num_proposals)

        # Transpose to (num_proposals, num_classes+4)
        output = raw[0].T

        x_scale = orig_w / self.input_w
        y_scale = orig_h / self.input_h

        boxes: List[list] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        num_classes = len(self.class_names)

        for row in output:
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            scores = row[4: 4 + num_classes]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < self.conf_threshold:
                continue

            x = int((cx - bw / 2) * x_scale)
            y = int((cy - bh / 2) * y_scale)
            w = int(bw * x_scale)
            h = int(bh * y_scale)

            boxes.append([x, y, w, h])
            confidences.append(confidence)
            class_ids.append(class_id)

        # Non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )
        if len(indices) == 0:
            return []

        detections: List[Detection] = []
        for i in (indices.flatten() if hasattr(indices, 'flatten') else indices):
            x, y, w, h = boxes[i]
            cid = class_ids[i]
            detections.append(Detection(
                class_id=cid,
                class_name=self.class_names[cid] if cid < len(self.class_names) else str(cid),
                confidence=confidences[i],
                box=Rectangle(x, y, w, h),
            ))

        return detections
