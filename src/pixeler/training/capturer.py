"""
Interactive capture session for labeling game screenshots.

Annotation window controls
--------------------------
Click + drag    Draw a bounding box.
0–9             Assign class to the most recently drawn box.
s               Save the current frame with all drawn boxes.
c               Clear all boxes on the current frame.
Space           Capture a fresh frame (auto-saves first if boxes exist).
q               Quit the session.

Usage::

    from pathlib import Path
    from pixeler.training.dataset import Dataset
    from pixeler.training.capturer import CaptureSession
    from pixeler.window.win32_window import Win32Window

    ds = Dataset(Path("datasets/mygame"), classes=["enemy", "loot"])
    session = CaptureSession(window=Win32Window("MyGame"), dataset=ds)
    session.run(fps=0.5)   # new frame available every 2 s; annotate manually
"""

from __future__ import annotations

from typing import List, Optional

import cv2

from pixeler.training.dataset import BoundingBox, Dataset
from pixeler.window.abstract_window import AbstractWindow


# One distinct BGR color per class index (up to 10)
_CLASS_COLORS: list[tuple[int, int, int]] = [
    (255,   0,   0),   # 0  blue
    (  0, 255,   0),   # 1  green
    (  0,   0, 255),   # 2  red
    (255, 255,   0),   # 3  cyan
    (  0, 255, 255),   # 4  yellow
    (255,   0, 255),   # 5  magenta
    (128, 255,   0),   # 6  lime
    (  0, 128, 255),   # 7  orange-blue
    (255, 128,   0),   # 8  sky-blue
    (128,   0, 255),   # 9  purple
]

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM   = 0.45
_FONT_MED  = 0.55
_LINE_AA   = cv2.LINE_AA
_WINDOW    = "Pixeler Annotator  (drag=box  0-9=class  s=save  c=clear  space=next  q=quit)"


class CaptureSession:
    """
    Interactive bounding-box annotation tool that runs in an OpenCV window.

    The game window is **never** modified — the tool operates on a frozen
    screenshot copy.

    :param window:  Any AbstractWindow (Window or Win32Window) pointing at
                    the running game.
    :param dataset: The Dataset to save labeled frames into.
    """

    def __init__(self, window: AbstractWindow, dataset: Dataset):
        self._window  = window
        self._dataset = dataset

        # Current frame state
        self._frame:   Optional[cv2.Mat]      = None   # original frozen screenshot
        self._canvas:  Optional[cv2.Mat]      = None   # redraw surface (committed boxes)
        self._boxes:   List[BoundingBox]      = []     # confirmed annotations
        self._current_class: int              = 0      # active class index

        # Mouse drag state
        self._drag_start: Optional[tuple[int, int]] = None
        self._drag_end:   Optional[tuple[int, int]] = None
        self._dragging:   bool                      = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_frame(self) -> cv2.Mat:
        """Return a fresh screenshot of the game window."""
        return self._window.screenshot()

    def run(self, fps: float = 1.0) -> None:
        """
        Start the interactive annotation loop.

        A new screenshot is automatically grabbed at startup.  Press Space to
        grab a fresh one at any time (the current frame is auto-saved first if
        it has any boxes).  Press 'q' to exit.

        :param fps: *Unused during annotation* — only affects the auto-advance
                    interval when the user presses Space.  Kept as a param for
                    future continuous-capture mode.
        """
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_WINDOW, self._on_mouse)

        self._new_frame(self.capture_frame())
        print(f"[Capturer] Session started.  Dataset: {self._dataset}")

        while True:
            cv2.imshow(_WINDOW, self._render_display())
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("[Capturer] Session ended.")
                break

            elif key == ord('s'):
                self._save_current()

            elif key == ord('c'):
                self._boxes.clear()
                self._drag_start = None
                self._drag_end   = None
                self._rebuild_canvas()
                print("[Capturer] Boxes cleared.")

            elif key == ord(' '):
                if self._boxes:
                    self._save_current()
                self._new_frame(self.capture_frame())

            elif ord('0') <= key <= ord('9'):
                cid = key - ord('0')
                self._current_class = cid
                self._reassign_last_box(cid)
                self._rebuild_canvas()
                name = self._class_name(cid)
                print(f"[Capturer] Active class → {cid} ({name})")

        cv2.destroyWindow(_WINDOW)

    # ------------------------------------------------------------------
    # Frame management
    # ------------------------------------------------------------------

    def _new_frame(self, image: cv2.Mat) -> None:
        self._frame  = image.copy()
        self._boxes.clear()
        self._drag_start = None
        self._drag_end   = None
        self._rebuild_canvas()
        print("[Capturer] New frame captured.")

    def _rebuild_canvas(self) -> None:
        """
        Redraw the committed-box layer from the original frame.
        Called after any box is added, removed, or reclassified.
        """
        img = self._frame.copy()
        h, w = img.shape[:2]
        for box in self._boxes:
            bx, by, bw, bh = box.to_pixel_rect(w, h)
            color = _CLASS_COLORS[box.class_id % len(_CLASS_COLORS)]
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2)
            label = f"{box.class_id}: {box.class_name}"
            # Dark backing rectangle so text is readable on any background
            (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_MED, 1)
            ty = max(by - 4, th + 4)
            cv2.rectangle(img, (bx, ty - th - 2), (bx + tw + 2, ty + 2),
                          color, cv2.FILLED)
            cv2.putText(img, label, (bx + 1, ty), _FONT, _FONT_MED,
                        (0, 0, 0), 1, _LINE_AA)
        self._canvas = img

    def _render_display(self) -> cv2.Mat:
        """
        Return the final image to display: canvas + live drag rect + HUD.
        Does not mutate any stored state.
        """
        img = self._canvas.copy()
        h, w = img.shape[:2]

        # Live drag rectangle
        if self._dragging and self._drag_start and self._drag_end:
            color = _CLASS_COLORS[self._current_class % len(_CLASS_COLORS)]
            cv2.rectangle(img, self._drag_start, self._drag_end, color, 1)

        # Class list HUD (top-right)
        for i, name in enumerate(self._dataset.classes[:10]):
            color  = _CLASS_COLORS[i % len(_CLASS_COLORS)]
            marker = ">" if i == self._current_class else " "
            text   = f"{marker}[{i}] {name}"
            cv2.putText(img, text, (w - 160, 18 + i * 18),
                        _FONT, _FONT_SM, color, 1, _LINE_AA)

        # Saved count (bottom-left)
        cv2.putText(img, f"Saved: {len(self._dataset)}  Boxes: {len(self._boxes)}",
                    (6, h - 8), _FONT, _FONT_SM, (200, 200, 200), 1, _LINE_AA)

        return img

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y)
            self._drag_end   = (x, y)
            self._dragging   = True

        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self._drag_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self._dragging:
            self._dragging = False
            self._drag_end = (x, y)
            self._commit_drag()

    def _commit_drag(self) -> None:
        """Convert the finished drag into a BoundingBox and append it."""
        if self._drag_start is None or self._drag_end is None:
            return
        x1, y1 = self._drag_start
        x2, y2 = self._drag_end
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return  # accidental micro-click — ignore

        rx = min(x1, x2)
        ry = min(y1, y2)
        rw = abs(x2 - x1)
        rh = abs(y2 - y1)

        h, w = self._frame.shape[:2]
        cid   = self._current_class
        name  = self._class_name(cid)
        box   = BoundingBox.from_pixel_rect(cid, name, rx, ry, rw, rh, w, h)
        self._boxes.append(box)
        self._rebuild_canvas()
        print(f"[Capturer] Box: class={cid} ({name})  pixel=({rx},{ry},{rw},{rh})")

    # ------------------------------------------------------------------
    # Box helpers
    # ------------------------------------------------------------------

    def _reassign_last_box(self, class_id: int) -> None:
        """Change the class of the most recently drawn box."""
        if not self._boxes:
            return
        b = self._boxes[-1]
        self._boxes[-1] = BoundingBox(
            class_id=class_id,
            class_name=self._class_name(class_id),
            cx=b.cx, cy=b.cy, w=b.w, h=b.h,
        )

    def _class_name(self, class_id: int) -> str:
        classes = self._dataset.classes
        return classes[class_id] if class_id < len(classes) else str(class_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_current(self) -> None:
        if self._frame is None:
            print("[Capturer] No frame loaded — nothing to save.")
            return
        if not self._boxes:
            print("[Capturer] No boxes drawn — skipping save.")
            return
        labeled = self._dataset.add_sample(self._frame, self._boxes)
        print(f"[Capturer] Saved {labeled.image_path.name} "
              f"with {len(self._boxes)} box(es).")
        self._boxes.clear()
        self._rebuild_canvas()
