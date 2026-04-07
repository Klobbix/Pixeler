# Pixeler
[![Publish Python Package](https://github.com/Klobbix/Pixeler/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Klobbix/Pixeler/actions/workflows/python-publish.yml)

> **Work in progress** — core systems are stable; the `GameBot` / module layer is under active development.

A Python framework for building game automation bots that can **see, understand, and interact** with Windows applications in a human-like way.

---

## How it works

```
Capture screenshots  →  Train a model  →  Run a GameBot  →  Write game modules
      (CLI)               (CLI/API)        (event loop)        (plugin API)
```

1. **Capture** — point the `pixeler-capture` tool at your game window and draw bounding boxes around the objects you care about (enemies, health bars, loot, UI elements). Frames are saved as a labeled dataset.
2. **Train** — run `pixeler-train` to fine-tune a YOLOv8 nano model on your dataset and export it as an ONNX file.
3. **Detect** — load the ONNX into a `GameBot` alongside color filters, templates, and OCR regions. Every frame, the `ScreenAnalyzer` runs all detectors and fires events.
4. **React** — write a `GameModule` that subscribes to events (`"detection.enemy"`, `"color.health_low"`, `"ocr.health_bar"`) and calls mouse/keyboard actions in response.

---

## Installation

```bash
pip install pixeler
```

Training support (YOLOv8 via Ultralytics) is an optional extra — skip it if you only need inference at runtime:

```bash
# Runtime only
pip install pixeler

# With training support
pip install "pixeler[train]"
```

**Additional requirement:** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) must be installed separately and its directory added to your system `PATH` before OCR functions will work.

---

## Quickstart

### Step 1 — Capture a dataset

```bash
pixeler-capture --game "MyGame" --dataset datasets/mygame --classes enemy,loot,health_bar
```

An OpenCV annotation window opens on top of a frozen game screenshot.

| Key | Action |
|---|---|
| Click + drag | Draw a bounding box |
| `0`–`9` | Assign class to the last box |
| `s` | Save this frame |
| `Space` | Grab a new frame (auto-saves) |
| `c` | Clear boxes |
| `q` | Quit |

### Step 2 — Train

```bash
pixeler-train --dataset datasets/mygame --out models/mygame.onnx --epochs 100
```

Exports the dataset into a YOLO-compatible structure, trains `yolov8n`, and writes the final model to `models/mygame.onnx`. Training artifacts (charts, confusion matrices, checkpoints) land in `training_runs/mygame/`.

### Step 3 — Write a bot

```python
# my_game_bot.py
from pathlib import Path
from src.pixeler.bot.game_bot import GameBot          # coming in phase 4
from src.pixeler.window.win32_window import Win32Window
from src.pixeler.vision.classifier import YOLOClassifier
from src.pixeler.vision.color import ColorFilter
from src.pixeler.math.rectangle import Rectangle
from my_game.combat_module import CombatModule

CLASSES = ["enemy", "loot", "health_bar"]

bot = GameBot(window=Win32Window("MyGame"))
bot.analyzer.add_yolo("detector", YOLOClassifier("models/mygame.onnx", CLASSES))
bot.analyzer.add_color("health_low", ColorFilter.from_rgb(220, 30, 30, hue_tol=10))
bot.analyzer.add_ocr("gold", Rectangle(10, 50, 120, 18), throttle_s=1.0)
bot.add_module(CombatModule())
bot.start()
```

### Step 4 — Write a game module

```python
# my_game/combat_module.py
from src.pixeler.modules.base_module import GameModule   # coming in phase 5
from src.pixeler.events.payloads import DetectionPayload, ColorRegionPayload

class CombatModule(GameModule):
    name = "combat"

    def on_register(self, bus, bot):
        self._bot = bot
        self.on("detection.enemy", self._attack)
        self.on("color.health_low", self._eat_food)

    def _attack(self, event):
        payload: DetectionPayload = event.data
        self.log(f"Enemy at {payload.center}")
        from src.pixeler.input.mouse import move_and_right_click
        move_and_right_click(*payload.center)

    def _eat_food(self, event):
        from src.pixeler.input.keyboard import press
        press("1")
```

---

## Feature Overview

### Vision

| Module | What it does |
|---|---|
| `vision/color.py` | `ColorFilter` — HSV-range pixel detection (illumination-independent). `Color` — BGR solid colors for drawing. |
| `vision/detection.py` | `find_color_regions()`, `find_template()`, `find_all_templates()` — results include `.center`, `.rect`, `.confidence`. |
| `vision/classifier.py` | `YOLOClassifier` — ONNX multi-class detection via `cv2.dnn`. `ORBMatcher` — feature-based sprite matching, no training needed. |
| `vision/ocr.py` | `read_text()`, `read_number()`, `read_words()`. Always call with `preprocess=True` on game screenshots. |
| `vision/utils.py` | `preprocess_for_ocr()` — upscale → CLAHE → threshold → denoise. |

### Training

| Module | What it does |
|---|---|
| `training/dataset.py` | `Dataset` — manages labeled screenshot storage in YOLO format. `BoundingBox` — normalized coordinates with pixel ↔ YOLO conversion. |
| `training/capturer.py` | `CaptureSession` — interactive OpenCV annotation UI against a live game window. |
| `training/trainer.py` | `YOLOTrainer` — trains YOLOv8, exports to ONNX (requires `[train]` extra). `ORBTrainer` — sprite library for `ORBMatcher`. |

### Event System

| Module | What it does |
|---|---|
| `events/event_bus.py` | `EventBus` — synchronous pub/sub. Wildcard subscriptions (`"detection.*"`), fault-isolated handlers, catch-all (`"*"`). |
| `events/event_names.py` | String constants and builder functions: `event_names.detection("enemy")` → `"detection.enemy"`. |
| `events/payloads.py` | Typed payloads for every event: `DetectionPayload`, `ColorRegionPayload`, `OCRPayload`, etc. All have `.center`, `.rect` shortcuts. |

### Input

| Module | What it does |
|---|---|
| `input/mouse.py` | `move_to()` with WindMouse algorithm. `click()`, `right_click()`, `scroll()`, `move_and_click()`. |
| `input/keyboard.py` | `write(text, wpm=65)` with human timing. `press()`, `hotkey()`. Optional `mistakes=True` for realistic typos. |

### Window

| Module | What it does |
|---|---|
| `window/win32_window.py` | `Win32Window` — Windows HWND targeting. `create_overlay()` factory. |
| `window/overlay.py` | `Overlay` — transparent always-on-top GDI drawing layer. `draw_rect()`, `draw_circle()`, `draw_text()`, `draw_label()`. |
| `window/window.py` | `Window` — cross-platform targeting via `pywinctl`. |

### Math & Geometry

| Module | What it does |
|---|---|
| `math/point.py` | `Point(x, y)` — immutable. `distance_to()`, `lerp()`, `normalize()`, `dot()`. |
| `math/rectangle.py` | `Rectangle(x, y, w, h)`. `random_point()` biased toward center. `screenshot()`. |
| `math/bezier.py` | `natural_path(start, end)` — human-like curved mouse paths. |
| `math/circle.py` | `Circle` — `random_point_normal()` for human-like aiming. |
| `math/polygon.py` | `Polygon` — ray-casting containment, `random_point()` via rejection sampling. |
| `math/random.py` | `reaction_delay()`, `idle_delay()`, `gaussian_jitter()` — human timing distributions. |

---

## Development Setup

```bash
git clone https://github.com/klobbix/Pixeler
cd Pixeler

# Install runtime dependencies
uv sync

# Install with training support
uv sync --extra train

# Run examples
uv run python examples/example_bot.py
```

---

## License

MIT — see `LICENSE`.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection training
- [pytesseract](https://github.com/madmaze/pytesseract) — OCR
- [OpenCV](https://opencv.org/) — computer vision
- [PyAutoGUI](https://github.com/asweigart/pyautogui) — input simulation
- WindMouse algorithm — [Jack Tasia](https://ben.land/post/2021/04/03/windmouse-human-mouse-movement/)

## Contact

[klobbix@gmail.com](mailto:klobbix@gmail.com) · [GitHub Issues](https://github.com/klobbix/Pixeler/issues)
