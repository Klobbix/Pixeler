# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pixeler is a Python automation framework (library) for creating game bots that can see, understand, and interact with Windows applications in a human-like way. It uses OpenCV for computer vision, pytesseract for OCR, mss for screen capture, and pyautogui for input simulation.

The full stack: capture labeled screenshots → train a YOLO model → run a `GameBot` that detects game elements and emits events → game-specific modules subscribe to those events and perform automation actions.

## Commands

This project uses `uv` for dependency management.

```bash
# Install runtime dependencies
uv sync

# Install with training support (Ultralytics/YOLOv8)
uv sync --extra train

# Capture labeled screenshots from a live game window
pixeler-capture --game "MyGame" --dataset datasets/mygame --classes enemy,loot,npc

# Train a YOLO model from a dataset
pixeler-train --dataset datasets/mygame --out models/mygame.onnx --epochs 100

# Run an example bot
uv run python examples/example_bot.py

# Build the package
uv build

# Run tests
uv run python -m pytest tests/
```

**Third-party requirement:** Tesseract must be installed separately and its path added to the system `PATH` before OCR functions will work.

## Architecture

Source is under `src/pixeler/`. Imports within the package use `from src.pixeler...` paths (not installed as editable — run from project root).

```
┌──────────────────────────────────────────────────────────┐
│                        GameBot                           │
│  ┌───────────────┐   ┌──────────────┐  ┌─────────────┐  │
│  │ScreenAnalyzer │──▶│  EventBus    │──▶│  GameModule │  │
│  │ (vision loop) │   │ (pub / sub)  │   │  (plugin)   │  │
│  └───────────────┘   └──────────────┘  └─────────────┘  │
└──────────────────────────────────────────────────────────┘
         ▲
  ┌──────┴──────┐
  │  Detectors  │  ← YOLOClassifier | ORBMatcher | ColorFilter | Template
  └─────────────┘
         ▲
  ┌──────┴──────┐
  │   Dataset   │  ← captured images + YOLO labels
  └─────────────┘
         ▲
  ┌──────┴──────┐
  │  Capturer   │  ← pixeler-capture CLI / CaptureSession
  └─────────────┘
```

---

### `bot/` — Bot lifecycle framework

- **`Bot`** (abstract base): Subclass and implement `step()` for one iteration of bot logic. The framework manages the loop, threading, and lifecycle — no `while` loop needed. Override `on_start()` / `on_stop()` for setup/teardown. Call `self.stop()` from anywhere to terminate cleanly.
- **`BotThread`**: Daemon thread that drives the `step()` loop. Uses `threading.Event` for cooperative shutdown. Catches exceptions in `step()`, stops the bot, and logs them.
- **`BotStatus`**: Enum — `RUNNING`, `PAUSED`, `STOPPED`.
- **`GameBot`** *(in progress)*: Extends `Bot` with an `EventBus`, `ScreenAnalyzer`, and `GameModule` registry. Users add detectors to `bot.analyzer` and modules to `bot.add_module()`. `step()` is sealed — all logic lives in modules.
- **`ScreenAnalyzer`** *(in progress)*: Runs configured detectors each frame and emits events onto the bus.

### `window/` — Window targeting and drawing

- **`AbstractWindow`**: Base interface for both window implementations.
- **`Window`**: Cross-platform window targeting using `pywinctl` + `mss` for screenshots. Find by title substring.
- **`Win32Window`**: Windows-specific targeting via `win32gui` HWND. Supports `create_overlay()`.
- **`Overlay`**: Win32 transparent layered window (`WS_EX_LAYERED | WS_EX_TRANSPARENT`) that sits topmost over a parent HWND. Created via `Win32Window.create_overlay()`. Supports GDI drawing (rects, circles, text) without modifying the game window.

### `vision/` — Computer vision

Four detection strategies:

- **`color.py`** — `Color` (BGR solid for drawing/GDI) and `ColorFilter` (HSV range for detection). Always use `ColorFilter` for game objects — HSV is illumination-independent. Pre-defined color constants (`RED`, `GREEN`, etc.).
- **`detection.py`** — `find_color_regions(image, ColorFilter)` → `list[ColorRegion]`. `find_template` / `find_all_templates` → `TemplateMatch`. Results have `.center`, `.rect`, `.confidence`.
- **`classifier.py`** — `ORBMatcher`: feature-based, finds a reference sprite under rotation/scale; no training needed. `YOLOClassifier`: loads a YOLO ONNX model via `cv2.dnn`, runs inference → `list[Detection]`.
- **`ocr.py`** — `read_text()` for strings, `read_number()` for numeric values, `read_words()` for words with bounding boxes and confidence. Always preprocess game screenshots first.
- **`utils.py`** — Preprocessing pipeline: `preprocess_for_ocr()` runs upscale → CLAHE → threshold → denoise. Individual steps also available.
- **`oir.py`** — Draw onto `cv2.Mat` in memory (for debug saves). Not for live screen drawing — use `Overlay` for that.

### `training/` — Dataset capture and model training

- **`dataset.py`** — `Dataset`: manages `images/` + `labels/` + `classes.json` on disk. `add_sample()` saves a screenshot + YOLO label. `load()` reads them back. `export_yolo_structure()` writes Ultralytics-compatible `data.yaml` with train/val/test splits. `load_existing()` reopens without re-passing class names. `BoundingBox` stores normalized YOLO coordinates with `from_pixel_rect()` / `to_pixel_rect()` / `to_yolo_line()`.
- **`capturer.py`** — `CaptureSession`: interactive OpenCV annotation UI. Click+drag draws boxes. `0–9` assigns class to the last box. `s` saves, `c` clears, `Space` grabs a new frame (auto-saves first), `q` quits. HUD shows class list + saved count.
- **`trainer.py`** — `YOLOTrainer`: wraps Ultralytics YOLOv8. `train()` exports the dataset, trains, and exports `best.pt` → `.onnx`. `export_to_onnx()` re-exports without retraining. Requires `uv sync --extra train`. `ORBTrainer`: manages a sprite library directory. `add_reference()` saves a crop. `load_matcher()` returns a ready `ORBMatcher`.
- **`cli.py`** — `pixeler-capture` and `pixeler-train` CLI entry points.

### `events/` — Pub/sub event system

- **`event_bus.py`** — `EventBus`: synchronous pub/sub. `subscribe(pattern, handler)` supports exact names, prefix wildcards (`"detection.*"`), and catch-all (`"*"`). `emit(Event)` dispatches to all matching handlers; exceptions are caught and logged without halting dispatch. `Event` is a frozen dataclass with `.name`, `.data`, `.source`, `.ts`.
- **`event_names.py`** — String constants and builder functions for all standard event names. Use `event_names.detection("enemy")` → `"detection.enemy"` rather than raw strings. Prefix constants (`DETECTION`, `COLOR`, `OCR`, etc.) for wildcard subscriptions.
- **`payloads.py`** — Typed dataclasses for every event kind: `DetectionPayload`, `ColorRegionPayload`, `TemplateMatchPayload`, `OCRPayload`, `BotLifecyclePayload`, `MouseActionPayload`, `KeyboardActionPayload`. Each has shortcut properties (`.center`, `.rect`, `.confidence`) so handlers don't need to drill into nested objects.

### `modules/` — Game module system *(in progress)*

- **`base_module.py`** — `GameModule` abstract base. Implement `on_register(bus, bot)` and call `self.on(event_name, handler)` to wire up automation logic. `on_deregister()` for cleanup.

### `input/` — Input simulation

- **`mouse.py`** — `move_to()` with WindMouse algorithm (realistic speed-varying curved paths). `click()`, `right_click()`, `double_click()`, `scroll()`. `move_and_click()` / `move_and_right_click()` compound actions. Click hold durations sampled from truncated-normal.
- **`keyboard.py`** — `write(text, wpm=65)` types at human speed with per-character timing variation. Optional `mistakes=True` for realistic typos with self-correction. `press()`, `key_down()`, `key_up()`, `hotkey()`.

### `math/` — Geometry and randomness

- **`Point`**: Immutable `NamedTuple(x, y)`. Arithmetic, `distance_to()`, `angle_to()`, `lerp()`, `normalize()`, `dot()`.
- **`Rectangle`**: `(x, y, w, h)`. Geometry, `random_point()` (truncated-normal biased toward center), `screenshot()`.
- **`Circle`**: `random_point()` (uniform and normal variants for human-like aiming).
- **`Polygon`**: Ray-casting `contains_point()`, `random_point()` (rejection sampling).
- **`bezier.py`** — `natural_path(start, end, steps=40)` produces human-like curved mouse paths. `random_cubic_control_points()` generates natural-looking offsets.
- **`random.py`** — Human timing distributions: `reaction_delay()` (ex-Gaussian), `idle_delay()` (chi-squared), `gaussian_jitter()`. Daily-stable seeds via `random_seeds()`.

---

## Key Design Notes

- **Color format:** `Color` takes RGB input but stores BGR internally. When drawing with Win32 GDI, `color.lower` index order is `[B, G, R]`.
- **Two window strategies:** `Window` (pywinctl, cross-platform) vs `Win32Window` (win32api, Windows-only with GDI). Use `Win32Window` when you need overlays or HDC access.
- **HSV for color detection:** `ColorFilter` uses HSV — hue is independent of brightness, so the same object stays the same hue under different lighting conditions.
- **Ultralytics is optional:** `training/trainer.py` guards the import with a clear error. The runtime (`YOLOClassifier`, `ORBMatcher`) uses only `cv2.dnn` and has no Ultralytics dependency.
- **Synchronous event bus:** Handlers run on the bot thread. No asyncio needed — keeps timing deterministic.
- **`GameBot.step()` is sealed:** Users write `GameModule` subclasses, not `step()` overrides. All automation logic lives in module event handlers.
- **Examples:** The `examples/` folder has working demos for color tracking, OCR, template matching, Win32 overlay drawing, and mouse tracking.
