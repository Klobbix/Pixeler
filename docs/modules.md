# Writing Your First Game Module

This guide walks through the full workflow: capturing training data, training a model, and writing a game module that reacts to what's on the screen.

---

## Overview

```
pixeler-capture  →  pixeler-train  →  GameBot + GameModule
  (label data)      (ONNX model)       (run automation)
```

A **GameModule** is a self-contained plugin that subscribes to events emitted by the bot's vision pipeline. Each module handles one responsibility (combat, looting, banking, etc.). Multiple modules run inside a single `GameBot` at the same time.

---

## Step 1 — Capture a Dataset

Point the capture tool at your running game and draw bounding boxes around the objects you want to detect.

```bash
pixeler-capture \
  --game "MyGame" \
  --dataset datasets/mygame \
  --classes enemy,loot_item,health_bar
```

An OpenCV annotation window opens on a frozen screenshot of your game window.

### Annotation controls

| Key | Action |
|---|---|
| Click + drag | Draw a bounding box |
| `0`–`9` | Assign a class to the most recently drawn box |
| `s` | Save this frame with all boxes |
| `Space` | Grab a new frame (auto-saves current if boxes exist) |
| `c` | Clear all boxes on the current frame |
| `q` | Quit the session |

### Tips for good data

- Capture at least **50–100 frames per class** for a usable model. More variety = better generalization.
- Cover different positions, lighting conditions, and map areas.
- Press `Space` often — variety matters more than volume per frame.
- Zoom the annotation window (`cv2.WINDOW_NORMAL`) to place boxes precisely.

---

## Step 2 — Train a Model

Install the training extra (one-time):

```bash
uv sync --extra train
```

Train and export to ONNX:

```bash
pixeler-train \
  --dataset datasets/mygame \
  --out models/mygame.onnx \
  --epochs 100 \
  --device cpu
```

Use `--device cuda:0` if you have an NVIDIA GPU — training is 10–50× faster.

Training artifacts (loss curves, confusion matrix, sample predictions) are written to `training_runs/mygame/yolo/train/`. Check them to judge model quality before deploying.

### Model sizes

| Flag | Model | Speed | Accuracy |
|---|---|---|---|
| `--model n` | yolov8**n**ano | Fastest — good for CPU | Good for simple sprites |
| `--model s` | yolov8**s**mall | Fast | Better generalization |
| `--model m` | yolov8**m**edium | Slower | Higher accuracy |

Start with nano (`n`). Upgrade only if accuracy is insufficient.

---

## Step 3 — Create a Bot File

```python
# my_game_bot.py
from pathlib import Path
from src.pixeler.bot.game_bot import GameBot
from src.pixeler.window.win32_window import Win32Window
from src.pixeler.vision.classifier import YOLOClassifier
from src.pixeler.vision.color import ColorFilter
from src.pixeler.math.rectangle import Rectangle

from my_game.combat_module import CombatModule
from my_game.loot_module import LootModule

bot = GameBot(window=Win32Window("MyGame"))

# YOLO detector — emits "detection.enemy", "detection.loot_item", etc.
bot.analyzer.add_yolo(
    "detector",
    YOLOClassifier("models/mygame.onnx", ["enemy", "loot_item"], conf_threshold=0.55),
    throttle_s=0.05,
)

# Color filter — emits "color.health_low" when HP bar is critically red
bot.analyzer.add_color(
    "health_low",
    ColorFilter.from_rgb(200, 20, 20, hue_tol=12),
    throttle_s=0.25,
)

# OCR — emits "ocr.gold" with the numeric gold count once per second
bot.analyzer.add_ocr(
    "gold",
    Rectangle(10, 50, 120, 18),
    throttle_s=1.0,
    numeric=True,
)

bot.add_module(CombatModule())
bot.add_module(LootModule())
bot.start()

import time
try:
    while bot.is_running():
        time.sleep(0.5)
except KeyboardInterrupt:
    bot.stop()
```

---

## Step 4 — Write a Module

```python
# my_game/combat_module.py
import time
from src.pixeler.modules.base_module import GameModule
from src.pixeler.events.payloads import DetectionPayload, ColorRegionPayload, OCRPayload

class CombatModule(GameModule):
    name = "combat"
    description = "Attack enemies; use health potions when HP is low."

    def __init__(self):
        super().__init__()
        self._last_attack = 0.0
        self._last_potion = 0.0

    def on_register(self, bus, bot):
        self._bot = bot
        self.on("detection.enemy",  self._attack)
        self.on("color.health_low", self._eat_potion)
        self.on("ocr.gold",         self._log_gold)

    def _attack(self, event):
        # Cooldown: don't spam attacks faster than once per 0.6 s
        if time.monotonic() - self._last_attack < 0.6:
            return

        payload: DetectionPayload = event.data
        self.log(f"Attacking {payload.class_name} at {payload.center}")

        self._bot.window.focus()
        from src.pixeler.input.keyboard import press
        press("ctrl")   # your attack key
        self._last_attack = time.monotonic()

    def _eat_potion(self, event):
        if time.monotonic() - self._last_potion < 3.0:
            return  # don't double-pot

        payload: ColorRegionPayload = event.data
        self.log(f"HP critical (area={payload.area}px²) — using potion")

        self._bot.window.focus()
        from src.pixeler.input.keyboard import press
        press("1")
        self._last_potion = time.monotonic()

    def _log_gold(self, event):
        payload: OCRPayload = event.data
        if payload.number is not None:
            self.log(f"Gold: {payload.number:.0f}")

    def on_deregister(self):
        self.log("Combat module stopped.")
```

---

## Event reference

### Subscribing

```python
# Exact class
self.on("detection.enemy", handler)

# All detections from any class
self.on("detection.*", handler)

# Everything on the bus (for debugging)
self.on("*", handler)
```

### Built-in event names

| Pattern | Fires when | Payload type |
|---|---|---|
| `detection.<class>` | YOLO/ORB detects a class | `DetectionPayload` |
| `color.<name>` | Color filter finds regions | `ColorRegionPayload` |
| `template.<name>` | Template match succeeds | `TemplateMatchPayload` |
| `ocr.<name>` | OCR reads text from a region | `OCRPayload` |
| `bot.started` | Bot starts | `BotLifecyclePayload` |
| `bot.stopped` | Bot stops | `BotLifecyclePayload` |
| `bot.paused` | Bot pauses | `BotLifecyclePayload` |
| `bot.resumed` | Bot resumes | `BotLifecyclePayload` |

### Payload shortcuts

Every payload type exposes `.center` and `.rect` for the most common case:

```python
def _on_enemy(self, event):
    payload: DetectionPayload = event.data
    payload.center      # Point(x, y) — best detection's center
    payload.rect        # Rectangle — best detection's bounding box
    payload.confidence  # float [0, 1]
    payload.best        # Detection — highest-confidence Detection object
    payload.detections  # list[Detection] — all detections this frame
```

```python
def _on_hp_low(self, event):
    payload: ColorRegionPayload = event.data
    payload.center   # center of the largest matching region
    payload.area     # area in pixels²
    payload.largest  # ColorRegion — biggest matching region
    payload.regions  # list[ColorRegion]
```

```python
def _on_hp_ocr(self, event):
    payload: OCRPayload = event.data
    payload.text    # full extracted string
    payload.number  # float if numeric=True was set, else None
    payload.words   # list[Word] with positions and confidence
```

---

## Detector configuration

### Throttling

Throttle expensive detectors so they don't run every frame:

```python
bot.analyzer.add_yolo(..., throttle_s=0.05)   # ~20 runs/s
bot.analyzer.add_ocr(...,  throttle_s=1.0)    # once per second
bot.analyzer.add_color(..., throttle_s=0.1)   # 10 checks/s
```

Color filters are cheap — they can run every frame (`throttle_s=0.0`).
YOLO and OCR are expensive — keep them throttled.

### Enable / disable at runtime

```python
bot.analyzer.disable("detector")   # stop running YOLO during a cutscene
bot.analyzer.enable("detector")    # resume
bot.analyzer.remove("detector")    # unregister permanently
```

---

## Inter-module communication

Modules can talk to each other by emitting custom events:

```python
# In module A — emit a custom event
from src.pixeler.events.event_bus import Event

self.emit(Event("game.boss_appeared", data={"name": "Zakum"}))

# In module B — subscribe to it
self.on("game.boss_appeared", self._on_boss)
```

---

## ORB sprite matching (no training needed)

For UI buttons, icons, or sprites with a fixed appearance, skip YOLO entirely:

```python
from src.pixeler.training.trainer import ORBTrainer

# One-time setup: save a reference crop
orb_lib = ORBTrainer(Path("sprites/mygame"))
orb_lib.add_reference("login_button", cv2.imread("login_crop.png"))

# At bot runtime
matcher = orb_lib.load_matcher("login_button")
bot.analyzer.add_orb("login_button", matcher)
# Emits "detection.login_button" when found
```

---

## Working example

See `examples/maplestory/` for a complete, runnable bot:

- `maplestory_bot.py` — bot setup with all detectors configured
- `combat_module.py` — attack cooldown, potion usage, HP tracking via OCR
- `loot_module.py` — loot all items with position-based debouncing
