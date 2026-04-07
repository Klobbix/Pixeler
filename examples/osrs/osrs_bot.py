"""
OSRS — Example GameBot

Demonstrates the full Pixeler stack end-to-end:

    1. Window targeting    — Win32Window finds the game by title substring.
    2. YOLO detection      — YOLOClassifier runs a trained ONNX model each frame.
    3. Color detection     — ColorFilter catches a critically-low HP bar quickly.
    4. OCR                 — read_number() reads the exact HP value from the HUD.
    5. Event bus           — ScreenAnalyzer emits events; modules react.
    6. Game modules        — CombatModule and LootModule handle all bot logic.

Before running
--------------
1. Capture a dataset::

       pixeler-capture --game "RuneLite" --dataset datasets/osrs \\
           --classes enemy,loot_item

2. Train a model (requires: uv sync --extra train)::

       pixeler-train --dataset datasets/osrs \\
           --out models/osrs.onnx --epochs 100

3. Adjust the constants below to match your server / class keybinds.

Run from the project root::

    uv run python examples/osrs/osrs_bot.py
"""

from pathlib import Path

from widget_module import WidgetModule
from pixeler.bot.game_bot import GameBot
from pixeler.math.rectangle import Rectangle
from pixeler.vision.classifier import YOLOClassifier
from pixeler.vision.color import ColorFilter
from pixeler.window.win32_window import Win32Window

# ---------------------------------------------------------------------------
# Configuration — adjust these for your setup
# ---------------------------------------------------------------------------

WINDOW_TITLE  = "RuneLite"          # substring of the game window title

MODEL_PATH    = Path("models/osrs.onnx")
CLASSES       = ["buttons"]

# Key bound to your primary attack / main skill
ATTACK_KEY    = "ctrl"
# Key bound to health potions in-game
POTION_KEY    = "1"
# Key bound to item pick-up
LOOT_KEY      = "z"

# Numeric HP threshold below which a potion is used (requires OCR)
HP_THRESHOLD  = 30.0

# Screen region where the HP number is displayed (x, y, width, height px)
# Adjust to match your resolution and HUD layout
HP_OCR_REGION = Rectangle(52, 12, 90, 16)

# HSV color of a critically low HP bar — tune with vision/color.py helpers
# Default: dark red (nearly empty bar)
HP_LOW_COLOR = ColorFilter.from_rgb(160, 20, 20, hue_tol=12, sat_tol=60, val_tol=60)

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------

def main() -> None:
    if not MODEL_PATH.exists():
        print(
            f"[Setup] Model not found at {MODEL_PATH}.\n"
            f"        Run: pixeler-train --dataset datasets/osrs "
            f"--out {MODEL_PATH}"
        )
        return

    print(f"[Setup] Connecting to '{WINDOW_TITLE}' ...")
    window = Win32Window(WINDOW_TITLE)

    bot = GameBot(window=window)

    # --- Detectors ---
    # YOLO: runs every frame (throttle_s=0) — adjust if CPU-bound
    bot.analyzer.add_yolo(
        name="main_detector",
        classifier=YOLOClassifier(
            model_path=MODEL_PATH,
            class_names=CLASSES,
            conf_threshold=0.1,
        ),
        throttle_s=0.05,   # at most ~20 YOLO runs/s
    )

    # --- Modules ---
    bot.add_module(WidgetModule())

    print("[Setup] Bot configured. Starting — press Ctrl+C to stop.\n")
    bot.start()
    # Keep the main thread alive; bot runs on a daemon thread
    try:
        import time
        while bot.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Main] Stopping bot...")
        bot.stop()


if __name__ == "__main__":
    main()
