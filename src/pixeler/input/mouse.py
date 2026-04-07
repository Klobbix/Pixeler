import math
import random
import time

import pyautogui as pag
import win32api

from pixeler.math.random import truncated_normal_sample


def position():
    """Returns the current mouse position."""
    return pag.position()


# ---------------------------------------------------------------------------
# Internal movement primitives
# ---------------------------------------------------------------------------

def _set_pos(x: int, y: int):
    """Low-level absolute cursor positioning via win32api."""
    win32api.SetCursorPos((int(x), int(y)))


def _wind_mouse(x_start: float, y_start: float, x_end: float, y_end: float,
                gravity: float = 9.0, wind: float = 3.0,
                min_wait: float = 2.0, max_wait: float = 8.0,
                max_step: float = 12.0, target_area: float = 8.0):
    """
    Moves the cursor along a wind/gravity physics path, producing the curved,
    speed-varied trajectories characteristic of human mouse movement.

    Based on the WindMouse algorithm (Jack Tasia, 2007).

    :param gravity: Pull strength toward the target.
    :param wind: Lateral turbulence magnitude.
    :param min_wait: Minimum sleep between steps (ms).
    :param max_wait: Maximum sleep between steps (ms) — used at low velocity.
    :param max_step: Maximum pixels moved per step.
    :param target_area: Pixel radius where wind dampens to avoid oscillation.
    """
    sqrt2 = math.sqrt(2)
    sqrt3 = math.sqrt(3)
    sqrt5 = math.sqrt(5)
    screen_w, screen_h = pag.size()

    wind_x = wind_y = 0.0
    vel_x = vel_y = 0.0
    x, y = float(x_start), float(y_start)

    while True:
        dist = math.hypot(x_end - x, y_end - y)
        if dist < 1:
            break

        wind_mag = min(wind, dist)

        if dist >= target_area:
            # Full turbulence
            wind_x = wind_x / sqrt3 + (random.random() * (wind_mag * 2 + 1) - wind_mag) / sqrt5
            wind_y = wind_y / sqrt3 + (random.random() * (wind_mag * 2 + 1) - wind_mag) / sqrt5
        else:
            # Dampen wind near target so the cursor settles cleanly
            wind_x /= sqrt2
            wind_y /= sqrt2

        vel_x += wind_x + gravity * (x_end - x) / dist
        vel_y += wind_y + gravity * (y_end - y) / dist

        vel_mag = math.hypot(vel_x, vel_y)
        if vel_mag > max_step:
            rand_speed = max_step / 2 + random.random() * max_step / 2
            vel_x = vel_x / vel_mag * rand_speed
            vel_y = vel_y / vel_mag * rand_speed

        x = max(0, min(x + vel_x, screen_w - 1))
        y = max(0, min(y + vel_y, screen_h - 1))
        _set_pos(int(round(x)), int(round(y)))

        # Higher velocity → less wait → faster apparent movement
        step = math.hypot(vel_x, vel_y)
        wait_ms = (max_wait - step) if step < max_wait else min_wait
        time.sleep(max(min_wait, wait_ms) / 1000.0)

    _set_pos(int(round(x_end)), int(round(y_end)))


# ---------------------------------------------------------------------------
# Public mouse API
# ---------------------------------------------------------------------------

def move_to(x: int, y: int, overshoot: bool = True):
    """
    Moves the mouse to (x, y) via a human-like WindMouse path.

    For longer distances, occasionally overshoots the target slightly then
    corrects, replicating a common human motor pattern.

    :param x: Target x coordinate.
    :param y: Target y coordinate.
    :param overshoot: Whether to allow overshoot-correction behaviour.
    """
    start_x, start_y = pag.position()
    dist = math.hypot(x - start_x, y - start_y)

    # Scale step size to distance so short moves stay precise
    max_step = max(5.0, min(25.0, dist / 18.0))
    gravity = truncated_normal_sample(7.0, 13.0)
    wind = truncated_normal_sample(1.0, 5.0)

    if overshoot and dist > 120 and random.random() < 0.35:
        # Overshoot past the target, pause, then correct with a tighter path
        ox = x + random.randint(-18, 18)
        oy = y + random.randint(-12, 12)
        _wind_mouse(start_x, start_y, ox, oy,
                    gravity=gravity, wind=wind, max_step=max_step)
        time.sleep(truncated_normal_sample(0.04, 0.12))
        _wind_mouse(ox, oy, x, y,
                    gravity=gravity * 1.5, wind=wind * 0.4,
                    max_step=max_step * 0.5, target_area=4.0)
    else:
        _wind_mouse(start_x, start_y, x, y,
                    gravity=gravity, wind=wind, max_step=max_step)


def _hold_duration() -> float:
    """Random button-hold time (seconds) sampled from a distribution that
    matches measured human click durations (~60–150 ms)."""
    return truncated_normal_sample(0.06, 0.16)


def click():
    """Left-clicks with a randomised hold duration."""
    pag.mouseDown()
    time.sleep(_hold_duration())
    pag.mouseUp()


def middle_click():
    """Middle-clicks with a randomised hold duration."""
    pag.mouseDown(button='middle')
    time.sleep(_hold_duration())
    pag.mouseUp(button='middle')


def right_click():
    """Right-clicks with a randomised hold duration."""
    pag.mouseDown(button='right')
    time.sleep(_hold_duration())
    pag.mouseUp(button='right')


def double_click():
    """Double-clicks with a randomised inter-click gap."""
    click()
    time.sleep(truncated_normal_sample(0.04, 0.14))
    click()


def move_and_click(x: int, y: int, overshoot: bool = True):
    """Moves to (x, y) then left-clicks after a brief natural pause."""
    move_to(x, y, overshoot=overshoot)
    time.sleep(truncated_normal_sample(0.05, 0.18))
    click()


def move_and_right_click(x: int, y: int, overshoot: bool = True):
    """Moves to (x, y) then right-clicks after a brief natural pause."""
    move_to(x, y, overshoot=overshoot)
    time.sleep(truncated_normal_sample(0.05, 0.18))
    right_click()


def scroll(clicks: int):
    """
    Scrolls the mouse wheel, inserting small random delays between ticks
    to avoid the perfectly uniform timing of programmatic scroll events.

    :param clicks: Number of ticks. Positive = up, negative = down.
    """
    direction = 1 if clicks > 0 else -1
    for _ in range(abs(clicks)):
        pag.scroll(direction)
        time.sleep(truncated_normal_sample(0.03, 0.10))
