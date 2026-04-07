import random
import time

import pyautogui as pag

from pixeler.math.random import truncated_normal_sample

# Keys whose neighbours are likely accidental press targets during fast typing.
# Used by write() when mistakes=True.
_NEIGHBOURS: dict[str, list[str]] = {
    'a': ['s', 'q', 'z'], 'b': ['v', 'n', 'g', 'h'], 'c': ['x', 'v', 'd', 'f'],
    'd': ['s', 'f', 'e', 'r', 'c', 'x'], 'e': ['w', 'r', 'd', 's'],
    'f': ['d', 'g', 'r', 't', 'v', 'c'], 'g': ['f', 'h', 't', 'y', 'b', 'v'],
    'h': ['g', 'j', 'y', 'u', 'n', 'b'], 'i': ['u', 'o', 'k', 'j'],
    'j': ['h', 'k', 'u', 'i', 'm', 'n'], 'k': ['j', 'l', 'i', 'o', 'm'],
    'l': ['k', 'o', 'p'], 'm': ['n', 'j', 'k'], 'n': ['b', 'm', 'h', 'j'],
    'o': ['i', 'p', 'k', 'l'], 'p': ['o', 'l'], 'q': ['w', 'a'],
    'r': ['e', 't', 'f', 'd'], 's': ['a', 'd', 'w', 'e', 'z', 'x'],
    't': ['r', 'y', 'g', 'f'], 'u': ['y', 'i', 'h', 'j'],
    'v': ['c', 'b', 'f', 'g'], 'w': ['q', 'e', 'a', 's'],
    'x': ['z', 'c', 's', 'd'], 'y': ['t', 'u', 'g', 'h'],
    'z': ['a', 's', 'x'],
}


def _wpm_to_base_interval(wpm: float) -> float:
    """Convert typing speed in WPM to average seconds per character.
    Assumes the standard 5 characters per word definition."""
    return 60.0 / (wpm * 5.0)


# ---------------------------------------------------------------------------
# Public keyboard API
# ---------------------------------------------------------------------------

def press(key: str, hold_duration: float = None):
    """
    Presses a key with a realistic randomised hold time.

    :param key: pyautogui key name (e.g. 'enter', 'space', 'a').
    :param hold_duration: Explicit hold duration in seconds. Randomised if None.
    """
    if hold_duration is None:
        hold_duration = truncated_normal_sample(0.04, 0.12)
    pag.keyDown(key)
    time.sleep(hold_duration)
    pag.keyUp(key)


def write(text: str, wpm: float = 65.0, mistakes: bool = False):
    """
    Types text at a human-like speed with natural timing variance.

    Each character gets an independently sampled hold duration and inter-key
    gap drawn from a distribution centred on the given WPM.  Spaces and
    punctuation get a slightly longer post-key pause, mirroring how humans
    pause briefly at word boundaries.

    :param text: The string to type.
    :param wpm: Target typing speed in words per minute (default 65 WPM).
    :param mistakes: If True, occasionally mistype a character and self-correct.
    """
    base = _wpm_to_base_interval(wpm)

    i = 0
    while i < len(text):
        char = text[i]

        # Rare typo: press a neighbour key, realise, backspace, then retype
        if mistakes and char.lower() in _NEIGHBOURS and random.random() < 0.025:
            wrong = random.choice(_NEIGHBOURS[char.lower()])
            hold = truncated_normal_sample(0.03, 0.09)
            pag.keyDown(wrong)
            time.sleep(hold)
            pag.keyUp(wrong)
            time.sleep(truncated_normal_sample(0.08, 0.25))  # brief hesitation
            press('backspace')
            time.sleep(truncated_normal_sample(0.05, 0.15))

        # Type the actual character
        hold = truncated_normal_sample(0.03, 0.10)
        pag.keyDown(char)
        time.sleep(hold)
        pag.keyUp(char)

        # Inter-key gap: word boundaries get a slightly longer pause
        if char in (' ', '.', ',', '!', '?', ';', ':'):
            gap = truncated_normal_sample(base * 0.8, base * 2.2)
        else:
            gap = truncated_normal_sample(base * 0.5, base * 1.6)

        remaining = max(0.0, gap - hold)
        time.sleep(remaining)

        i += 1


def key_down(key: str):
    """Holds a key down."""
    pag.keyDown(key)


def key_up(key: str):
    """Releases a held key."""
    pag.keyUp(key)


def hotkey(*keys: str):
    """Presses a hotkey combination (e.g. hotkey('ctrl', 'c'))."""
    pag.hotkey(*keys)
