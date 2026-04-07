"""
Randomness utilities tuned for human-like game bot behaviour.

Distributions are chosen to match empirical measurements of human timing and
spatial targeting rather than uniform or simple Gaussian models.
"""

import math
import random
import secrets
import time
from datetime import datetime
from typing import List, Union

import numpy as np


# ---------------------------------------------------------------------------
# Seeds / reproducibility
# ---------------------------------------------------------------------------

def random_seeds(mod: int = 0, start: int = 8, stop: int = 12) -> List[List[float]]:
    """
    Generate a set of daily-stable random seeds.

    The seeds are deterministic within a calendar day (useful for session-
    consistent click patterns) but differ between days, preventing trivial
    temporal fingerprinting.

    :param mod:   Offset added to today's date integer before seeding.
    :param start: Minimum number of seeds.
    :param stop:  Maximum number of seeds.
    """
    sg = secrets.SystemRandom()
    date = int(datetime.now().strftime("%Y%m%d"))
    random.seed(date + mod)
    count = sg.randrange(start, stop)
    return [[random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
            for _ in range(count)]


# ---------------------------------------------------------------------------
# Spatial sampling
# ---------------------------------------------------------------------------

def random_point_in(x_min: float, y_min: float,
                    width: float, height: float,
                    seeds: List[List[float]]) -> List[int]:
    """
    Return a random [x, y] inside a bounding box, biased toward the centre
    using the supplied seeds. Resembles how humans click UI elements (rarely
    the very edge, usually somewhere central).
    """
    sg = secrets.SystemRandom()

    if sg.randrange(0, 101) > 75:
        return _random_from(x_min, y_min, width, height)

    offset_pct = sg.uniform(0.15, 0.35)
    inner_x = round(width * offset_pct + x_min)
    inner_y = round(height * offset_pct + y_min)
    inner_w = round(width * (1.0 - offset_pct * 2))
    inner_h = round(height * (1.0 - offset_pct * 2))

    idx = sg.randrange(0, len(seeds))
    ratio_x = round(inner_w * seeds[idx][0])
    ratio_y = round(inner_h * seeds[idx][1])

    start_x = inner_x + ratio_x
    start_y = inner_y + ratio_y
    fix_w = start_x - x_min
    fix_h = start_y - y_min
    end_w = width - ratio_x
    end_h = height - ratio_y

    cell_w = fix_w if fix_w <= end_w else end_w
    cell_h = fix_h if fix_h <= end_h else end_h

    return _random_from(start_x, start_y, cell_w, cell_h, centered=False)


def gaussian_jitter(x: float, y: float, sigma: float = 2.0) -> tuple[float, float]:
    """
    Add independent Gaussian noise to a coordinate pair.

    Useful for adding sub-pixel variation to a computed target position so
    repeated clicks on the same logical target land at slightly different pixels.

    :param sigma: Standard deviation in pixels.
    """
    return (x + np.random.normal(0.0, sigma),
            y + np.random.normal(0.0, sigma))


def _random_from(x_min: float, y_min: float,
                 width: float, height: float,
                 centered: bool = True) -> List[int]:
    if centered:
        x_min = x_min + math.ceil(width / 2)
        y_min = y_min + math.ceil(height / 2)

    x_lo = x_min - math.ceil(width / 2)
    x_hi = x_min + math.ceil(width / 2)
    y_lo = y_min - math.ceil(height / 2)
    y_hi = y_min + math.ceil(height / 2)

    sigma_x = (width / 2) * 0.33
    sigma_y = (height / 2) * 0.33

    x = int(truncated_normal_sample(x_lo, x_hi, x_min, sigma_x))
    y = int(truncated_normal_sample(y_lo, y_hi, y_min, sigma_y))
    return [x, y]


# ---------------------------------------------------------------------------
# Timing distributions
# ---------------------------------------------------------------------------

def reaction_delay(mean_ms: float = 250.0,
                   min_ms: float = 120.0,
                   max_ms: float = 500.0) -> float:
    """
    Sample a human-like visual reaction time in **seconds**.

    Modelled as ex-Gaussian (normal + exponential tail), which matches
    empirical measurements of human RT distributions. The exponential tail
    accounts for occasional attentional lapses.

    :param mean_ms: Target mean reaction time in milliseconds.
    :param min_ms:  Hard lower bound (fastest possible reaction).
    :param max_ms:  Hard upper bound (slowest considered reactive).
    """
    normal_part = truncated_normal_sample(min_ms, mean_ms * 1.3,
                                          mean=mean_ms, std=45.0)
    exp_tail = np.random.exponential(28.0)  # ~28 ms scale for attentional noise
    ms = min(max_ms, normal_part + exp_tail)
    return ms / 1000.0


def idle_delay(min_s: float = 0.5, max_s: float = 3.0) -> float:
    """
    Sample a longer pause between bot actions, drawn from a chi-squared
    distribution (right-skewed — short pauses are most common, with a tail
    toward longer breaks matching human browsing/gaming behaviour).

    :returns: Delay duration in seconds.
    """
    mean_ms = (min_s + max_s) / 2 * 1000
    ms = chisquared_sample(df=int(mean_ms), min=min_s * 1000, max=max_s * 1000)
    return ms / 1000.0


def random_sleep(min_s: float, max_s: float):
    """Block for a uniformly random duration between *min_s* and *max_s* seconds."""
    time.sleep(random.uniform(min_s, max_s))


def exponential_sample(mean: float, min_val: float = 0.0,
                        max_val: float = float('inf')) -> float:
    """
    Sample from an exponential distribution with the given mean, clamped to
    [min_val, max_val].  Useful for inter-event timing (skill cooldowns,
    action cadence) where most events cluster near zero but rare long gaps occur.
    """
    while True:
        x = np.random.exponential(mean)
        if min_val <= x <= max_val:
            return x


# ---------------------------------------------------------------------------
# Core distribution primitives
# ---------------------------------------------------------------------------

def truncated_normal_sample(lower_bound: float, upper_bound: float,
                             mean: float = None, std: float = None) -> float:
    """
    Sample from a truncated normal distribution using the Box-Muller transform.

    :param lower_bound: Minimum returnable value.
    :param upper_bound: Maximum returnable value.
    :param mean: Distribution centre (default: midpoint of bounds).
    :param std:  Standard deviation (default: range / 9).
    """
    if mean is None:
        mean = (lower_bound + upper_bound) / 2
    if std is None:
        std = (upper_bound - lower_bound) / 9
    while True:
        x1, x2 = np.random.normal(0, 1), np.random.normal(0, 1)
        z = x1 ** 2 + x2 ** 2
        if 0 < z <= 1:
            sample = mean + std * x1 * np.sqrt(-2 * math.log(z) / z)
            if lower_bound <= sample <= upper_bound:
                return sample


def fancy_normal_sample(lower_bound: float, upper_bound: float) -> float:
    """
    Sample from a bimodal truncated normal with randomly-selected means,
    producing the kind of multi-modal timing patterns seen in human gameplay
    (e.g., fast reaction followed by a slower deliberate action).
    """
    means = [
        lower_bound + (upper_bound - lower_bound) * 0.33,
        lower_bound + (upper_bound - lower_bound) * 0.66,
    ]
    p = [(i + 1) ** 2 / sum((j + 1) ** 2 for j in range(len(means)))
         for i in range(len(means))][::-1]
    mean = means[np.random.choice(range(len(means)), p=p)]
    return truncated_normal_sample(lower_bound, upper_bound, mean=mean)


def chisquared_sample(df: int,
                       min: float = 0.0,
                       max: float = float('inf')) -> float:
    """
    Sample from a chi-squared distribution with *df* degrees of freedom
    (approximately the distribution mean), clamped to [min, max].
    """
    while True:
        x = np.random.chisquare(df)
        if min <= x <= max:
            return x


def random_chance(probability: float) -> bool:
    """
    Return True with the given probability.

    :param probability: Float in [0.0, 1.0].
    """
    if not isinstance(probability, (int, float)):
        raise TypeError("probability must be a float")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    return secrets.SystemRandom().random() < probability
