"""Scoring utilities for interval forecasts."""

from __future__ import annotations

from dataclasses import dataclass


EPSILON = 1e-12


@dataclass(frozen=True)
class IntervalScore:
    """Single realized interval forecast evaluation."""

    covered: bool
    width: float
    miss_distance: float
    interval_score: float
    relative_interval_score: float


def score_interval(
    actual: float,
    lower: float,
    upper: float,
    confidence: float,
    scale: float | None = None,
) -> IntervalScore:
    """Score a central prediction interval.

    Lower is better for interval_score. It rewards narrow intervals, but adds a
    penalty when the actual value falls outside the range.
    """

    if lower > upper:
        raise ValueError("lower must be <= upper")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    alpha = 1.0 - confidence
    width = upper - lower
    miss_distance = 0.0
    penalty = 0.0

    if actual < lower:
        miss_distance = lower - actual
        penalty = (2.0 / alpha) * miss_distance
    elif actual > upper:
        miss_distance = actual - upper
        penalty = (2.0 / alpha) * miss_distance

    interval_score = width + penalty
    denominator = max(abs(scale if scale is not None else actual), EPSILON)
    return IntervalScore(
        covered=lower <= actual <= upper,
        width=width,
        miss_distance=miss_distance,
        interval_score=interval_score,
        relative_interval_score=interval_score / denominator,
    )


def scores_to_weights(scores: dict[str, float]) -> dict[str, float]:
    """Convert positive model scores into normalized weights."""

    cleaned = {name: max(value, 0.0) for name, value in scores.items()}
    total = sum(cleaned.values())
    if total <= EPSILON:
        equal = 1.0 / max(len(cleaned), 1)
        return {name: equal for name in cleaned}
    return {name: value / total for name, value in cleaned.items()}
