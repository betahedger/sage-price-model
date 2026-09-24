"""Walk-forward backtesting for expected price and interval models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .models import PredictionInterval, PriceRangeModel
from .scoring import score_interval


@dataclass(frozen=True)
class BacktestResult:
    """Backtest rows and per-model metrics."""

    rows: pd.DataFrame
    metrics: pd.DataFrame


def run_backtest(
    prices: pd.Series,
    models: list[PriceRangeModel],
    min_train_days: int = 252 * 3,
    horizon_days: int = 252,
    step_days: int = 252,
    confidence: float = 0.80,
) -> BacktestResult:
    """Run a walk-forward expected-price and interval backtest.

    Each test point trains on data through a historical date, predicts the price
    horizon, then compares the expected price and interval with the realized
    price.
    """

    cleaned = pd.Series(prices).astype(float).dropna()
    cleaned = cleaned[cleaned > 0]
    if len(cleaned) < min_train_days + horizon_days + 1:
        raise ValueError(
            "not enough prices for backtest: need at least "
            f"{min_train_days + horizon_days + 1}, got {len(cleaned)}"
        )

    rows: list[dict[str, object]] = []
    last_origin = len(cleaned) - horizon_days - 1

    for origin_idx in range(min_train_days, last_origin + 1, step_days):
        train = cleaned.iloc[: origin_idx + 1]
        current_price = float(cleaned.iloc[origin_idx])
        actual_price = float(cleaned.iloc[origin_idx + horizon_days])
        origin_date = cleaned.index[origin_idx]
        target_date = cleaned.index[origin_idx + horizon_days]

        for model in models:
            fitted = model.fit(train.copy())
            prediction: PredictionInterval = fitted.predict(
                current_price=current_price,
                horizon_days=horizon_days,
                confidence=confidence,
            )
            score = score_interval(
                actual=actual_price,
                lower=prediction.lower,
                upper=prediction.upper,
                confidence=confidence,
                scale=current_price,
            )
            expected_error = actual_price - prediction.expected
            relative_expected_error = expected_error / current_price
            absolute_relative_expected_error = abs(relative_expected_error)
            rows.append(
                {
                    "origin_date": origin_date,
                    "target_date": target_date,
                    "model": prediction.model_name,
                    "current_price": current_price,
                    "actual_price": actual_price,
                    "lower": prediction.lower,
                    "expected": prediction.expected,
                    "upper": prediction.upper,
                    "expected_error": expected_error,
                    "absolute_expected_error": abs(expected_error),
                    "relative_expected_error": relative_expected_error,
                    "absolute_relative_expected_error": absolute_relative_expected_error,
                    "covered": score.covered,
                    "width": score.width,
                    "miss_distance": score.miss_distance,
                    "interval_score": score.interval_score,
                    "relative_interval_score": score.relative_interval_score,
                }
            )

    row_frame = pd.DataFrame(rows)
    if row_frame.empty:
        raise ValueError("backtest produced no rows; reduce min_train_days or step_days")

    row_frame["squared_relative_expected_error"] = row_frame["relative_expected_error"] ** 2
    metrics = (
        row_frame.groupby("model")
        .agg(
            tests=("covered", "size"),
            avg_absolute_relative_expected_error=("absolute_relative_expected_error", "mean"),
            median_absolute_relative_expected_error=("absolute_relative_expected_error", "median"),
            expected_rmse_relative=(
                "squared_relative_expected_error",
                lambda value: float(value.mean() ** 0.5),
            ),
            coverage_rate=("covered", "mean"),
            avg_relative_interval_score=("relative_interval_score", "mean"),
            avg_relative_width=("width", lambda value: float((value / row_frame.loc[value.index, "current_price"]).mean())),
            avg_relative_miss=("miss_distance", lambda value: float((value / row_frame.loc[value.index, "current_price"]).mean())),
        )
        .reset_index()
    )
    metrics["raw_weight_score"] = 1.0 / (
        metrics["avg_absolute_relative_expected_error"] + 1e-12
    )
    metrics = metrics.sort_values("raw_weight_score", ascending=False).reset_index(drop=True)
    return BacktestResult(rows=row_frame, metrics=metrics)
