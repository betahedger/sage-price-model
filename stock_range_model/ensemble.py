"""Weighted ensemble forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .models import PredictionInterval, PriceRangeModel
from .scoring import scores_to_weights


@dataclass(frozen=True)
class EnsembleForecast:
    """Final forecast and diagnostics."""

    current_price: float
    lower: float
    expected: float
    upper: float
    expected_return: float
    reference_margin: float
    weighted_interval_lower: float
    weighted_interval_upper: float
    weights: pd.DataFrame
    model_predictions: pd.DataFrame
    ensemble_backtest: pd.DataFrame
    backtest: BacktestResult


def forecast_with_weighted_models(
    prices: pd.Series,
    models: list[PriceRangeModel],
    min_train_days: int = 252 * 3,
    horizon_days: int = 252,
    step_days: int = 252,
    confidence: float = 0.80,
) -> EnsembleForecast:
    """Backtest models, weight them, and forecast the next expected price.

    The main forecast is a single weighted expected price. The lower and upper
    values are a reference range based on historical ensemble expected-price
    errors, not a direct weighted average of every model's wide interval.
    """

    cleaned = pd.Series(prices).astype(float).dropna()
    cleaned = cleaned[cleaned > 0]
    backtest = run_backtest(
        cleaned,
        models=models,
        min_train_days=min_train_days,
        horizon_days=horizon_days,
        step_days=step_days,
        confidence=confidence,
    )

    raw_scores = dict(zip(backtest.metrics["model"], backtest.metrics["raw_weight_score"]))
    weight_map = scores_to_weights(raw_scores)
    weights = backtest.metrics.copy()
    weights["weight"] = weights["model"].map(weight_map)
    weights = weights.sort_values("weight", ascending=False).reset_index(drop=True)

    current_price = float(cleaned.iloc[-1])
    predictions: list[PredictionInterval] = []
    for model in models:
        fitted = model.fit(cleaned.copy())
        predictions.append(
            fitted.predict(
                current_price=current_price,
                horizon_days=horizon_days,
                confidence=confidence,
            )
        )

    prediction_frame = pd.DataFrame(
        [
            {
                "model": prediction.model_name,
                "lower": prediction.lower,
                "expected": prediction.expected,
                "upper": prediction.upper,
                "weight": weight_map.get(prediction.model_name, 0.0),
            }
            for prediction in predictions
        ]
    )
    for column in ["lower", "expected", "upper"]:
        prediction_frame[f"weighted_{column}"] = prediction_frame[column] * prediction_frame["weight"]
    prediction_frame["expected_return"] = prediction_frame["expected"] / current_price - 1.0
    prediction_frame["weighted_expected_return"] = (
        prediction_frame["expected_return"] * prediction_frame["weight"]
    )

    weighted_expected = float(prediction_frame["weighted_expected"].sum())
    weighted_interval_lower = float(prediction_frame["weighted_lower"].sum())
    weighted_interval_upper = float(prediction_frame["weighted_upper"].sum())
    ensemble_backtest = _build_ensemble_expected_backtest(backtest.rows)
    reference_margin = _reference_margin_from_backtest(
        ensemble_backtest,
        confidence=confidence,
    )
    reference_lower = max(0.0, weighted_expected * (1.0 - reference_margin))
    reference_upper = weighted_expected * (1.0 + reference_margin)

    return EnsembleForecast(
        current_price=current_price,
        lower=reference_lower,
        expected=weighted_expected,
        upper=reference_upper,
        expected_return=weighted_expected / current_price - 1.0,
        reference_margin=reference_margin,
        weighted_interval_lower=weighted_interval_lower,
        weighted_interval_upper=weighted_interval_upper,
        weights=weights,
        model_predictions=prediction_frame.sort_values("weight", ascending=False).reset_index(drop=True),
        ensemble_backtest=ensemble_backtest,
        backtest=backtest,
    )


def _build_ensemble_expected_backtest(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Combine each historical forecast using only outcomes known by its date."""

    model_names = list(rows["model"].unique())
    origins = rows["origin_date"].drop_duplicates().sort_values()
    combined_rows: list[dict[str, object]] = []

    for origin_date in origins:
        known = rows.loc[
            (rows["target_date"] <= origin_date)
            & (rows["origin_date"] < origin_date)
        ]
        if known.empty:
            weights = {name: 1.0 / len(model_names) for name in model_names}
        else:
            errors = known.groupby("model")["absolute_relative_expected_error"].mean()
            weights = scores_to_weights(
                {name: 1.0 / (float(errors[name]) + 1e-12) for name in model_names}
            )

        current = rows.loc[rows["origin_date"] == origin_date]
        expected = sum(
            float(row.expected) * weights[row.model]
            for row in current.itertuples(index=False)
        )
        first = current.iloc[0]
        combined_rows.append(
            {
                "origin_date": origin_date,
                "target_date": first["target_date"],
                "current_price": float(first["current_price"]),
                "actual_price": float(first["actual_price"]),
                "ensemble_expected": expected,
                "completed_prior_tests": int(known["origin_date"].nunique()),
            }
        )

    ensemble = pd.DataFrame(combined_rows)
    ensemble["expected_error"] = ensemble["actual_price"] - ensemble["ensemble_expected"]
    ensemble["absolute_expected_error"] = ensemble["expected_error"].abs()
    ensemble["absolute_percentage_error"] = (
        ensemble["absolute_expected_error"] / ensemble["ensemble_expected"].abs().clip(lower=1e-12)
    )
    ensemble["relative_error_to_current"] = ensemble["expected_error"] / ensemble["current_price"]
    return ensemble


def _reference_margin_from_backtest(
    ensemble_backtest: pd.DataFrame,
    confidence: float,
) -> float:
    if ensemble_backtest.empty:
        return 0.0
    margin = float(ensemble_backtest["absolute_percentage_error"].quantile(confidence))
    return max(margin, 0.0)
