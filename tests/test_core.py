import unittest

import numpy as np
import pandas as pd

from stock_range_model.data import generate_synthetic_prices
from stock_range_model.ensemble import (
    _build_ensemble_expected_backtest,
    forecast_with_weighted_models,
)
from stock_range_model.models import (
    APTMacroFactorModel,
    FamaFrenchFactorModel,
    GordonGrowthModel,
    ModelContext,
    default_models,
)
from stock_range_model.scoring import score_interval, scores_to_weights


class ScoringTests(unittest.TestCase):
    def test_interval_score_rewards_covered_intervals(self):
        covered = score_interval(actual=100, lower=90, upper=110, confidence=0.8)
        missed = score_interval(actual=130, lower=90, upper=110, confidence=0.8)

        self.assertTrue(covered.covered)
        self.assertFalse(missed.covered)
        self.assertGreater(missed.interval_score, covered.interval_score)

    def test_scores_to_weights_normalizes(self):
        weights = scores_to_weights({"a": 2.0, "b": 1.0})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["a"], weights["b"])


class ForecastTests(unittest.TestCase):
    def test_forecast_with_weighted_models_runs_on_synthetic_data(self):
        prices = generate_synthetic_prices(years=6, seed=7)
        models = default_models()

        forecast = forecast_with_weighted_models(
            prices,
            models=models,
            min_train_days=252 * 2,
            horizon_days=126,
            step_days=126,
            confidence=0.8,
        )

        self.assertLessEqual(forecast.lower, forecast.expected)
        self.assertLessEqual(forecast.expected, forecast.upper)
        self.assertAlmostEqual(
            float(forecast.model_predictions["weighted_expected"].sum()),
            forecast.expected,
        )
        self.assertAlmostEqual(float(forecast.weights["weight"].sum()), 1.0)
        self.assertIn("avg_absolute_relative_expected_error", forecast.weights.columns)
        self.assertEqual(len(forecast.model_predictions), 5)
        self.assertFalse(forecast.ensemble_backtest.empty)
        self.assertFalse(forecast.backtest.rows.empty)

    def test_factor_forecasts_do_not_reuse_fitted_intercept_as_future_alpha(self):
        rng = np.random.default_rng(17)
        dates = pd.bdate_range("2020-01-01", periods=121)
        market_returns = rng.normal(0.0002, 0.01, 120)
        smb = rng.normal(0.0001, 0.006, 120)
        hml = rng.normal(-0.0001, 0.005, 120)
        rate_change = rng.normal(0, 0.001, 120)
        fx_returns = rng.normal(0.0001, 0.004, 120)
        asset_returns = (
            0.001 + 1.1 * market_returns + 0.3 * smb
            - 0.2 * hml + 0.5 * rate_change + 0.4 * fx_returns
        )
        prices = pd.Series(100 * np.exp(np.r_[0, asset_returns].cumsum()), index=dates)
        market_prices = pd.Series(
            100 * np.exp(np.r_[0, market_returns].cumsum()), index=dates
        )
        context = ModelContext(
            market_prices=market_prices,
            ff_factors=pd.DataFrame({"SMB": smb, "HML": hml}, index=dates[1:]),
            macro_factors=pd.DataFrame(
                {"RATE_CHANGE": rate_change, "FX_RETURN": fx_returns},
                index=dates[1:],
            ),
        )

        ff = FamaFrenchFactorModel(context=context).fit(prices)
        apt = APTMacroFactorModel(context=context).fit(prices)
        historical_mean = float(asset_returns.mean())
        self.assertGreater(abs(ff.expected_daily_return - historical_mean), 0.0005)
        self.assertGreater(abs(apt.expected_daily_return - historical_mean), 0.0005)

    def test_historical_fit_uses_only_risk_free_rates_available_at_that_date(self):
        prices = generate_synthetic_prices(years=2, seed=7)
        early_date = prices.index[200]
        future_date = prices.index[400]
        rates = pd.Series([0.01, 0.10], index=[prices.index[0], future_date])
        context = ModelContext(risk_free_rate=0.03, risk_free_rates=rates)

        early_model = GordonGrowthModel(context=context).fit(prices.loc[:early_date])
        late_model = GordonGrowthModel(context=context).fit(prices)
        self.assertAlmostEqual(early_model.required_return, 0.07)
        self.assertAlmostEqual(late_model.required_return, 0.16)

    def test_ensemble_backtest_weights_use_only_completed_prior_outcomes(self):
        dates = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"])
        rows = []
        for index, actual in enumerate([100.0, 200.0, 150.0]):
            for model, expected in [("A", 100.0), ("B", 200.0)]:
                rows.append({
                    "origin_date": dates[index],
                    "target_date": dates[index + 1],
                    "model": model,
                    "current_price": 100.0,
                    "actual_price": actual,
                    "expected": expected,
                    "absolute_relative_expected_error": abs(actual - expected) / 100.0,
                })

        result = _build_ensemble_expected_backtest(pd.DataFrame(rows))
        self.assertEqual(result["completed_prior_tests"].tolist(), [0, 1, 2])
        self.assertAlmostEqual(result.loc[0, "ensemble_expected"], 150.0)
        self.assertAlmostEqual(result.loc[1, "ensemble_expected"], 100.0, places=6)
        self.assertAlmostEqual(result.loc[2, "ensemble_expected"], 150.0)


if __name__ == "__main__":
    unittest.main()
