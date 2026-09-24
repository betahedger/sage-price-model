import unittest

from stock_range_model.data import generate_synthetic_prices
from stock_range_model.ensemble import forecast_with_weighted_models
from stock_range_model.models import default_models
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


if __name__ == "__main__":
    unittest.main()
