"""Forecast models that output one-year expected prices and reference intervals."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Protocol

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PredictionInterval:
    """A model's expected price forecast with an auxiliary interval."""

    model_name: str
    lower: float
    expected: float
    upper: float


@dataclass(frozen=True)
class ModelContext:
    """Optional market, factor, dividend, and accounting inputs for final models."""

    market_prices: pd.Series | None = None
    ff_factors: pd.DataFrame | None = None
    macro_factors: pd.DataFrame | None = None
    dividends: pd.Series | None = None
    risk_free_rate: float = 0.03
    risk_free_rates: pd.Series | None = None
    required_return: float | None = None
    book_value_per_share: float | None = None
    roe: float | None = None
    terminal_growth: float | None = None
    dividend_yield_fallback: float = 0.02
    price_to_book_fallback: float = 1.5


class PriceRangeModel(Protocol):
    """Interface shared by every price range model."""

    name: str

    def fit(self, prices: pd.Series) -> "PriceRangeModel":
        """Fit the model to historical adjusted close prices."""

    def predict(
        self,
        current_price: float,
        horizon_days: int,
        confidence: float,
    ) -> PredictionInterval:
        """Predict lower/expected/upper price for the forecast horizon."""


def _clean_prices(prices: pd.Series) -> pd.Series:
    cleaned = pd.Series(prices).astype(float).dropna()
    cleaned = cleaned[cleaned > 0]
    if len(cleaned) < 30:
        raise ValueError("at least 30 positive prices are required")
    return cleaned


def _log_returns(prices: pd.Series) -> pd.Series:
    cleaned = _clean_prices(prices)
    returns = np.log(cleaned / cleaned.shift(1)).dropna()
    if returns.empty:
        raise ValueError("not enough data to calculate returns")
    return returns


def _z_value(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def _lognormal_interval(
    model_name: str,
    current_price: float,
    horizon_days: int,
    confidence: float,
    daily_mu: float,
    daily_sigma: float,
) -> PredictionInterval:
    horizon_mu = daily_mu * horizon_days
    horizon_sigma = max(daily_sigma, 1e-12) * np.sqrt(horizon_days)
    z = _z_value(confidence)

    lower = current_price * float(np.exp(horizon_mu - z * horizon_sigma))
    expected = current_price * float(np.exp(horizon_mu))
    upper = current_price * float(np.exp(horizon_mu + z * horizon_sigma))
    return PredictionInterval(model_name, lower, expected, upper)


def _safe_context(context: ModelContext | None) -> ModelContext:
    return context if context is not None else ModelContext()


def _risk_free_annual(context: ModelContext, as_of: object) -> float:
    """Use only a rate observed by the end of the model's training window."""

    rates = context.risk_free_rates
    if rates is not None and not rates.empty:
        available = rates.loc[rates.index <= pd.Timestamp(as_of)].dropna().sort_index()
        if not available.empty:
            return float(available.iloc[-1])
    return float(context.risk_free_rate)


def _risk_free_daily(context: ModelContext, as_of: object) -> float:
    annual_rate = _risk_free_annual(context, as_of)
    return float(np.log1p(max(annual_rate, -0.99)) / TRADING_DAYS_PER_YEAR)


def _annual_to_daily_log(annual_return: float) -> float:
    return float(np.log1p(max(annual_return, -0.99)) / TRADING_DAYS_PER_YEAR)


def _annualized_log_return(prices: pd.Series) -> float:
    returns = _log_returns(prices)
    return float(returns.mean() * TRADING_DAYS_PER_YEAR)


def _clip_annual_growth(value: float, lower: float = -0.05, upper: float = 0.08) -> float:
    if not np.isfinite(value):
        return 0.02
    return float(np.clip(value, lower, upper))


def _clip_expected_price(expected: float, current_price: float) -> float:
    if not np.isfinite(expected) or expected <= 0:
        return current_price
    return float(np.clip(expected, current_price * 0.2, current_price * 3.0))


def _interval_around_expected(
    model_name: str,
    current_price: float,
    expected_price: float,
    daily_sigma: float,
    horizon_days: int,
    confidence: float,
) -> PredictionInterval:
    expected = _clip_expected_price(expected_price, current_price)
    horizon_sigma = max(float(daily_sigma), 1e-12) * np.sqrt(horizon_days)
    z = _z_value(confidence)
    lower = expected * float(np.exp(-z * horizon_sigma))
    upper = expected * float(np.exp(z * horizon_sigma))
    return PredictionInterval(model_name, lower, expected, upper)


def _returns_frame(prices: pd.Series, name: str) -> pd.DataFrame:
    return _log_returns(prices).rename(name).to_frame()


def _market_returns(context: ModelContext, fallback_prices: pd.Series) -> pd.Series:
    market_prices = context.market_prices if context.market_prices is not None else fallback_prices
    return _log_returns(market_prices).rename("MKT")


def _aligned_regression_frame(
    prices: pd.Series,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    asset_returns = _log_returns(prices).rename("asset")
    frame = pd.concat([asset_returns, factors], axis=1).dropna()
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def _ols(y: pd.Series, x: pd.DataFrame) -> tuple[float, np.ndarray]:
    y_values = y.to_numpy(dtype=float)
    x_values = x.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x_values)), x_values])
    coefficients = np.linalg.lstsq(design, y_values, rcond=None)[0]
    return float(coefficients[0]), coefficients[1:]


def _residual_sigma(y: pd.Series, x: pd.DataFrame, alpha: float, betas: np.ndarray) -> float:
    fitted = alpha + x.to_numpy(dtype=float) @ betas
    residuals = y.to_numpy(dtype=float) - fitted
    if len(residuals) < 2:
        return float(y.std(ddof=1))
    return float(max(np.std(residuals, ddof=1), 1e-12))


@dataclass
class CAPMModel:
    """CAPM expected return model using market beta."""

    context: ModelContext | None = None
    name: str = "CAPM"
    expected_daily_return: float = 0.0
    daily_sigma: float = 0.0
    beta: float = 1.0

    def fit(self, prices: pd.Series) -> "CAPMModel":
        context = _safe_context(self.context)
        asset_returns = _log_returns(prices).rename("asset")
        market_returns = _market_returns(context, prices)
        frame = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if len(frame) < 30:
            self.expected_daily_return = float(asset_returns.mean())
            self.daily_sigma = float(asset_returns.std(ddof=1))
            return self

        rf_daily = _risk_free_daily(context, prices.index.max())
        market_excess = frame["MKT"] - rf_daily
        asset_excess = frame["asset"] - rf_daily
        market_var = float(market_excess.var(ddof=1))
        if market_var <= 1e-12:
            self.beta = 1.0
        else:
            self.beta = float(asset_excess.cov(market_excess) / market_var)

        self.expected_daily_return = float(rf_daily + self.beta * market_excess.mean())
        residuals = asset_excess - self.beta * market_excess
        self.daily_sigma = float(max(residuals.std(ddof=1), asset_returns.std(ddof=1) * 0.25, 1e-12))
        return self

    def predict(
        self,
        current_price: float,
        horizon_days: int,
        confidence: float,
    ) -> PredictionInterval:
        return _lognormal_interval(
            self.name,
            current_price,
            horizon_days,
            confidence,
            self.expected_daily_return,
            self.daily_sigma,
        )


@dataclass
class FamaFrenchFactorModel:
    """Fama-French-style factor model using market, SMB, and HML proxies."""

    context: ModelContext | None = None
    name: str = "FamaFrench3"
    expected_daily_return: float = 0.0
    daily_sigma: float = 0.0

    def fit(self, prices: pd.Series) -> "FamaFrenchFactorModel":
        context = _safe_context(self.context)
        rf_daily = _risk_free_daily(context, prices.index.max())
        market_excess = (_market_returns(context, prices) - rf_daily).rename("MKT")

        if context.ff_factors is not None and not context.ff_factors.empty:
            factor_frame = context.ff_factors.copy()
        else:
            factor_frame = pd.DataFrame(index=market_excess.index)
        factor_frame["MKT"] = market_excess
        if "SMB" not in factor_frame:
            factor_frame["SMB"] = 0.0
        if "HML" not in factor_frame:
            factor_frame["HML"] = 0.0
        factor_frame = factor_frame[["MKT", "SMB", "HML"]]

        frame = _aligned_regression_frame(prices, factor_frame)
        if len(frame) < 40:
            returns = _log_returns(prices)
            self.expected_daily_return = float(returns.mean())
            self.daily_sigma = float(returns.std(ddof=1))
            return self

        y = frame["asset"] - rf_daily
        x = frame[["MKT", "SMB", "HML"]]
        alpha, betas = _ols(y, x)
        factor_means = x.mean().to_numpy(dtype=float)
        # The fitted intercept is an in-sample residual, not a forecast of alpha.
        self.expected_daily_return = float(rf_daily + factor_means @ betas)
        self.daily_sigma = _residual_sigma(y, x, alpha, betas)
        return self

    def predict(
        self,
        current_price: float,
        horizon_days: int,
        confidence: float,
    ) -> PredictionInterval:
        return _lognormal_interval(
            self.name,
            current_price,
            horizon_days,
            confidence,
            self.expected_daily_return,
            self.daily_sigma,
        )


@dataclass
class APTMacroFactorModel:
    """APT-style macro factor model using market, rate, and FX proxies."""

    context: ModelContext | None = None
    name: str = "APTMacro"
    expected_daily_return: float = 0.0
    daily_sigma: float = 0.0

    def fit(self, prices: pd.Series) -> "APTMacroFactorModel":
        context = _safe_context(self.context)
        rf_daily = _risk_free_daily(context, prices.index.max())
        market = (_market_returns(context, prices) - rf_daily).rename("MKT")

        if context.macro_factors is not None and not context.macro_factors.empty:
            factor_frame = context.macro_factors.copy()
        else:
            factor_frame = pd.DataFrame(index=market.index)
        factor_frame["MKT"] = market
        if "RATE_CHANGE" not in factor_frame:
            factor_frame["RATE_CHANGE"] = 0.0
        if "FX_RETURN" not in factor_frame:
            factor_frame["FX_RETURN"] = 0.0
        factor_frame = factor_frame[["MKT", "RATE_CHANGE", "FX_RETURN"]]

        frame = _aligned_regression_frame(prices, factor_frame)
        if len(frame) < 40:
            returns = _log_returns(prices)
            self.expected_daily_return = float(returns.mean())
            self.daily_sigma = float(returns.std(ddof=1))
            return self

        y = frame["asset"] - rf_daily
        x = frame[["MKT", "RATE_CHANGE", "FX_RETURN"]]
        alpha, betas = _ols(y, x)
        factor_means = x.mean().to_numpy(dtype=float)
        self.expected_daily_return = float(rf_daily + factor_means @ betas)
        self.daily_sigma = _residual_sigma(y, x, alpha, betas)
        return self

    def predict(
        self,
        current_price: float,
        horizon_days: int,
        confidence: float,
    ) -> PredictionInterval:
        return _lognormal_interval(
            self.name,
            current_price,
            horizon_days,
            confidence,
            self.expected_daily_return,
            self.daily_sigma,
        )


@dataclass
class GordonGrowthModel:
    """Dividend Discount Model / Gordon Growth valuation model."""

    context: ModelContext | None = None
    name: str = "DDM_Gordon"
    annual_dividend: float = 0.0
    dividend_growth: float = 0.02
    required_return: float = 0.09
    daily_sigma: float = 0.0
    dividend_yield_fallback: float = 0.02

    def fit(self, prices: pd.Series) -> "GordonGrowthModel":
        context = _safe_context(self.context)
        cleaned = _clean_prices(prices)
        returns = _log_returns(cleaned)
        self.daily_sigma = float(returns.std(ddof=1))
        self.required_return = float(
            context.required_return
            if context.required_return is not None
            else max(_risk_free_annual(context, cleaned.index.max()) + 0.06, 0.04)
        )
        self.dividend_yield_fallback = context.dividend_yield_fallback

        dividends = context.dividends
        if dividends is not None and not dividends.empty:
            train_end = cleaned.index.max()
            train_start = train_end - pd.Timedelta(days=365)
            trailing = dividends[(dividends.index > train_start) & (dividends.index <= train_end)]
            self.annual_dividend = float(trailing.sum())

            yearly = dividends[dividends.index <= train_end].resample("YE").sum()
            yearly = yearly[yearly > 0]
            if len(yearly) >= 2 and yearly.iloc[0] > 0:
                years = max((yearly.index[-1] - yearly.index[0]).days / 365.25, 1.0)
                growth = (yearly.iloc[-1] / yearly.iloc[0]) ** (1.0 / years) - 1.0
                self.dividend_growth = _clip_annual_growth(float(growth), -0.05, 0.06)
            else:
                self.dividend_growth = _clip_annual_growth(_annualized_log_return(cleaned), -0.05, 0.06)
        else:
            self.annual_dividend = 0.0
            self.dividend_growth = _clip_annual_growth(_annualized_log_return(cleaned), -0.05, 0.06)
        return self

    def predict(
        self,
        current_price: float,
        horizon_days: int,
        confidence: float,
    ) -> PredictionInterval:
        annual_dividend = self.annual_dividend
        if annual_dividend <= 0:
            annual_dividend = current_price * self.dividend_yield_fallback

        growth = min(self.dividend_growth, self.required_return - 0.01)
        denominator = max(self.required_return - growth, 0.01)
        next_dividend = annual_dividend * (1.0 + growth)
        fair_value = next_dividend / denominator
        horizon_fraction = horizon_days / TRADING_DAYS_PER_YEAR
        expected = current_price + (fair_value - current_price) * min(horizon_fraction, 1.0)
        return _interval_around_expected(
            self.name,
            current_price,
            expected,
            self.daily_sigma,
            horizon_days,
            confidence,
        )


@dataclass
class ResidualIncomeModel:
    """Residual Income valuation model using accounting inputs or proxies."""

    context: ModelContext | None = None
    name: str = "ResidualIncome"
    cost_of_equity: float = 0.09
    roe: float = 0.10
    terminal_growth: float = 0.03
    daily_sigma: float = 0.0
    book_value_per_share: float | None = None
    price_to_book_fallback: float = 1.5

    def fit(self, prices: pd.Series) -> "ResidualIncomeModel":
        context = _safe_context(self.context)
        cleaned = _clean_prices(prices)
        returns = _log_returns(cleaned)
        self.daily_sigma = float(returns.std(ddof=1))
        self.cost_of_equity = float(
            context.required_return
            if context.required_return is not None
            else max(_risk_free_annual(context, cleaned.index.max()) + 0.06, 0.04)
        )
        historical_growth = _clip_annual_growth(_annualized_log_return(cleaned), -0.05, 0.06)
        self.roe = float(context.roe if context.roe is not None else np.clip(0.10 + historical_growth / 2.0, 0.02, 0.25))
        self.terminal_growth = float(
            context.terminal_growth
            if context.terminal_growth is not None
            else np.clip(historical_growth, 0.00, min(0.04, self.cost_of_equity - 0.01))
        )
        self.book_value_per_share = context.book_value_per_share
        self.price_to_book_fallback = context.price_to_book_fallback
        return self

    def predict(
        self,
        current_price: float,
        horizon_days: int,
        confidence: float,
    ) -> PredictionInterval:
        book_value = self.book_value_per_share
        if book_value is None or book_value <= 0:
            book_value = current_price / max(self.price_to_book_fallback, 0.1)

        growth = min(self.terminal_growth, self.cost_of_equity - 0.01)
        denominator = max(self.cost_of_equity - growth, 0.01)
        residual_income = book_value * (self.roe - self.cost_of_equity)
        fair_value = book_value + residual_income / denominator
        horizon_fraction = horizon_days / TRADING_DAYS_PER_YEAR
        expected = current_price + (fair_value - current_price) * min(horizon_fraction, 1.0)
        return _interval_around_expected(
            self.name,
            current_price,
            expected,
            self.daily_sigma,
            horizon_days,
            confidence,
        )


def default_models(context: ModelContext | None = None) -> list[PriceRangeModel]:
    """Return the five confirmed asset pricing and valuation models."""

    return [
        CAPMModel(context=context),
        APTMacroFactorModel(context=context),
        FamaFrenchFactorModel(context=context),
        GordonGrowthModel(context=context),
        ResidualIncomeModel(context=context),
    ]
