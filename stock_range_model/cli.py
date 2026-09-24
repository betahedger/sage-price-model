"""Command line entry point for weighted expected price forecasting."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    download_yahoo_dividends,
    download_yahoo_prices,
    generate_synthetic_prices,
    load_csv_prices,
)
from .ensemble import forecast_with_weighted_models
from .models import ModelContext, default_models


def main() -> None:
    args = parse_args()
    prices = load_prices_from_args(args)

    min_train_days = int(args.min_train_years * 252)
    context = build_model_context(args, prices)
    models = default_models(context=context)
    forecast = forecast_with_weighted_models(
        prices,
        models=models,
        min_train_days=min_train_days,
        horizon_days=args.horizon_days,
        step_days=args.step_days,
        confidence=args.confidence,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast.weights.to_csv(output_dir / "model_weights.csv", index=False)
    forecast.model_predictions.to_csv(output_dir / "model_predictions.csv", index=False)
    forecast.backtest.rows.to_csv(output_dir / "backtest_rows.csv", index=False)
    forecast.ensemble_backtest.to_csv(output_dir / "ensemble_backtest_rows.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "current_price": forecast.current_price,
                "forecast_expected": forecast.expected,
                "expected_return": forecast.expected_return,
                "reference_lower": forecast.lower,
                "reference_upper": forecast.upper,
                "reference_margin": forecast.reference_margin,
                "weighted_interval_lower": forecast.weighted_interval_lower,
                "weighted_interval_upper": forecast.weighted_interval_upper,
                "confidence": args.confidence,
                "horizon_days": args.horizon_days,
            }
        ]
    )
    summary.to_csv(output_dir / "forecast_summary.csv", index=False)

    print()
    print("Weighted one-year expected price forecast")
    print("-----------------------------------------")
    print(f"Current price : {summary.loc[0, 'current_price']:,.2f}")
    print(f"Expected price: {forecast.expected:,.2f}")
    print(f"Expected return: {forecast.expected_return:.2%}")
    print(
        f"Reference range ({args.confidence:.0%} historical error): "
        f"{forecast.lower:,.2f} ~ {forecast.upper:,.2f}"
    )
    print()
    print("Model weights")
    print(
        forecast.weights[
            [
                "model",
                "avg_absolute_relative_expected_error",
                "expected_rmse_relative",
                "coverage_rate",
                "weight",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Saved CSV outputs to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest five models and forecast a weighted one-year expected price."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--ticker", help="Yahoo Finance ticker, e.g. AAPL or 005930.KS")
    source.add_argument("--csv", help="CSV file with Date and Adj Close/Close columns")
    source.add_argument("--synthetic", action="store_true", help="Use generated demo data")

    parser.add_argument("--price-column", help="Price column name when using --csv")
    parser.add_argument("--years", type=int, default=10, help="Years of data to download or generate")
    parser.add_argument("--start", help="Download start date YYYY-MM-DD")
    parser.add_argument("--end", help="Download end date YYYY-MM-DD")
    parser.add_argument("--confidence", type=float, default=0.80, help="Interval confidence, e.g. 0.80")
    parser.add_argument("--horizon-days", type=int, default=252, help="Forecast horizon in trading days")
    parser.add_argument("--min-train-years", type=float, default=3.0, help="Minimum training window in years")
    parser.add_argument("--step-days", type=int, default=252, help="Walk-forward backtest step size")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV outputs")
    parser.add_argument("--market-ticker", help="Market proxy ticker, e.g. ^GSPC or ^KS11")
    parser.add_argument("--risk-free-rate", type=float, help="Annual risk-free rate as decimal, e.g. 0.035")
    parser.add_argument("--required-return", type=float, help="Annual required return for DDM/Residual Income")
    parser.add_argument("--book-value-per-share", type=float, help="Book value per share for Residual Income")
    parser.add_argument("--roe", type=float, help="Expected ROE for Residual Income, e.g. 0.10")
    parser.add_argument("--terminal-growth", type=float, help="Terminal growth for Residual Income, e.g. 0.03")
    parser.add_argument("--dividend-yield-fallback", type=float, default=0.02, help="Fallback dividend yield for DDM")
    parser.add_argument("--price-to-book-fallback", type=float, default=1.5, help="Fallback P/B for Residual Income")
    return parser.parse_args()


def load_prices_from_args(args: argparse.Namespace) -> pd.Series:
    if args.csv:
        return load_csv_prices(args.csv, price_column=args.price_column)

    if args.ticker:
        end = args.end or date.today().isoformat()
        start = args.start
        if start is None:
            start_date = date.fromisoformat(end) - timedelta(days=int(args.years * 365.25))
            start = start_date.isoformat()
        return download_yahoo_prices(args.ticker, start=start, end=end)

    return generate_synthetic_prices(years=args.years)


def build_model_context(args: argparse.Namespace, prices: pd.Series) -> ModelContext:
    start = prices.index.min().date()
    end = prices.index.max().date()

    market_prices = None
    ff_factors = None
    macro_factors = None
    dividends = None
    risk_free_rate = args.risk_free_rate

    if args.ticker:
        market_ticker = args.market_ticker or infer_market_ticker(args.ticker)
        market_prices = safe_download_prices(market_ticker, start, end)
        ff_factors = build_fama_french_proxy_factors(start, end)
        macro_factors = build_macro_proxy_factors(start, end)
        dividends = safe_download_dividends(args.ticker, start, end)
        if risk_free_rate is None:
            risk_free_rate = infer_risk_free_rate(start, end)

    return ModelContext(
        market_prices=market_prices,
        ff_factors=ff_factors,
        macro_factors=macro_factors,
        dividends=dividends,
        risk_free_rate=0.03 if risk_free_rate is None else risk_free_rate,
        required_return=args.required_return,
        book_value_per_share=args.book_value_per_share,
        roe=args.roe,
        terminal_growth=args.terminal_growth,
        dividend_yield_fallback=args.dividend_yield_fallback,
        price_to_book_fallback=args.price_to_book_fallback,
    )


def infer_market_ticker(ticker: str) -> str:
    if ticker.upper().endswith(".KS") or ticker.upper().endswith(".KQ"):
        return "^KS11"
    return "^GSPC"


def safe_download_prices(ticker: str, start: date, end: date) -> pd.Series | None:
    try:
        return download_yahoo_prices(ticker, start=start, end=end)
    except Exception:
        return None


def safe_download_dividends(ticker: str, start: date, end: date) -> pd.Series | None:
    try:
        return download_yahoo_dividends(ticker, start=start, end=end)
    except Exception:
        return None


def infer_risk_free_rate(start: date, end: date) -> float:
    rates = safe_download_prices("^TNX", start, end)
    if rates is None or rates.empty:
        return 0.03
    return float(rates.dropna().iloc[-1] / 100.0)


def build_fama_french_proxy_factors(start: date, end: date) -> pd.DataFrame | None:
    spy = safe_download_prices("SPY", start, end)
    iwm = safe_download_prices("IWM", start, end)
    ive = safe_download_prices("IVE", start, end)
    ivw = safe_download_prices("IVW", start, end)
    if any(series is None or series.empty for series in [spy, iwm, ive, ivw]):
        return None

    spy_returns = log_returns(spy, "SPY")
    smb = log_returns(iwm, "IWM") - spy_returns
    hml = log_returns(ive, "IVE") - log_returns(ivw, "IVW")
    factors = pd.concat([smb.rename("SMB"), hml.rename("HML")], axis=1).dropna()
    return factors if not factors.empty else None


def build_macro_proxy_factors(start: date, end: date) -> pd.DataFrame | None:
    rates = safe_download_prices("^TNX", start, end)
    fx = safe_download_prices("KRW=X", start, end)
    factors: list[pd.Series] = []

    if rates is not None and not rates.empty:
        rate_change = (rates.astype(float) / 100.0).diff().dropna().rename("RATE_CHANGE")
        factors.append(rate_change)
    if fx is not None and not fx.empty:
        factors.append(log_returns(fx, "FX_RETURN"))

    if not factors:
        return None
    frame = pd.concat(factors, axis=1).dropna(how="all").fillna(0.0)
    return frame if not frame.empty else None


def log_returns(prices: pd.Series, name: str) -> pd.Series:
    cleaned = prices.astype(float).dropna()
    returns = np.log(cleaned / cleaned.shift(1)).dropna()
    return returns.rename(name)


if __name__ == "__main__":
    main()
