"""Price data loaders for the stock range model project."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .models import TRADING_DAYS_PER_YEAR


def load_csv_prices(path: str | Path, price_column: str | None = None) -> pd.Series:
    """Load prices from a CSV file.

    The CSV should contain a date column and a close-like price column. Common
    names such as Date, Adj Close, Close, price, and close are detected.
    """

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("CSV file is empty")

    lower_columns = {column.lower().strip(): column for column in frame.columns}
    date_column = (
        lower_columns.get("date")
        or lower_columns.get("datetime")
        or lower_columns.get("time")
        or frame.columns[0]
    )

    if price_column is None:
        for candidate in ["adj close", "adjusted close", "close", "price"]:
            if candidate in lower_columns:
                price_column = lower_columns[candidate]
                break
    if price_column is None:
        raise ValueError("could not find a price column; pass price_column explicitly")

    dates = pd.to_datetime(frame[date_column])
    prices = pd.Series(frame[price_column].astype(float).to_numpy(), index=dates, name="price")
    return prices.sort_index().dropna()


def download_yahoo_prices(
    ticker: str,
    start: str | date,
    end: str | date | None = None,
) -> pd.Series:
    """Download daily adjusted close prices from Yahoo Finance chart endpoint."""

    start_date = _parse_date(start)
    end_date = _parse_date(end) if end is not None else date.today()
    if end_date <= start_date:
        raise ValueError("end must be after start")

    period1 = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    # Yahoo's period2 is exclusive, so add one day to include the requested end.
    period2 = int(datetime.combine(end_date + timedelta(days=1), datetime.min.time()).timestamp())
    query = urlencode(
        {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{query}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"failed to download Yahoo Finance data for {ticker}: {exc}") from exc

    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"Yahoo Finance returned no usable data for {ticker}")

    chart = result[0]
    timestamps = chart.get("timestamp", [])
    indicators = chart.get("indicators", {})
    adjclose = indicators.get("adjclose", [{}])[0].get("adjclose")
    closes = indicators.get("quote", [{}])[0].get("close")
    values = adjclose or closes
    if not timestamps or not values:
        raise RuntimeError(f"Yahoo Finance returned no prices for {ticker}")

    dates = pd.to_datetime(timestamps, unit="s").normalize()
    prices = pd.Series(values, index=dates, name=ticker, dtype="float64")
    return prices.sort_index().dropna()


def download_yahoo_dividends(
    ticker: str,
    start: str | date,
    end: str | date | None = None,
) -> pd.Series:
    """Download per-share dividend events from Yahoo Finance chart endpoint."""

    start_date = _parse_date(start)
    end_date = _parse_date(end) if end is not None else date.today()
    if end_date <= start_date:
        raise ValueError("end must be after start")

    period1 = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    period2 = int(datetime.combine(end_date + timedelta(days=1), datetime.min.time()).timestamp())
    query = urlencode(
        {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history,div",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{query}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"failed to download Yahoo Finance dividends for {ticker}: {exc}") from exc

    result = payload.get("chart", {}).get("result")
    if not result:
        return pd.Series(dtype="float64", name=f"{ticker}_dividends")

    events = result[0].get("events", {})
    dividends = events.get("dividends", {})
    if not dividends:
        return pd.Series(dtype="float64", name=f"{ticker}_dividends")

    rows = [
        (pd.to_datetime(item["date"], unit="s").normalize(), float(item["amount"]))
        for item in dividends.values()
        if "date" in item and "amount" in item
    ]
    if not rows:
        return pd.Series(dtype="float64", name=f"{ticker}_dividends")

    dates, amounts = zip(*rows)
    series = pd.Series(amounts, index=pd.DatetimeIndex(dates), name=f"{ticker}_dividends")
    return series.sort_index()


def generate_synthetic_prices(
    years: int = 12,
    start_price: float = 100.0,
    annual_return: float = 0.08,
    annual_volatility: float = 0.25,
    seed: int = 42,
) -> pd.Series:
    """Generate sample adjusted-close-like prices for demos and tests."""

    if years < 2:
        raise ValueError("years must be at least 2")
    if start_price <= 0:
        raise ValueError("start_price must be positive")

    rng = np.random.default_rng(seed)
    days = years * TRADING_DAYS_PER_YEAR
    daily_mu = annual_return / TRADING_DAYS_PER_YEAR
    daily_sigma = annual_volatility / np.sqrt(TRADING_DAYS_PER_YEAR)
    returns = rng.normal(loc=daily_mu, scale=daily_sigma, size=days)
    prices = start_price * np.exp(np.cumsum(returns))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.Series(prices, index=dates, name="synthetic")


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()
