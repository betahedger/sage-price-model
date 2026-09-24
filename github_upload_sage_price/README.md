# SAGE-Price Model

## 프로젝트 범위와 해석

이 저장소는 팀으로 수행한 SAGE-Price 프로젝트의 교육·연구용 프로토타입을
정리한 자료입니다. 공동 작업 결과이며 개인의 단독 구현 성과를 뜻하지 않습니다.

- 일부 시장·규모·가치·거시경제 데이터는 대리지표(proxy)를 사용합니다. 배당과
  장부가치 자료가 없으면 가정값으로 계산하므로 실제 기업가치 추정과 차이가 납니다.
- 현재 앙상블 검증은 전체 과거 검증 결과로 정한 모형 가중치를 같은 기간에 다시
  적용합니다. 따라서 제시된 앙상블 오차는 독립적인 외표본 성능 추정치가 아닙니다.
- 하단·상단 가격은 과거 예측 오차의 분위수로 만든 **참고범위**입니다. 검증된
  80% 예측구간이나 미래 수익 보장으로 해석해서는 안 됩니다.

SAGE-Price Model stands for **Score-weighted Asset-pricing Guided Ensemble**.
This project backtests several asset-pricing-style models, converts their
historical expected-price accuracy into weights, and combines their forecasts
into a one-year expected stock price.

It is a research and education tool, not investment advice.

## Model set

The default model set now uses five traditional asset pricing and valuation
models:

- CAPM
- Fama-French Factor Model
- APT / Macro Factor Model
- Dividend Discount Model / Gordon Growth Model
- Residual Income Model

The detailed project specification and model-selection defense are in
`PROJECT_SPEC.md`.

## What the MVP does

- Loads adjusted close prices from Yahoo Finance, CSV, or synthetic demo data.
- Runs five confirmed models:
  - CAPM
  - APT / Macro Factor Model
  - Fama-French Factor Model
  - Dividend Discount Model / Gordon Growth Model
  - Residual Income Model
- Runs walk-forward backtests.
- Scores each model by historical expected-price error.
- Converts model scores into ensemble weights.
- Calculates the final expected price as a weighted average of model expected
  prices.
- Shows a reference range based on historical ensemble forecast error.
- Saves CSV diagnostics under `outputs/`.

## Quick start

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

Run with synthetic demo data:

```bash
python -m stock_range_model.cli --synthetic
```

On Windows, if `python` is a Microsoft Store alias, use:

```bash
py -m stock_range_model.cli --synthetic
```

## Jupyter notebook

This project also includes a notebook:

```text
notebooks/SAGE-Price_Model.ipynb
```

To use the notebook, install its dependencies as well:

```bash
python -m pip install -r requirements-jupyter.txt
```

Then start Jupyter:

```bash
py -m notebook
```

Open `notebooks/SAGE-Price_Model.ipynb` and run the cells from top to
bottom.

Run with a Yahoo Finance ticker:

```bash
python -m stock_range_model.cli --ticker AAPL --years 10
```

For Korean stocks, Yahoo usually uses suffixes such as:

```bash
python -m stock_range_model.cli --ticker 005930.KS --years 10
```

Run from a CSV file:

```bash
python -m stock_range_model.cli --csv prices.csv
```

Expected CSV columns are `Date` and one of `Adj Close`, `Close`, or `price`.
If your file has another price column name:

```bash
python -m stock_range_model.cli --csv prices.csv --price-column ClosePrice
```

## Outputs

The CLI writes:

- `outputs/forecast_summary.csv`
- `outputs/model_weights.csv`
- `outputs/model_predictions.csv`
- `outputs/backtest_rows.csv`
- `outputs/ensemble_backtest_rows.csv`

## Tests

```bash
python -m unittest
```

On Windows:

```bash
py -m unittest discover -s tests -v
```

## Data notes

For ticker runs, the CLI automatically downloads price, market, style-factor,
macro-factor, and dividend proxies from Yahoo Finance when available. Residual
Income can use explicit accounting assumptions via CLI options such as
`--book-value-per-share`, `--roe`, `--required-return`, and `--terminal-growth`.
When accounting data is not supplied, it uses conservative proxies so the model
can still be backtested.
