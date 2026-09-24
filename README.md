# SAGE-Price Model

**SAGE-Price Model**은 제가 개발한 교육·연구용 프로젝트입니다. 여러 금융 모형이 산출한 기대주가를 과거 예측 오차에 따라 결합해 약 1년 뒤의 기대주가와 참고범위를 제시합니다. SAGE는 *Score-weighted Asset-pricing Guided Ensemble*의 약자입니다.

## 무엇을 구현했나요?

1. Yahoo Finance의 조정종가, 사용자가 준비한 CSV 또는 합성 예시 데이터를 불러옵니다.
2. 아래 다섯 모형으로 각 시점의 기대주가를 계산합니다.
3. 과거 시점으로 돌아가 약 1년 뒤 주가를 예측하고, 실제 주가와의 오차를 비교합니다.
4. 모형별 과거 예측 오차를 점수와 가중치로 바꿔 최종 기대주가를 계산합니다.
5. 과거 앙상블 오차를 이용해 참고용 하단·상단 가격을 제시하고 상세 결과를 CSV로 저장합니다.

| 모형 | 적용 관점 |
| --- | --- |
| CAPM | 시장위험을 반영한 기대수익률 |
| Fama-French 3요인 스타일 모형 | 시장·규모·가치 요인의 대리지표를 활용한 기대수익률 |
| APT/거시경제 요인 모형 | 시장·금리·환율 대리지표를 활용한 기대수익률 |
| 배당할인모형(DDM/Gordon Growth) | 배당을 바탕으로 한 가치평가 |
| 잔여이익모형(Residual Income) | 장부가치와 초과이익을 바탕으로 한 가치평가 |

모형 선택 이유와 계산 흐름은 [PROJECT_SPEC.md](PROJECT_SPEC.md)에 정리했습니다.

## 빠르게 실행하기

저장소의 최상위 폴더에서 먼저 필요한 패키지를 설치합니다.

```bash
python -m pip install -r requirements.txt
```

인터넷 연결이나 실제 시세 데이터 없이 합성 데이터로 실행할 수 있습니다.

```bash
python -m stock_range_model.cli --synthetic
```

Windows에서 `python` 명령이 Microsoft Store 별칭으로 연결된다면 명령의 `python`을 `py`로 바꿔 실행하세요.

실제 종목의 Yahoo Finance 데이터를 조회하려면 종목 코드를 지정합니다. 예시는 미국 주식과 한국 주식 코드입니다.

```bash
python -m stock_range_model.cli --ticker AAPL --years 10
python -m stock_range_model.cli --ticker 005930.KS --years 10
```

직접 준비한 CSV도 사용할 수 있습니다. 날짜 열은 `Date`, 가격 열은 `Adj Close`, `Close`, `price` 중 하나를 사용합니다.

```bash
python -m stock_range_model.cli --csv prices.csv
```

가격 열 이름이 다르면 `--price-column`으로 지정합니다.

```bash
python -m stock_range_model.cli --csv prices.csv --price-column ClosePrice
```

## 노트북으로 살펴보기

Jupyter용 패키지를 설치하고 노트북을 실행합니다.

```bash
python -m pip install -r requirements-jupyter.txt
python -m notebook
```

브라우저에서 `notebooks/SAGE-Price_Model.ipynb`를 열고 셀을 위에서 아래로 실행하세요. 노트북의 기본 종목 코드는 `005930.KS`이며, 코드의 `TICKER`와 `USE_SYNTHETIC` 값을 바꿔 다른 데이터를 사용할 수 있습니다.

## 결과 파일과 테스트

명령행 프로그램은 `outputs/`에 다음 파일을 저장합니다.

- `forecast_summary.csv`: 현재 주가, 최종 기대주가, 기대수익률, 참고범위
- `model_weights.csv`: 모형별 과거 오차와 최종 가중치
- `model_predictions.csv`: 모형별 기대주가와 가중 결과
- `backtest_rows.csv`: 모형별 과거 예측과 실제값
- `ensemble_backtest_rows.csv`: 앙상블의 과거 오차 계산 결과

기존 테스트는 다음 명령으로 실행할 수 있습니다.

```bash
python -m unittest discover -s tests -v
```

## 데이터와 결과를 해석할 때

- 일부 시장·규모·가치·거시경제 요인은 **대리지표(proxy)**입니다. 배당이나 장부가치 자료가 없을 때는 기본 가정값을 사용하므로 실제 기업가치 추정과 차이가 날 수 있습니다.
- 현재 앙상블의 과거 오차 계산은 **전체 과거 검증 결과로 정한 모형 가중치를 같은 기간에 다시 적용**합니다. 이 수치를 독립적인 외표본 성능으로 해석할 수 없습니다.
- 하단·상단 가격은 **과거 예측 오차 분위수에 따른 참고범위**입니다. `0.80` 설정을 사용하더라도 통계적으로 보정·검증된 80% 예측구간은 아닙니다.
- 데이터 제공 상태와 입력 가정에 따라 결과가 달라집니다. 이 코드는 투자 판단이나 실거래를 위한 권고가 아닙니다.
