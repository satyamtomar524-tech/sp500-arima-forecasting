# S&P 500 Time-Series Forecasting with ARIMA

This repository contains a **later, AI-assisted Python reconstruction** of an academic group project on forecasting the S&P 500. Its purpose is simple: make the forecasting question, evaluation method, baselines, results, and limitations understandable and reproducible.

## Project history

A private group presentation supplied on 25 October 2026 names **Satyam Tomar as one of three group members**. It describes an S&P 500 study covering 2000–2020, with ARIMA(1,1,1), stationarity testing, differencing, and an evaluation period from 2014–2020. The presentation reports test RMSE **712.46374** and MAE **581.8584**.

Those figures are **source-reported college results**. They have not been independently reproduced because the original notebook and code, exact dataset, evaluation protocol, division of individual work, submission, and grade were not recovered. The presentation remains private.

This public repository is a separate reconstruction created on 14 October 2026. It adds an explicit chronological split, simple baselines, fixed-origin and rolling-origin backtests, interval-coverage checks, and reproducibility safeguards. Its executed results must not be presented as the original group results.

## Research question

Can a prespecified ARIMA(1,1,1) forecast the daily S&P 500 **price-index close** more accurately than simply carrying forward the last observed close?

The target is measured in index points. This is a forecasting comparison, not a trading strategy, total-return analysis, or investment recommendation.

## Reconstruction design

- **Reference data:** 5,284 daily `^GSPC` closing observations from 2000–2020.
- **Training period:** 2000–2013, containing 3,521 observations.
- **Test period:** 2014–2020, containing 1,763 observations.
- **Model:** ARIMA(1,1,1), no drift, fitted only on the training period.
- **Evaluation:** one fixed-origin forecast and rolling-origin forecasts at 1, 5, and 20 observed trading sessions.
- **Baselines:** last observed close and a random walk with drift estimated from the training period.
- **Metrics:** MAE, RMSE, and nominal 95% ARIMA interval coverage.

During the test period, the fitted ARIMA coefficients and innovation variance remain frozen. Each rolling forecast is produced before the target close is revealed.

## Reconstruction results

| Evaluation | ARIMA RMSE | Last-close RMSE | Result versus last close |
|---|---:|---:|---:|
| Fixed origin, full test path | 799.62 | 798.29 | 0.17% higher RMSE |
| Rolling, 1 session | 29.17 | 29.33 | 0.55% lower RMSE |
| Rolling, 5 sessions | 60.33 | 60.19 | 0.24% higher RMSE |
| Rolling, 20 sessions | 123.56 | 123.51 | 0.04% higher RMSE |

Fixed-origin and rolling-origin rows answer different forecasting questions and use different information sets, so their absolute RMSE values should not be compared directly across rows.

The reconstruction's fixed-origin ARIMA MAE is **642.60**. This is separate from the presentation-reported MAE of **581.8584** because the underlying data and evaluation protocol are not established as identical.

The model achieved a very small improvement for the next session, but that advantage did not continue at 5 or 20 sessions. Its nominal 95% one-session intervals covered **83.8%** of test observations and only **55.7%** in the descriptive 2020 slice, indicating that the model-based intervals were too narrow for many realised observations, especially in 2020. The fixed-origin forecast also failed to follow the sustained market rise across the full test period.

The honest conclusion is that ARIMA is a useful statistical baseline here, but this experiment does **not** establish statistical superiority, a trading edge, or reliable long-horizon market prediction. Full aggregate results are available in [`results/metrics.csv`](results/metrics.csv).

## Repository guide

- [`analysis.py`](analysis.py) performs validation, diagnostics, forecasting, backtesting, and metric calculation.
- [`download_data.py`](download_data.py) downloads the historical series only when run explicitly.
- [`sp500_reconstruction.ipynb`](sp500_reconstruction.ipynb) explains the analysis step by step; published outputs are cleared.
- [`results/metrics.csv`](results/metrics.csv) contains the reviewed aggregate metrics from the historical reference run.
- [`AI_CONTRIBUTIONS.md`](AI_CONTRIBUTIONS.md) records how AI supported the later reconstruction.

## Run it locally

Use Python 3.12 and the pinned packages in [`requirements.txt`](requirements.txt):

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe download_data.py
& .\.venv\Scripts\python.exe analysis.py
```

The download is never automatic, and `download_data.py` refuses to overwrite an existing extract. A fresh download may differ from the 14 October 2026 reference vintage. You can instead supply a legally obtained CSV with `Date,Close` columns:

```powershell
& .\.venv\Scripts\python.exe analysis.py --data path\to\file.csv
```

Run the analysis in your own clone or working copy. It refreshes local forecast files and charts, overwrites the tracked aggregate `results/metrics.csv`, and rewrites `requirements.txt` with the versions in that environment.

## Boundaries

Raw historical closes, detailed forecasts containing actual prices, plot arrays, and executed notebook outputs are not published because redistribution rights were not verified. The calendar audit checks session coverage but does not independently verify every closing value. The historical extract is also not verified as a contemporaneous point-in-time vintage.

The ARIMA order was prespecified rather than selected through a test-period search. Overlapping multi-session forecast errors, parameter uncertainty, changing market regimes, and time-varying volatility limit inference. ARIMAX, GARCH, machine-learning, LLM forecasting, and trading simulations are possible future extensions only; none was implemented in this repository.

AI helped scope, code, review, explain, and verify the later reconstruction. Python, Pandas, and Statsmodels performed the calculations. Any academic reuse should follow the applicable course policy.

## Method references

- [Yahoo Finance S&P 500 history](https://finance.yahoo.com/quote/%5EGSPC/history/) — historical data source; data is not redistributed here.
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) and [state-space forecasting](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html) — model and forecast protocol.
- [Statsmodels ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) — unit-root diagnostic.
- [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars) — XNYS trading-session proxy.
