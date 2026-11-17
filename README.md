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

| Evaluation | ARIMA RMSE | Last-close RMSE | Drift RMSE |
|---|---:|---:|---:|
| Fixed origin, full test path | 799.62 | 798.29 | **687.60** |
| Rolling, 1 session | **29.17** | 29.33 | 29.33 |
| Rolling, 5 sessions | 60.33 | 60.19 | **60.14** |
| Rolling, 20 sessions | 123.56 | 123.51 | **123.14** |

Fixed-origin and rolling-origin rows answer different forecasting questions and use different information sets, so their absolute RMSE values should not be compared directly across rows.

The reconstruction's fixed-origin ARIMA MAE is **642.60**. This is separate from the presentation-reported MAE of **581.8584** because the underlying data and evaluation protocol are not established as identical.

The model achieved a very small improvement for the next session: last-close RMSE minus ARIMA RMSE was **0.1604 index points** in this historical sample, about **0.55%** of last-close RMSE. That is an aggregate difference, not an improvement of 0.1604 points on every forecast. It did not continue at 5 or 20 sessions. The drift baseline did substantially better on the fixed-origin path; that result is specific to this historical path and does not prove that drift is always preferable. The no-drift ARIMA specification helps explain why its long-path forecast flattened.

Its nominal 95% one-session intervals covered **83.8%** of test observations and only **55.7%** in the descriptive 2020 slice, indicating poor calibration for these observations, especially in 2020. This alone does not identify the cause. New runs also report interval width and Winkler score, which penalises both wide intervals and misses.

The honest conclusion is that ARIMA is a useful statistical baseline here, but this experiment does **not** establish statistical superiority, a trading edge, or reliable long-horizon market prediction. Full aggregate results are available in [`results/metrics.csv`](results/metrics.csv).

## Repository guide

- [`analysis.py`](analysis.py) performs validation, diagnostics, forecasting, backtesting, and metric calculation.
- [`evaluation.py`](evaluation.py) builds and checks a row-level forecast ledger, calculates interval quality, and runs an exploratory paired block bootstrap for the one-session RMSE difference.
- [`refit_experiment.py`](refit_experiment.py) is a separate, retrospective comparison with the same ARIMA order refitted every 20 observed sessions.
- [`download_data.py`](download_data.py) downloads the historical series only when run explicitly.
- [`sp500_reconstruction.ipynb`](sp500_reconstruction.ipynb) explains the analysis step by step; published outputs are cleared.
- [`results/metrics.csv`](results/metrics.csv) contains the reviewed aggregate metrics from the historical reference run.
- [`LEARNING_START_HERE.md`](LEARNING_START_HERE.md) starts with one small calculation you can do yourself.
- [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) explains the complete project and November audit chapter by chapter in plain language. A locally generated PDF version is kept outside Git.
- [`AI_CONTRIBUTIONS.md`](AI_CONTRIBUTIONS.md) records how AI supported the later reconstruction.

## What a new run verifies

Each run saves a **private, ignored** `runs/<id>/results/forecast_ledger.csv`. It has one row per model and forecasting task: origin date, target date, horizon, actual close, forecast, interval where applicable, the last date used to fit parameters, and the last observation available at the origin. The audit checks that all three models share each task, that the target is the stated number of observed sessions ahead, and that last-close and drift forecasts use only information available at the origin. A separate calculation recomputes the one-session MAE and RMSE from those ledger rows.

New runs also save the exact data hash, source-code hashes, settings, package versions, diagnostics, warnings, charts, and metrics. ADF output records the constant, default maximum-lag candidate, AIC selection, chosen lag and effective sample size. Ljung–Box output records `model_df=2` and the resulting test degrees of freedom (`8` at lag 10 and `18` at lag 20). The lag-20 result tests autocorrelation jointly through that lag.

For the primary next-session comparison, the code reports the observed RMSE difference and **exploratory** 95% percentile intervals from paired circular-block resampling with block lengths 5, 20 and 60 sessions. These checks retain the pairing and some time dependence. Their assumptions and sensitivity matter, and the 2014–2020 outcomes were already known when this audit was designed. The intervals are not proof that ARIMA will beat last close in a new period. Five- and twenty-session comparisons remain descriptive because their forecast errors overlap.

In the verified local run, the 20-session-block interval for **last-close RMSE minus frozen-ARIMA RMSE** was **-0.1205 to +0.4713 index points**. The intervals for 5- and 60-session blocks also included zero. This supports a cautious reading of the observed 0.1604-point advantage. The frozen model's one-session interval coverage was **83.83%**, mean width **57.64 points**, and mean Winkler score **227.90 points**; in the descriptive 2020 slice, coverage was **55.73%** and mean Winkler score **835.83 points**.

## Separate retrospective refit experiment

After the alignment and safe-output audit, we compared the frozen model with the **same ARIMA(1,1,1) specification**, refitted on an expanding sample after every 20 newly observed test closes. Each forecast is issued before its target close is revealed. The two models, last close, and drift are scored on identical rows. All 88 scheduled refits converged in the local run, with no recorded fit warnings.

| Forecast horizon | Last close RMSE | Frozen ARIMA RMSE | Refit ARIMA RMSE |
|---|---:|---:|---:|
| Next session | 29.3321 | 29.1717 | **29.0386** |
| 5 sessions | **60.1906** | 60.3333 | 60.4273 |
| 20 sessions | **123.5064** | 123.5569 | 123.6451 |

The refit model's next-session nominal 95% interval coverage was **85.93%** versus **83.83%** for frozen ARIMA; both were below 95%. Its mean width was **61.67** versus **57.64** points. Annual next-session results are mixed; the 2020 RMSE was **59.96** for refit versus **60.39** for frozen. This is an **exploratory historical comparison**, since the 2014–2020 outcomes were already known when the extension was designed. It does not provide an untouched new-period assessment or establish superiority. Detailed local rows and annual results remain ignored by Git.

## Run it locally

Use the pinned packages in [`requirements.txt`](requirements.txt). Python 3.13 is the current local runtime used for this update:

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

Each call to `analysis.py` creates a new directory under `runs/`. It **refuses** an existing `--output-dir`. It does not change the tracked `results/metrics.csv` or `requirements.txt`. Open the printed run directory to inspect `run_config.json`, `results/forecast_ledger.csv`, `results/one_step_block_bootstrap.csv`, `results/metrics.csv`, `results/summary.json`, and `charts/`. The historical reference and a fresh data vintage should not be described as the same executed run.

To reproduce the separate refit experiment, first run `analysis.py` and then pass its printed run folder to:

```powershell
& .\.venv\Scripts\python.exe refit_experiment.py --data path\to\file.csv --reference-run runs\your-reference-run-id
```

The extension creates another ignored run folder with its own configuration, aligned forecasts, metrics, annual breakdown, and fit-event record. A future evaluation on later data would need a separately reserved period and the same predeclared comparison rules.

For code checks, install the separate [`requirements-dev.txt`](requirements-dev.txt) into this project's environment, then run `python -m unittest discover -s tests` and `python -m ruff check analysis.py evaluation.py download_data.py tests` from the repository root.

## Boundaries

Raw historical closes, detailed forecasts containing actual prices, plot arrays, and executed notebook outputs are not published because redistribution rights were not verified. The calendar audit checks session coverage but does not independently verify every closing value. The historical extract is also not verified as a contemporaneous point-in-time vintage.

The ARIMA order was prespecified rather than selected through a test-period search. Overlapping multi-session forecast errors, parameter uncertainty, changing market regimes, and time-varying volatility limit inference. ARIMAX, GARCH, machine-learning, LLM forecasting, and trading simulations are possible future extensions only; none was implemented in this repository.

AI helped scope, code, review, explain, and verify the later reconstruction. Python, Pandas, and Statsmodels performed the calculations. Any academic reuse should follow the applicable course policy.

## Method references

- [Yahoo Finance S&P 500 history](https://finance.yahoo.com/quote/%5EGSPC/history/) — historical data source; data is not redistributed here.
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) and [state-space forecasting](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html) — model and forecast protocol.
- [Statsmodels ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) — unit-root diagnostic.
- [Statsmodels Ljung–Box](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html) — residual test settings and degrees of freedom.
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/toolbox.html) — baselines, time-series evaluation, and Winkler interval scoring.
- [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars) — XNYS trading-session proxy.
