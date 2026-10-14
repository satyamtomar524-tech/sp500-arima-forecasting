# S&P 500 forecasting: can ARIMA beat a simple baseline?

An **AI-assisted academic/portfolio reconstruction** of a historical forecasting exercise, built to make the evaluation design inspectable. This is new work inspired by a short project note; the original notebook, dataset, code, and results were not available. It is not a claim of independent original coursework or employment.

## Question and method

Can a prespecified ARIMA(1,1,1) forecast the S&P 500 daily **price-index close** better than simply carrying forward the last observed close? The target is index points, not total return or investment profit.

- Historical source: Yahoo Finance `^GSPC` daily Close, 2000–2020. The local extract is checked for missing/nonpositive closes, duplicate or out-of-order dates, and coverage against the XNYS session calendar. The calendar is a completeness proxy, not independent verification of prices.
- Chronological split: train on 2000–2013; evaluate on 2014–2020. ARIMA(1,1,1) has no drift and is fitted on training data only. Its parameters and innovation variance remain frozen through the test.
- Two distinct protocols: one fixed-origin forecast for the entire test path, and rolling-origin forecasts for 1, 5 and 20 observed trading sessions. Each rolling forecast is made before the target close is revealed.
- Baselines: last observed close and a training-estimated random walk with drift. Metrics: MAE, RMSE and nominal 95% ARIMA interval coverage. Diagnostics are calculated on training data.

## Results from the local historical extract

| Evaluation | ARIMA RMSE | Last-close RMSE | ARIMA vs. last close |
|---|---:|---:|---:|
| Rolling, 1 session | 29.17 | 29.33 | 0.55% lower RMSE |
| Rolling, 5 sessions | 60.33 | 60.19 | 0.24% higher RMSE |
| Rolling, 20 sessions | 123.56 | 123.51 | 0.04% higher RMSE |

The small one-session advantage did **not** persist at longer horizons. One-session nominal 95% interval coverage was **83.8%** overall and **55.7%** in the descriptive 2020 slice, showing that these model-based intervals were poorly calibrated for those observations. The full aggregate metric table is in [`results/metrics.csv`](results/metrics.csv). These results describe one historical extract and protocol; they do not demonstrate a trading edge.

## Run it yourself

Use Python 3.12 and the pinned packages in [`requirements.txt`](requirements.txt):

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe download_data.py
& .\.venv\Scripts\python.exe analysis.py
```

The download is an explicit step, never automatic. It requests a fresh historical vintage; source availability and values can change, so a new run may not exactly reproduce the local results above. `download_data.py` refuses to overwrite an existing extract. For an existing, legally obtained CSV with `Date,Close` columns, run `analysis.py --data path\to\file.csv` instead. The script writes diagnostics, forecasts, and plots locally under ignored `results/` and `charts/` paths.

The notebook [`sp500_reconstruction.ipynb`](sp500_reconstruction.ipynb) contains the explanatory code and method. Its outputs and execution counts were cleared before publication because executed plots and tables included unlicensed market observations. Run the download and analysis script before executing the notebook in the same environment.

## Boundaries

Raw market data, detailed forecasts containing actual prices, plot arrays, and the rendered notebook are **not** included: redistribution rights for the data were not verified. The code and aggregate metrics are published for review. The calendar check does not validate each close against an independent provider; the historical extract is not verified as a contemporaneous point-in-time vintage. Model order was prespecified, not chosen after a test-set search. Overlapping 5/20-session errors, parameter uncertainty, and market regime changes limit inference.

AI helped scope the reconstruction, generate and review code and prose, and organise reproducibility checks. The numerical forecasts were calculated by the fitted statistical model. See [`AI_CONTRIBUTIONS.md`](AI_CONTRIBUTIONS.md) for the disclosure. Any academic reuse should follow the applicable course policy.

## Method references

- [Yahoo Finance S&P 500 history](https://finance.yahoo.com/quote/%5EGSPC/history/) — historical data source; data not redistributed here.
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) and [state-space forecasting](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html) — model and forecast protocol.
- [Statsmodels ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) — unit-root diagnostic.
- [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars) — XNYS session proxy.
