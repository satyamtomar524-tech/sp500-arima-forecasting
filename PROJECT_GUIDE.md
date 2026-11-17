# S&P 500 ARIMA project: a beginner's chapter guide

**Updated 17 November 2026.** This guide describes a later, AI-assisted reconstruction and its exploratory extension. It is not the original college submission. All accuracy numbers below are historical backtest results in S&P 500 index points, not a live forecast or investment return.

## Chapter 1 - What this project is, and why we did it

The question is: **Can an ARIMA(1,1,1) model predict the next daily S&P 500 price-index close more accurately than simply using the last known close?** We also ask how that answer changes when we forecast 5 or 20 observed trading sessions ahead. This turns the vague idea of "predicting the market" into a testable comparison. A useful result can be that the simple forecast is just as good.

The earlier academic work was a **three-person group presentation**. It reported RMSE **712.46374** and MAE **581.8584**. The original data file, code, exact forecast protocol, division of work, submission and grade have not been recovered, so those numbers are source-reported college results. The Python results in this guide come from a **separate reconstruction**, first made in October 2026 and improved in November 2026. Do not combine the two sets of results or claim the later code was the original submission.

## Chapter 2 - What we are trying to predict

The target is the daily `^GSPC` **Close**, the level of the S&P 500 price index, measured in **index points**. A close near 3,000 means an index level, not EUR 3,000 of profit. It does not include reinvested dividends. Price levels, daily point changes, percentage returns and an investor's total return are different quantities. This project scores forecasts of the **level**; changing to returns or log prices would be a new experiment and would need metrics on a comparable scale.

The local historical extract has **5,284** daily observations from **3 January 2000 to 31 December 2020**. It came from Yahoo Finance. Its exact CSV hash is recorded locally. The raw observations are kept out of the public repository because open redistribution rights were not verified. The calendar check confirms that the dates match an XNYS trading-session proxy; it does not independently verify every index value or prove that the current historical extract is the same vintage available on each old date.

## Chapter 3 - How the data was checked and split

The code checks the two expected columns (`Date,Close`), dates in increasing order, duplicate dates, missing and nonpositive closes, and finite numbers. It compares the dates with exchange sessions. It does **not** fill weekends or holidays by interpolation. It then divides the data by time: **3,521 training observations through 31 December 2013**, followed by **1,763 evaluation observations from 2 January 2014 through 31 December 2020**. A random split would allow later market observations to influence a model evaluated on earlier dates, so it would answer the wrong forecasting question.

**Try it:** If a forecast is made after Monday's close, can it use Tuesday's close? No. Tuesday's close is known only after Tuesday's session ends. That simple timing rule controls every baseline, model update and test row in this project.

## Chapter 4 - Forecast error, MAE and RMSE

For one prediction, **error = actual close - forecast close**. A positive error means the forecast was too low. A negative error means it was too high. If the actual close is 105 and the forecast is 108, the error is -3: the forecast was 3 points too high.

**MAE** is the average absolute error. It tells us the typical mistake without letting positive and negative mistakes cancel. **RMSE** is the square root of the average squared error. Squaring makes large misses count more. Both are measured here in index points, and **lower is better**. A visually close forecast line is not enough: we must calculate errors on observations that were genuinely unseen at each forecast origin.

**Your first learning checkpoint:** The actual close is 100 and the forecast is 98. Was the forecast too high or too low, and by how many points? Write your answer before opening the code.

## Chapter 5 - Levels, differences and the ADF test

The index **level** may rise for years. The **first difference** is today's close minus the previous session's close. ARIMA(1,1,1) uses one difference (`d=1`) so its short-run model works with point changes. Differencing does not manufacture predictability or guarantee stable volatility.

The ADF test asks whether a unit root is compatible with the observed series. In this run, both training-level and training-difference tests used a **constant** in the test regression, **AIC** to select lag length, and the Statsmodels default maximum-lag rule. The default candidate maximum was **30 lags** for both series. AIC selected **22** for the levels and **21** for first differences; each regression used **3,498 effective observations**. The level test did not reject its unit-root null at 5%; the first-difference test did. This supports the modelling choice, but it does not prove that the market is easy to predict or that its volatility is constant.

## Chapter 6 - What ARIMA(1,1,1) does

Here `p=1` means the model can use the previous point change; `d=1` means it works with first differences; and `q=1` means it can use the previous one-step surprise, called an innovation. The reconstruction deliberately uses **no drift**. Statsmodels estimates the coefficients and innovation variance from the training data by state-space maximum likelihood. The order was inherited from the project note; it was **not** selected by searching the test years.

A simplified next-change calculation is: `predicted next change = phi * latest change + theta * latest innovation`. Then `predicted next close = known close + predicted next change`. For teaching only, let the current close be 101, the previous close 103, the latest innovation +1, `phi=0.4845`, and `theta=-0.5684`. The latest change is -2. The predicted next change is `0.4845*(-2) + (-0.5684)*(1) = -1.5374`, giving a predicted close of **99.4626**. The last-close baseline predicts **101**. These are invented example values, not a row from the dataset.

For a long forecast without drift, expected future changes tend to fade; the level path can flatten. That helps explain the reconstruction's nearly flat fixed-origin ARIMA line. It does **not** mean that every possible ARIMA specification must be flat.

## Chapter 7 - Two forecast protocols and three original comparators

**Fixed origin:** stand at 31 December 2013 and forecast the entire 1,763-session test path without seeing any 2014-2020 closes. Its overall RMSE pools many different forecast horizons. **Rolling origin:** after each actual close becomes available, update the model's filtered state and issue a new forecast. The original reconstruction keeps its fitted coefficients and innovation variance **frozen at the end of 2013**. It evaluates 1, 5 and 20 observed-session horizons.

The first baseline is **last close**: carry the close known at the forecast origin forward. The second is **drift**: add `horizon * average training-period point change` to that origin close. The third is the frozen **ARIMA(1,1,1)**. The drift average is estimated only through 2013. A five-session baseline issued on Monday must use **Monday's** close, not the close just before the eventual target date.

## Chapter 8 - How we checked that the comparison is fair

Every new run writes a **forecast ledger**. A ledger row states the origin date, target date, horizon, model, forecast, actual close, interval when applicable, last parameter-fit date, and last observation available to the model state. The code checks that all three methods face exactly the same task and actual value, that the target is the stated number of observed sessions ahead, and that the baselines use the correct origin close. It also recomputes one-session MAE and RMSE from the ledger separately from the main metric calculation.

There are **1,763** one-session tasks, **1,759** five-session tasks and **1,744** twenty-session tasks. Longer horizons have fewer target dates near the end of 2020. These counts are expected; the key is that all methods use matching rows **within** each horizon. The five- and twenty-session forecast errors overlap, which matters when discussing uncertainty.

## Chapter 9 - What the preserved reconstruction found

| Evaluation | Last-close RMSE | Drift RMSE | Frozen ARIMA RMSE |
|---|---:|---:|---:|
| Fixed origin, whole test path | 798.29 | 687.60 | 799.62 |
| Rolling, 1 session | 29.33 | 29.33 | 29.17 |
| Rolling, 5 sessions | 60.19 | 60.14 | 60.33 |
| Rolling, 20 sessions | 123.51 | 123.14 | 123.56 |

The frozen ARIMA's one-session RMSE was **29.1717** versus **29.3321** for last close. The aggregate gap was **0.1604 index points**, or about **0.55%** of last-close RMSE. This does not say that every individual forecast improved. The advantage was absent at five and twenty sessions. The drift baseline did much better along the historical fixed-origin path. Fixed-origin and rolling-origin RMSE values answer different questions, so do not compare 799.62 directly with 29.17 as if one were a better score for the same task. The new November run reproduced **every number in the tracked historical metrics file exactly** at stored precision.

## Chapter 10 - Prediction intervals: coverage, width and score

An ARIMA point forecast is one number. A nominal **95% prediction interval** gives a lower and upper value that the model says should contain a future actual close about 95% of the time under its assumptions. **Coverage** is the observed fraction inside. **Width** is upper minus lower. An interval that spans almost everything can have high coverage but give little useful precision. The **Winkler score** adds the width and a penalty when the actual falls outside; lower is better.

For the frozen model's one-session forecasts, coverage was **83.83%**, mean width **57.64 points**, and mean Winkler score **227.90 points**. In the descriptive 2020 slice, coverage was only **55.73%** and mean score **835.83**. The fixed training variance gave essentially the same mean one-session width in every year while realised market changes varied. These numbers show poor calibration in this sample, especially 2020. They do not by themselves prove which model assumption caused it. A later volatility model such as GARCH is a hypothesis to test, not a guaranteed fix for the central forecast.

## Chapter 11 - What the diagnostic tests say

The training residual Ljung-Box calculation used `model_df=2` for the one AR and one MA term. Its reference degrees of freedom are `test lag - 2`: **8 at lag 10** and **18 at lag 20**. The lag-10 p-value was about **0.18**; the lag-20 p-value was about **0.0000037**. A lag-20 result tests residual autocorrelations **jointly through lag 20**; it does not identify one unique effect at exactly lag 20. An ARCH diagnostic also found evidence of changing conditional variance in the training residuals. Diagnostic p-values are about model assumptions, not proof of an investment opportunity.

## Chapter 12 - How uncertain is the small RMSE gap?

The first question is the **effect size**: last-close RMSE minus frozen-ARIMA RMSE = **+0.1604 points** for the next-session task. Positive means the ARIMA score was lower in the historical sample. To see how sensitive that difference is to time dependence, the code resamples **paired** forecast-error rows in circular blocks, using **2,000** repetitions and block lengths of **5, 20 and 60 sessions**. The middle, 20-session choice gave an exploratory 95% percentile interval of **-0.1205 to +0.4713 points**. All three block-length intervals included zero.

That interval is not a universal p-value or proof of no difference. Its interpretation depends on the block method and the historical series. The test outcomes were already known when we designed this check, so it remains retrospective. The defensible conclusion is that the observed edge is **small and uncertain**.

## Chapter 13 - The separate 20-session refit experiment

The original frozen model answers what happens when we keep the end-2013 coefficients. A controlled challenger asks what happens if we keep the **same ARIMA(1,1,1) no-drift order** but **re-estimate it on an expanding sample after every 20 newly observed test closes**. The refit happens before forecasting the next target. No observation after a forecast origin can enter that forecast. Failed fits would stop the run rather than silently remove a task. The local experiment completed **88 scheduled refits** after the initial fit, with no fit warnings, and aligned **5,266** forecast tasks across the three horizons.

| Horizon | Last-close RMSE | Frozen ARIMA RMSE | Refit ARIMA RMSE |
|---|---:|---:|---:|
| 1 session | 29.3321 | 29.1717 | 29.0386 |
| 5 sessions | 60.1906 | 60.3333 | 60.4273 |
| 20 sessions | 123.5064 | 123.5569 | 123.6451 |

Refitting reduced the next-session RMSE by about **0.1331 points** relative to the frozen model in this sample, but worsened the 5- and 20-session RMSE. Its next-session 95% interval coverage rose from **83.83% to 85.93%**, still short of 95%, while its mean width grew from **57.64 to 61.67 points**. Annual results were mixed, although the refit model did better on RMSE in the descriptive 2020 slice (**59.96** versus **60.39**). Because the 2014-2020 outcomes were already known when this extension was designed, this is an **exploratory retrospective result**, not new untouched evidence of superiority.

## Chapter 14 - Reproducibility and safe reruns

The public repository contains source code, this guide, tests, and aggregate reference metrics. It does not contain the raw Yahoo extract or row-level actual-price files. The project pins its main numerical packages in `requirements.txt`; new runs **never rewrite it**. A run creates a unique ignored `runs/<id>/` folder with its configuration, input and code hashes, Python/package versions, diagnostics, forecast ledger, metrics, charts and warnings. If an explicit output folder already exists, the script refuses to replace it. The refit extension creates its own separate ignored run folder with aligned rows, an annual breakdown and fit-event dates.

Run locally from the repository folder after installing its pinned requirements and obtaining a permitted `Date,Close` CSV:

`python analysis.py --data path/to/file.csv`

Then pass the first printed run folder to:

`python refit_experiment.py --data path/to/file.csv --reference-run runs/your-run-id`

For code checks, run `python -m unittest discover -s tests` and `python -m ruff check analysis.py evaluation.py refit_experiment.py download_data.py tests` in the project environment. A fresh provider download may produce a different historical vintage, so compare its hash and never call it the October 2026 reference without checking.

## Chapter 15 - What remains uncertain, and how to present it

This is a historical forecasting exercise, not a trading strategy. It does not establish profit after costs, statistical superiority, or good forecasts in an unseen later period. The raw price observations were not checked against another provider, the old data vintage is not verified as point-in-time, and model-based intervals leave out parameter uncertainty. Multi-session forecast errors overlap. The 2020 analysis is a descriptive stress view, not a second untouched holdout. A later evaluation period should be reserved **before** inspecting the errors of another procedure. A future ARIMAX experiment would need genuinely available-at-origin external variables; a GARCH experiment would mainly address conditional variance. Neither is implemented here.

A one-minute defence is: "Our college group studied S&P 500 forecasting with ARIMA. Later, I requested an AI-assisted reconstruction so the method and its limits could be checked. The reconstruction used 2000-2013 for initial fitting and 2014-2020 for historical evaluation. The frozen ARIMA only narrowly beat last close for the next session, and its intervals were poorly calibrated. A separate refit experiment changed the updating policy but reused a known test period, so I describe it as exploratory. I can show the forecast ledger, code, and checks, while keeping the original group results separate."

AI assisted with reconstruction, coding, explanation and review. Python, Pandas and Statsmodels calculated the numerical results; no LLM forecasted the index. Satyam's later learning answers and code contributions should be recorded as he completes them, rather than assumed from the presence of the repository.

## Chapter 16 - How this connects to your master's subjects

These are **learning connections to module names you previously shared**, not a claim about an official ISM assessment rubric. **Introduction to Business Intelligence and Data Science** connects to turning a broad question into a target, data pipeline, model and evidence-backed conclusion. **Structure-detecting Methods with Test Theory** connects to plots, differences, ACF/PACF and the ADF test. **Structure Validating Procedures** connects to checking a claim on later observations and fair baselines. **Linear Algebra and Numerical Mathematics** connects to the model's lagged combinations and numerical parameter estimation. **Database Systems and Information Integration** connects to unique dates, ordered rows, missing-value rules and tracing a result to its source observation. **Software Engineering** connects to safe output folders, tests, versions and reruns. **Corporate Governance and Business Ethics** connects to correct attribution, AI disclosure and careful limits on what the evidence proves.

Your finance background helps you explain why a price-index close is different from a return or a trading result. Do not add SQL, Power BI, SAP or LLM forecasting to your project description unless you actually build and evaluate them. The strongest presentation is the one in which you can explain each step and defend the small, uncertain result honestly.

## Appendix - Short glossary and method sources

**Origin:** The date through which observations are known when a forecast is issued. **Target:** The future date being predicted. **Horizon:** Number of observed trading sessions from origin to target. **Baseline:** A simple forecast used for comparison. **Drift:** Average training-period point change projected forward. **Innovation:** A model's one-step surprise. **Coverage:** Share of actuals inside an interval. **Calibration:** Agreement between claimed interval probability and observed containment. **Refit:** Estimate parameters again using newly available observations. **State update:** Feed a new observation into the fitted model without changing its parameters.

Method sources: [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html); [Statsmodels forecasting and updating](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html); [Statsmodels ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html); [Statsmodels Ljung-Box](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html); [Forecasting: Principles and Practice](https://otexts.com/fpp3/toolbox.html); [Yahoo Finance historical series](https://finance.yahoo.com/quote/%5EGSPC/history/). Repository code and run configuration are the source for this reconstruction's exact settings and results.
