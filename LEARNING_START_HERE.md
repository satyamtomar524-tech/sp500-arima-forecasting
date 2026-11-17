# Learn this project from the beginning

This is a later, AI-assisted reconstruction of a three-person college project. The original group results and the reconstruction's results are different records. You do not need to understand all of `analysis.py` at once. Work through one idea, one example, and one action at a time.

## First idea: forecast error

A **forecast** is a number predicted before the actual close is known. The **error** is:

`actual close - forecast close`

Both values here are S&P 500 **index points**. A positive error means the forecast was too low; a negative error means it was too high.

**Worked example:** The forecast is 108 and the actual close is 105. The error is `105 - 108 = -3`, so the forecast was **3 points too high**.

**Your action:** The actual close is 100 and the forecast is 98. Was the forecast too high or too low, and by how many points? Write your answer before reading the next topic with your tutor.

## Where this goes

1. Forecast error: tell the direction and size of one mistake.
2. Last-close baseline: predict the next close using only the current close.
3. MAE and RMSE: combine several errors into two summary numbers.
4. Fair comparison: match the same origin, target and horizon for every model.
5. Differencing: change an index level into a session-to-session change.
6. ARIMA terms: calculate one simplified forecast with supplied coefficients.
7. Diagnostic tests: state the ADF and Ljung–Box questions accurately.
8. Prediction intervals: check coverage, width and missed observations.
9. Reproducibility: trace one metric to its forecast rows and input data.
10. Project defence: explain what the group project reported, what the later reconstruction found, what AI helped with, and what remains uncertain.

The repo's `results/metrics.csv` is the preserved historical reconstruction reference. New calculations go into ignored `runs/<id>/` folders. Keep your own answers separate from code-generated results so you can show what you understand.
