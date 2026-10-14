"""Reproducible historical S&P 500 forecast reconstruction, not original results.

Run from any working directory: python analysis.py
No interpolation, shuffled splits, fitted values labelled as test forecasts, or
test-period parameter fitting. All forecast horizons are observed trading rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import warnings
from pathlib import Path

import exchange_calendars as xcals
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parent
TRAIN_END = "2013-12-31"
TEST_START = "2014-01-01"
HORIZONS = (1, 5, 20)
COLORS = {"Actual": "#253247", "ARIMA(1,1,1)": "#126B98",
          "Naive": "#A06C22", "Drift": "#727A48"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_data(data_path):
    """Support supplied decimal-comma CSV and standard decimal-point downloads."""
    frame = pd.read_csv(data_path, dtype={"Date": str, "Close": str})
    assert list(frame.columns) == ["Date", "Close"], "Unexpected columns"
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="raise")
    frame["Close"] = pd.to_numeric(frame["Close"].str.replace(",", ".", regex=False), errors="raise")
    assert not frame.isna().any().any(), "Missing source observations"
    assert frame.Date.is_unique and frame.Date.is_monotonic_increasing
    assert np.isfinite(frame.Close).all() and frame.Close.gt(0).all()
    assert frame.Date.min() >= pd.Timestamp("2000-01-01")
    assert frame.Date.max() <= pd.Timestamp("2020-12-31")
    return frame.set_index("Date")


def audit_data(frame, data_path):
    # XNYS is an exchange-session proxy for a US equity index, not an independent
    # check of Yahoo values or the S&P index provider's record-level lineage.
    calendar = xcals.get_calendar("XNYS", start="2000-01-01", end="2020-12-31")
    expected = calendar.sessions.tz_localize(None)
    absent = expected.difference(frame.index)
    extra = frame.index.difference(expected)
    result = {
        "observations": len(frame), "first_date": str(frame.index[0].date()),
        "last_date": str(frame.index[-1].date()), "nulls": 0, "duplicates": 0,
        "nonpositive": 0, "interpolated_rows": 0,
        "calendar_reference": "exchange_calendars XNYS sessions; proxy, not value validation",
        "expected_sessions": len(expected),
        "missing_expected_sessions": [str(x.date()) for x in absent],
        "unexpected_sessions": [str(x.date()) for x in extra],
        "csv_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "independent_price_crosscheck": "Not performed",
        "point_in_time_vintage": "Not verified; current historical extract",
        "annual_observation_counts": {str(k): int(v) for k, v in frame.groupby(frame.index.year).size().items()},
    }
    assert not len(absent) and not len(extra), "Calendar discrepancy requires review"
    metadata_path = data_path.parent / "source_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if "csv_sha256" in metadata:
            assert result["csv_sha256"].lower() == metadata["csv_sha256"].lower()
        assert len(frame) == metadata["row_count"]
        result["metadata_integrity_checked"] = True
    return result


def fit_model(train):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ARIMA(train, order=(1, 1, 1), trend="n",
                      enforce_stationarity=True, enforce_invertibility=True)
        fit = model.fit(method="statespace", method_kwargs={"maxiter": 1000})
    converged = bool(fit.mle_retvals.get("converged", False))
    assert converged, "MLE did not converge; do not publish forecast metrics"
    return fit, [str(item.message) for item in caught]


def diagnostics(train, fit, fit_warnings):
    rows = []
    for label, data in [("Training close", train), ("Training first difference", np.diff(train))]:
        out = adfuller(data, regression="c", autolag="AIC")
        rows.append({"series": label, "statistic": float(out[0]), "p_value": float(out[1]),
                     "lags": int(out[2]), "nobs": int(out[3]),
                     "critical_5pct": float(out[4]["5%"]),
                     "null": "Unit root under regression with constant",
                     "reject_at_5pct": bool(out[1] < .05)})
    # Exclude the diffuse initial observation; burn-in can have a large residual.
    burn = max(1, int(fit.loglikelihood_burn))
    residual = np.asarray(fit.resid)[burn:]
    lb = acorr_ljungbox(residual, lags=[10, 20], model_df=2, return_df=True)
    arch = het_arch(residual, nlags=10, ddof=2)
    return {
        "adf": rows, "adf_settings": "constant; AIC lag selection; default maximum lag",
        "parameters": {str(k): float(v) for k, v in zip(fit.param_names, fit.params)},
        "aic": float(fit.aic), "bic": float(fit.bic), "log_likelihood": float(fit.llf),
        "optimizer_converged": bool(fit.mle_retvals["converged"]),
        "optimizer_iterations": int(fit.mle_retvals.get("iterations", -1)),
        "warnings": fit_warnings,
        "residual_burn_excluded": burn,
        "ljung_box": [{"lag": int(i), "statistic": float(r.lb_stat), "p_value": float(r.lb_pvalue)}
                      for i, r in lb.iterrows()],
        "arch_lm": {"lags": 10, "statistic": float(arch[0]), "p_value": float(arch[1])},
    }, residual


def make_forecasts(frame, fit):
    train_mask = frame.index <= TRAIN_END
    n_train = int(train_mask.sum())
    dates = frame.index
    values = frame.Close.to_numpy()
    train = values[:n_train]
    test = values[n_train:]
    assert dates[n_train - 1] < pd.Timestamp(TEST_START) <= dates[n_train]
    assert dates[n_train - 1] == pd.Timestamp(TRAIN_END)
    drift = float(np.diff(train).mean())
    fixed_forecast = fit.get_forecast(steps=len(test))
    intervals = np.asarray(fixed_forecast.conf_int(alpha=.05))
    fixed = pd.DataFrame({
        "Date": dates[n_train:], "Origin": dates[n_train - 1],
        "Horizon": np.arange(1, len(test) + 1), "Actual": test,
        "ARIMA(1,1,1)": np.asarray(fixed_forecast.predicted_mean),
        "Lower95": intervals[:, 0], "Upper95": intervals[:, 1],
        "Naive": train[-1], "Drift": train[-1] + drift * np.arange(1, len(test) + 1),
    })
    rows = []
    current = fit
    # Freeze model coefficients and innovation variance at the end of 2013.
    # Forecast FIRST, then reveal this row's actual close to the state filter.
    # No parameter re-estimation and no test-set order selection occurs.
    for target_index in range(n_train, len(values)):
        available = min(max(HORIZONS), len(values) - target_index)
        forecast = current.get_forecast(steps=available)
        mean = np.asarray(forecast.predicted_mean)
        ci = np.asarray(forecast.conf_int(alpha=.05))
        origin = target_index - 1
        for h in HORIZONS:
            if h <= available:
                destination = origin + h
                assert origin < destination
                rows.append({"Date": dates[destination], "Origin": dates[origin], "Horizon": h,
                             "Actual": values[destination], "ARIMA(1,1,1)": mean[h - 1],
                             "Lower95": ci[h - 1, 0], "Upper95": ci[h - 1, 1],
                             "Naive": values[origin], "Drift": values[origin] + h * drift})
        current = current.extend([values[target_index]])
        assert np.array_equal(current.params, fit.params), "Frozen parameters changed"
    rolling = pd.DataFrame(rows)
    assert (rolling.Origin < rolling.Date).all()
    assert rolling.groupby("Horizon").size().to_dict() == {h: len(test) - h + 1 for h in HORIZONS}
    return fixed, rolling, drift


def calculate_metrics(fixed, rolling):
    rows = []
    groups = [("Fixed origin", "all horizons", fixed)]
    groups += [("Rolling origin", str(h), subset) for h, subset in rolling.groupby("Horizon")]
    groups += [("Rolling origin 2020", "1", rolling[(rolling.Horizon == 1) & (rolling.Date.dt.year == 2020)])]
    for protocol, horizon, data in groups:
        naive_error = data.Actual.to_numpy() - data.Naive.to_numpy()
        naive_mae = np.abs(naive_error).mean()
        naive_rmse = np.sqrt((naive_error ** 2).mean())
        for model in ["Naive", "Drift", "ARIMA(1,1,1)"]:
            error = data.Actual.to_numpy() - data[model].to_numpy()
            mae = float(np.abs(error).mean())
            rmse = float(np.sqrt((error ** 2).mean()))
            # Separate, scalar Python recomputation: catches column/aggregation mistakes.
            independent = [float(actual) - float(pred) for actual, pred in zip(data.Actual, data[model])]
            assert math.isclose(mae, math.fsum(abs(e) for e in independent) / len(independent), rel_tol=1e-12)
            assert math.isclose(rmse, math.sqrt(math.fsum(e * e for e in independent) / len(independent)), rel_tol=1e-12)
            coverage = ((data.Actual >= data.Lower95) & (data.Actual <= data.Upper95)).mean()
            rows.append({"protocol": protocol, "horizon_sessions": horizon, "model": model,
                         "n": len(data), "mae_points": mae, "rmse_points": rmse,
                         "mae_improvement_vs_naive_pct": 100 * (1 - mae / naive_mae),
                         "rmse_improvement_vs_naive_pct": 100 * (1 - rmse / naive_rmse),
                         "arima_95pct_interval_coverage_pct": 100 * float(coverage) if model == "ARIMA(1,1,1)" else None})
    return pd.DataFrame(rows)


def configure_charts():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 16, "axes.titleweight": "bold", "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#253247", "text.color": "#253247",
                         "xtick.color": "#526070", "ytick.color": "#526070",
                         "grid.color": "#DDE2E7", "grid.linewidth": .6,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def save_chart(fig, name, root, note=None):
    if note:
        fig.text(.08, .015, note, ha="left", va="bottom", fontsize=9, color="#526070")
    fig.tight_layout(rect=(0, .05 if note else 0, 1, 1))
    for extension in ["png", "svg"]:
        fig.savefig(root / "charts" / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def date_axis(ax, dates):
    start, end = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    ax.set_xlim(start, end)
    interior = pd.date_range(start, end, periods=5)[1:-1]
    ax.set_xticks([start, *interior, end])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y" if (end - start).days > 500 else "%d %b"))
    ax.grid(axis="y")


def make_charts(frame, fixed, rolling, metrics, train, residual, root):
    configure_charts()
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(frame.index, frame.Close, color=COLORS["Actual"], lw=1.2)
    axes[0].axvspan(pd.Timestamp(TEST_START), frame.index[-1], color="#E6F0F5", zorder=0)
    axes[0].axvline(pd.Timestamp(TRAIN_END), color=COLORS["ARIMA(1,1,1)"], ls="--", lw=1)
    axes[0].set(title="S&P 500 close: development history and held-out evaluation", ylabel="Index points")
    axes[0].text(.015, .92, "Develop: 2000–2013", transform=axes[0].transAxes)
    axes[0].text(.69, .92, "Test: 2014–2020", transform=axes[0].transAxes)
    returns = frame.Close.pct_change() * 100
    axes[1].plot(frame.index, returns, color=COLORS["ARIMA(1,1,1)"], lw=.6)
    axes[1].set(ylabel="Daily return (%)", xlabel="Trading-session date")
    for ax in axes: date_axis(ax, frame.index)
    save_chart(fig, "01_history_returns", root, "Daily Yahoo ^GSPC Close; price index, excluding dividends. No missing-day interpolation.")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    plot_acf(np.diff(train), lags=30, zero=False, ax=axes[0], color=COLORS["ARIMA(1,1,1)"], title="Training differences: ACF")
    plot_pacf(np.diff(train), lags=30, zero=False, method="ywm", ax=axes[1], color=COLORS["ARIMA(1,1,1)"], title="Training differences: PACF")
    for ax in axes: ax.set(xlabel="Lag (trading sessions)", ylabel="Correlation"); ax.grid(axis="y")
    save_chart(fig, "02_training_acf_pacf", root, "Training only, 2000–2013. Shaded bands are approximate 95% correlation limits; no order search on test data.")

    fig, ax = plt.subplots(figsize=(11.4, 5.5))
    ax.fill_between(fixed.Date, fixed.Lower95, fixed.Upper95, color=COLORS["ARIMA(1,1,1)"], alpha=.12, label="ARIMA nominal 95% prediction interval")
    for model, style in [("Actual", "-"), ("ARIMA(1,1,1)", "-"), ("Naive", "--"), ("Drift", ":")]:
        ax.plot(fixed.Date, fixed[model], label=model, color=COLORS[model], ls=style, lw=1.6)
    ax.set(title="One forecast origin: 31 December 2013", ylabel="S&P 500 index points", xlabel="Target date")
    date_axis(ax, fixed.Date)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    save_chart(fig, "03_fixed_origin", root, "No 2014–2020 observations enter these forecasts. Horizon grows from 1 to 1,763 trading sessions; parameters fixed.")

    one = rolling[rolling.Horizon == 1]
    zoom = one[(one.Date >= "2020-02-18") & (one.Date <= "2020-04-30")]
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 7), gridspec_kw={"height_ratios": [1, 1]})
    for ax, data, title in [(axes[0], one, "Rolling one-session forecasts: close tracking can hide little added value"),
                            (axes[1], zoom, "2020 close-up: forecasts react after new information arrives")]:
        for model, style in [("Actual", "-"), ("ARIMA(1,1,1)", "--"), ("Naive", ":")]:
            ax.plot(data.Date, data[model], label=model, color=COLORS[model], ls=style, lw=1.5)
        ax.set(title=title, ylabel="Index points")
        date_axis(ax, data.Date)
        ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=10)
    axes[1].set_xlabel("Target date (2020 for close-up)")
    save_chart(fig, "04_rolling_and_zoom", root, "Each forecast uses observations only through the previous session. ARIMA coefficients remain fixed at their 2013 fit.")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.1))
    test_metrics = metrics[metrics.protocol == "Rolling origin"]
    for ax, measure, label in [(axes[0], "mae_points", "MAE"), (axes[1], "rmse_points", "RMSE")]:
        x = np.arange(len(HORIZONS))
        for offset, model in zip([-.25, 0, .25], ["Naive", "Drift", "ARIMA(1,1,1)"]):
            data = test_metrics[test_metrics.model == model].set_index("horizon_sessions").loc[[str(h) for h in HORIZONS]]
            bars = ax.bar(x + offset, data[measure], .24, label=model, color=COLORS[model])
            ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        ax.set(title=f"Rolling forecast {label}", xlabel="Horizon (trading sessions)", ylabel=f"{label} (index points)")
        ax.set_xticks(x, HORIZONS)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=10)
    save_chart(fig, "05_error_benchmarks", root, "Same target rows within each horizon. Origins overlap at 5/20 sessions; comparisons are descriptive, without significance claims.")

    annual = one.groupby(one.Date.dt.year).apply(lambda data: pd.Series({
        "ARIMA_RMSE": np.sqrt(np.mean((data.Actual - data["ARIMA(1,1,1)"])**2)),
        "Naive_RMSE": np.sqrt(np.mean((data.Actual - data.Naive)**2)),
        "Coverage_pct": 100 * np.mean((data.Actual >= data.Lower95) & (data.Actual <= data.Upper95))
    }), include_groups=False)
    annual.index.name = "Year"
    annual.to_csv(root / "results" / "annual_one_step.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    axes[0].plot(annual.index, annual.ARIMA_RMSE, "o-", label="ARIMA(1,1,1)", color=COLORS["ARIMA(1,1,1)"])
    axes[0].plot(annual.index, annual.Naive_RMSE, "x--", label="Naive", color=COLORS["Naive"])
    axes[0].set(title="One-session RMSE by target year", ylabel="RMSE (index points)", xlabel="Year", ylim=(0, None))
    axes[0].legend(frameon=False)
    axes[1].bar(annual.index, annual.Coverage_pct, color=COLORS["ARIMA(1,1,1)"])
    axes[1].axhline(95, color=COLORS["Actual"], ls="--", label="Nominal 95%")
    axes[1].set(title="ARIMA interval coverage by year", ylabel="Actual closes inside interval (%)", xlabel="Year", ylim=(0, 105))
    axes[1].legend(frameon=False)
    for ax in axes: ax.grid(axis="y"); ax.set_axisbelow(True); ax.set_xticks(annual.index)
    save_chart(fig, "06_annual_error_coverage", root, "Intervals assume the training model and innovation variance remain appropriate; parameter uncertainty is excluded.")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    plot_acf(residual, lags=30, zero=False, ax=axes[0], color=COLORS["ARIMA(1,1,1)"], title="Training residual autocorrelation")
    plot_acf(residual**2, lags=30, zero=False, ax=axes[1], color=COLORS["ARIMA(1,1,1)"], title="Squared residual autocorrelation")
    for ax in axes: ax.set(xlabel="Lag (trading sessions)", ylabel="Correlation"); ax.grid(axis="y")
    save_chart(fig, "07_residual_diagnostics", root, "Diffuse initialization residual excluded. Squared residual dependence is evidence against constant conditional variance.")
    return annual


def run(data_path=None, output_root=ROOT):
    data_path = Path(data_path or ROOT / "local_data" / "yahoo_gspc_2000_2020_close.csv")
    output_root = Path(output_root)
    for folder in ["results", "charts"]: (output_root / folder).mkdir(parents=True, exist_ok=True)
    frame = load_data(data_path)
    audit = audit_data(frame, data_path)
    print(f"Validated {len(frame):,} daily observations and calendar coverage.", flush=True)
    train_frame = frame.loc[:TRAIN_END]
    train = train_frame.Close.to_numpy()
    fit, fit_warnings = fit_model(train)
    print("Training MLE converged; running diagnostics and forecasts.", flush=True)
    diagnostic, residual = diagnostics(train, fit, fit_warnings)
    fixed, rolling, drift = make_forecasts(frame, fit)
    print("Fixed-origin and rolling forecasts completed without leakage assertions failing.", flush=True)
    metrics = calculate_metrics(fixed, rolling)
    for name, data in [("fixed_origin_forecasts", fixed), ("rolling_forecasts", rolling), ("metrics", metrics)]:
        data.to_csv(output_root / "results" / f"{name}.csv", index=False)
    annual = make_charts(frame, fixed, rolling, metrics, train, residual, output_root)
    # Retain the chart-ready EDA data for the local academic presentation.
    frame.assign(Return_pct=frame.Close.pct_change() * 100).to_csv(output_root / "results" / "history_chart_data.csv")
    versions = {package: importlib.metadata.version(package) for package in
                ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "exchange-calendars", "nbformat", "nbclient", "nbconvert", "ipykernel"]}
    result = {
        "status": "EXECUTED_NEW_RECONSTRUCTION_NOT_ORIGINAL_COLLEGE_RESULTS",
        "data_audit": audit, "target": "Daily ^GSPC Close in index points; not total return",
        "train": {"start": str(train_frame.index[0].date()), "end": str(train_frame.index[-1].date()), "n": len(train)},
        "test": {"start": str(fixed.Date.iloc[0].date()), "end": str(fixed.Date.iloc[-1].date()), "n": len(fixed)},
        "model": {"order": [1, 1, 1], "trend": "n (no drift)", "estimation": "state-space Gaussian maximum likelihood",
                  "selection": "Prespecified from supplied note, not tuned on the test set",
                  "parameter_refitting": "None during 2014–2020; coefficients and variance frozen at end-2013",
                  "rolling_update": "Forecast before revealing target; extend filter with each newly observed close",
                  "fixed_origin": "Only observations through 2013-12-31; all 1763 future targets forecast together",
                  "horizon_units": "Observed trading sessions, not calendar days",
                  "drift_baseline_points_per_session": drift},
        "diagnostics": diagnostic, "metrics": metrics.where(pd.notna(metrics), None).to_dict("records"),
        "limitations": ["New reconstruction; original data, notebook, fitted coefficients and forecast protocol were not supplied.",
                        "Current Yahoo historical vintage, not verified contemporaneous point-in-time data.",
                        "No independent cross-provider value verification; calendar checked against XNYS proxy.",
                        "Nominal Gaussian ARIMA prediction intervals exclude parameter uncertainty and allow constant innovation variance.",
                        "5/20-session forecast errors overlap; no significance test or trading profitability claim.",
                        "2020 slice is a descriptive stress-period view, not an independently chosen second holdout.",
                        "No exogenous/LLM/GARCH/ML forecaster was fitted; suggested extensions are untested.",
                        "No open raw-data redistribution licence verified; keep local_data and detailed chart/forecast extracts private."],
        "validation": {"chronological_forecast_assertions": "Passed", "frozen_parameter_assertions": "Passed",
                       "independent_scalar_error_recomputation": "Passed", "calendar_reconciliation": "Passed",
                       "python": platform.python_version(), "packages": versions},
    }
    # Pandas numeric columns keep NaN when replacing with None; convert via JSON.
    result["metrics"] = json.loads(metrics.to_json(orient="records"))
    write_json(output_root / "results" / "summary.json", result)
    monthly_history = frame.reset_index().groupby(frame.index.to_period("M")).tail(1)
    monthly_fixed = fixed.groupby(fixed.Date.dt.to_period("M")).tail(1)
    zoom = rolling[(rolling.Horizon == 1) & (rolling.Date >= "2020-02-01") & (rolling.Date <= "2020-04-30")]
    def chart_arrays(data, columns):
        return {"dates": data.Date.dt.strftime("%Y-%m-%d").tolist(),
                **{output: data[source].astype(float).tolist() for output, source in columns.items()}}
    deck_data = {
        "history_monthly": chart_arrays(monthly_history, {"close": "Close"}),
        "fixed_monthly": chart_arrays(monthly_fixed, {"actual": "Actual", "arima": "ARIMA(1,1,1)", "naive": "Naive", "drift": "Drift", "lower95": "Lower95", "upper95": "Upper95"}),
        "rolling_2020_zoom": chart_arrays(zoom, {"actual": "Actual", "arima": "ARIMA(1,1,1)", "naive": "Naive", "lower95": "Lower95", "upper95": "Upper95"}),
        "metrics": result["metrics"], "diagnostics": diagnostic,
        "methodology": {**result["model"], "train": result["train"], "test": result["test"],
                        "target": result["target"], "plot_sampling": "Month-end observations for selected charts only; all metrics use daily targets"},
        "annual": json.loads(annual.reset_index().to_json(orient="records")),
        "data_audit": audit,
    }
    write_json(output_root / "results" / "deck_data.json", deck_data)
    (output_root / "results" / "model_summary.txt").write_text(str(fit.summary()), encoding="utf-8")
    (output_root / "requirements.txt").write_text("\n".join(f"{key}=={value}" for key, value in versions.items()) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.data)
    print(pd.DataFrame(result["metrics"])[["protocol", "horizon_sessions", "model", "n", "mae_points", "rmse_points", "rmse_improvement_vs_naive_pct"]].to_string(index=False))
    print("Saved results, diagnostics, and charts under", ROOT)
