"""Exploratory 20-session expanding refit of the same ARIMA(1,1,1) order.

The 2014-2020 outcomes were known before this extension was designed. Its
results are retrospective, not a new untouched holdout result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from analysis import (
    HORIZONS,
    ROOT,
    TEST_START,
    TRAIN_END,
    audit_data,
    fit_model,
    load_data,
    write_json,
)
from evaluation import audit_forecast_ledger, interval_quality

REFIT_EVERY = 20
REFIT_NAME = "ARIMA(1,1,1), refit every 20"
FROZEN_NAME = "ARIMA(1,1,1), frozen"


def make_refit_forecasts(frame):
    """Forecast first; refit only after each block of 20 actual closes exists."""
    dates = frame.index
    values = frame.Close.to_numpy()
    n_train = int((dates <= TRAIN_END).sum())
    if dates[n_train - 1] != pd.Timestamp(TRAIN_END) or dates[n_train] < pd.Timestamp(TEST_START):
        raise ValueError("Unexpected training/test boundary")
    current, initial_warnings = fit_model(values[:n_train])
    last_fit = dates[n_train - 1]
    fit_events = [{"last_observation_date": last_fit, "observations": n_train,
                   "warnings": initial_warnings}]
    rows = []
    for target_index in range(n_train, len(values)):
        observed_test_sessions = target_index - n_train
        if observed_test_sessions and observed_test_sessions % REFIT_EVERY == 0:
            current, fit_warnings = fit_model(values[:target_index])
            last_fit = dates[target_index - 1]
            fit_events.append({"last_observation_date": last_fit,
                               "observations": target_index, "warnings": fit_warnings})
        available = min(max(HORIZONS), len(values) - target_index)
        forecast = current.get_forecast(steps=available)
        means = np.asarray(forecast.predicted_mean)
        intervals = np.asarray(forecast.conf_int(alpha=.05))
        origin = target_index - 1
        for horizon in HORIZONS:
            if horizon <= available:
                target = origin + horizon
                rows.append({"Date": dates[target], "Origin": dates[origin], "Horizon": horizon,
                             "Actual": values[target], "RefitARIMA": means[horizon - 1],
                             "RefitLower95": intervals[horizon - 1, 0],
                             "RefitUpper95": intervals[horizon - 1, 1],
                             "LastParameterFitDate": last_fit,
                             "LastStateObservationDate": dates[origin]})
        # The new close becomes available only after this origin's forecasts.
        current = current.extend([values[target_index]])
    challenger = pd.DataFrame(rows)
    if challenger.groupby("Horizon").size().to_dict() != {
        horizon: len(values) - n_train - horizon + 1 for horizon in HORIZONS
    }:
        raise ValueError("A refit forecast task is missing")
    return challenger, fit_events


def compare_on_identical_tasks(reference, challenger, frame):
    """Reject missing, duplicate, shifted, or future-informed comparisons."""
    keys = ["Date", "Origin", "Horizon"]
    if reference.duplicated(keys).any() or challenger.duplicated(keys).any():
        raise ValueError("Duplicate forecast task")
    joined = reference.merge(challenger, on=keys, how="outer", validate="one_to_one",
                             indicator=True, suffixes=("", "_refit"))
    if len(joined) != len(reference) or not joined._merge.eq("both").all():
        raise ValueError("Frozen and refit procedures do not share every task")
    if not np.allclose(joined.Actual, joined.Actual_refit, rtol=0, atol=1e-10):
        raise ValueError("Actual values differ between procedures")
    origin_pos = frame.index.get_indexer(joined.Origin)
    target_pos = frame.index.get_indexer(joined.Date)
    if (origin_pos < 0).any() or (target_pos < 0).any():
        raise ValueError("Forecast date not in the source series")
    if not np.array_equal(target_pos - origin_pos, joined.Horizon.to_numpy()):
        raise ValueError("Forecast horizon misaligned")
    if not np.allclose(joined.Actual, frame.Close.to_numpy()[target_pos], rtol=0, atol=1e-10):
        raise ValueError("Reference actual does not match the source series")
    if not np.allclose(joined.Naive, frame.Close.to_numpy()[origin_pos], rtol=0, atol=1e-10):
        raise ValueError("Last-close baseline does not match the forecast origin")
    training_close = frame.loc[:TRAIN_END, "Close"].to_numpy()
    drift = float(np.diff(training_close).mean())
    if not np.allclose(joined.Drift, joined.Naive + joined.Horizon * drift, rtol=0, atol=1e-8):
        raise ValueError("Drift baseline used an incorrect information cutoff")
    if (joined.LastParameterFitDate > joined.Origin).any():
        raise ValueError("Refit used a future observation")
    if not joined.LastStateObservationDate.eq(joined.Origin).all():
        raise ValueError("Refit state has an invalid information cutoff")
    joined = joined.drop(columns=["_merge", "Actual_refit"])
    return joined.sort_values(["Horizon", "Date"]).reset_index(drop=True)


def score_comparison(joined):
    rows = []
    models = (("Naive", "Naive", None), ("Drift", "Drift", None),
              (FROZEN_NAME, "ARIMA(1,1,1)", ("Lower95", "Upper95")),
              (REFIT_NAME, "RefitARIMA", ("RefitLower95", "RefitUpper95")))
    for horizon, data in joined.groupby("Horizon"):
        for name, forecast, bounds in models:
            errors = data.Actual.to_numpy() - data[forecast].to_numpy()
            interval = interval_quality(data.Actual, data[bounds[0]], data[bounds[1]]) if bounds else None
            rows.append({"horizon_sessions": int(horizon), "model": name, "n": len(data),
                         "mae_points": float(np.mean(np.abs(errors))),
                         "rmse_points": float(math.sqrt(np.mean(errors**2))),
                         "coverage95_pct": interval["coverage_pct"] if interval else None,
                         "mean_width95_points": interval["mean_width_points"] if interval else None,
                         "mean_winkler95_points": interval["mean_winkler_score_points"] if interval else None})
    return pd.DataFrame(rows)


def run(data_path, reference_run, output_dir=None):
    data_path = Path(data_path).resolve()
    reference_run = Path(reference_run).resolve()
    frame = load_data(data_path)
    audit = audit_data(frame, data_path)
    reference_config = json.loads((reference_run / "run_config.json").read_text(encoding="utf-8"))
    if (reference_config["data_sha256"] != audit["csv_sha256"] or
            reference_config["arima_order"] != [1, 1, 1] or
            reference_config["parameters_refitted"] is not False or
            reference_config["train_end"] != TRAIN_END or
            reference_config["horizons_sessions"] != list(HORIZONS)):
        raise ValueError("Reference run does not match the controlled refit protocol")
    reference = pd.read_csv(reference_run / "results" / "rolling_forecasts.csv",
                            parse_dates=["Date", "Origin"])
    reference_ledger = pd.read_csv(
        reference_run / "results" / "forecast_ledger.csv",
        parse_dates=["origin_date", "target_date", "last_parameter_fit_date", "last_state_observation_date"],
    )
    train = frame.loc[:TRAIN_END, "Close"].to_numpy()
    audit_forecast_ledger(reference_ledger, frame, float(np.diff(train).mean()), TRAIN_END)
    if output_dir is None:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_refit20_" + uuid4().hex[:8]
        output_dir = ROOT / "runs" / run_id
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "results").mkdir()
    config = {
        "status": "RETROSPECTIVE_EXPLORATORY_EXTENSION_NOT_ORIGINAL_COLLEGE_RESULTS",
        "created_utc": datetime.now(UTC).isoformat(), "data_sha256": audit["csv_sha256"],
        "reference_run": str(reference_run),
        "reference_rolling_sha256": hashlib.sha256((reference_run / "results" / "rolling_forecasts.csv").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "exchange-calendars")},
        "target": "Daily S&P 500 price-index close, index points",
        "first_test_target": str(reference.Date.min().date()),
        "last_test_target": str(reference.Date.max().date()),
        "primary_task": "Next-session close; compare refit ARIMA with frozen ARIMA and last close",
        "secondary_horizons_sessions": [5, 20],
        "refit_policy": "Expanding sample, same ARIMA(1,1,1) no-drift specification; refit after every 20 newly observed test closes",
        "initial_fit_last_observation": TRAIN_END,
        "failure_rule": "Abort and report failure; no task may be silently excluded",
        "intervals": "Model-based nominal 95%; assess coverage, width, Winkler score",
        "known_test_outcomes": True,
        "inference_scope": "Exploratory retrospective comparison, not an untouched holdout or statistical-superiority claim",
    }
    write_json(output_dir / "run_config.json", config)
    challenger, fit_events = make_refit_forecasts(frame)
    joined = compare_on_identical_tasks(reference, challenger, frame)
    metrics = score_comparison(joined)
    annual = pd.concat(
        [score_comparison(subset).assign(target_year=int(year))
         for year, subset in joined[joined.Horizon == 1].groupby(joined.Date.dt.year)],
        ignore_index=True,
    )
    joined.to_csv(output_dir / "results" / "aligned_forecasts.csv", index=False)
    metrics.to_csv(output_dir / "results" / "metrics.csv", index=False)
    annual.to_csv(output_dir / "results" / "annual_one_step.csv", index=False)
    write_json(output_dir / "results" / "fit_events.json", [
        {**event, "last_observation_date": str(event["last_observation_date"].date())}
        for event in fit_events
    ])
    summary = {"status": config["status"], "data_sha256": audit["csv_sha256"],
               "tasks_aligned": len(joined), "fit_events": len(fit_events),
               "refits_after_initial_fit": len(fit_events) - 1,
               "fit_warnings": sum(len(event["warnings"]) for event in fit_events),
               "metrics": json.loads(metrics.to_json(orient="records")),
               "run_directory": str(output_dir)}
    write_json(output_dir / "results" / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.data, args.reference_run, args.output_dir)
    print(pd.DataFrame(result["metrics"]).to_string(index=False))
    print("Saved retrospective extension under", result["run_directory"])
