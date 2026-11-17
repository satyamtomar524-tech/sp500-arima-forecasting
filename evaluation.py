"""Auditable forecast comparisons for the later S&P 500 reconstruction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MODELS = ("Naive", "Drift", "ARIMA(1,1,1)")


def forecast_ledger(fixed, rolling, frame, drift, train_end):
    """Keep one row per model and task, with its information cutoff visible."""
    pieces = []
    lower_parts = []
    upper_parts = []
    fit_date_parts = []
    for protocol, wide in (("Fixed origin", fixed), ("Rolling origin", rolling)):
        origins = pd.to_datetime(wide.Origin)
        targets = pd.to_datetime(wide.Date)
        origin_closes = frame.Close.reindex(origins).to_numpy()
        for model in MODELS:
            is_arima = model == "ARIMA(1,1,1)"
            pieces.append(pd.DataFrame({
                "protocol": protocol,
                "origin_date": origins.to_numpy(),
                "target_date": targets.to_numpy(),
                "horizon_sessions": wide.Horizon.to_numpy(),
                "model": model,
                "origin_close": origin_closes,
                "forecast": wide[model].to_numpy(),
                "actual": wide.Actual.to_numpy(),
                "last_state_observation_date": origins.to_numpy(),
            }))
            lower_parts.append(wide.Lower95.to_numpy() if is_arima else np.full(len(wide), np.nan))
            upper_parts.append(wide.Upper95.to_numpy() if is_arima else np.full(len(wide), np.nan))
            fit_date_parts.append(np.full(len(wide), np.datetime64(train_end) if model != "Naive" else np.datetime64("NaT"), dtype="datetime64[ns]"))
    ledger = pd.concat(pieces, ignore_index=True)
    ledger["lower95"] = np.concatenate(lower_parts)
    ledger["upper95"] = np.concatenate(upper_parts)
    ledger["last_parameter_fit_date"] = np.concatenate(fit_date_parts)
    audit_forecast_ledger(ledger, frame, drift, train_end)
    return ledger


def audit_forecast_ledger(ledger, frame, drift, train_end):
    """Check target alignment and the information available at each origin."""
    required = {"protocol", "origin_date", "target_date", "horizon_sessions", "model",
                "origin_close", "forecast", "actual", "lower95", "upper95",
                "last_parameter_fit_date", "last_state_observation_date"}
    if not required.issubset(ledger.columns) or ledger.empty:
        raise ValueError("Forecast ledger is empty or has missing columns")
    keys = ["protocol", "origin_date", "target_date", "horizon_sessions"]
    if ledger.duplicated(keys + ["model"]).any():
        raise ValueError("Duplicate model forecast for a task")
    if not ledger.groupby(keys).model.agg(lambda x: set(x) == set(MODELS)).all():
        raise ValueError("Models do not share every forecast task")
    origins = pd.to_datetime(ledger.origin_date)
    targets = pd.to_datetime(ledger.target_date)
    origin_pos = frame.index.get_indexer(origins)
    target_pos = frame.index.get_indexer(targets)
    if (origin_pos < 0).any() or (target_pos < 0).any():
        raise ValueError("Forecast date is absent from the input series")
    if not np.array_equal(target_pos - origin_pos, ledger.horizon_sessions.to_numpy()):
        raise ValueError("Origin, target, and trading-session horizon disagree")
    np.testing.assert_allclose(ledger.actual, frame.Close.to_numpy()[target_pos], rtol=0, atol=1e-10)
    np.testing.assert_allclose(ledger.origin_close, frame.Close.to_numpy()[origin_pos], rtol=0, atol=1e-10)
    if not (pd.to_datetime(ledger.last_state_observation_date) == origins).all():
        raise ValueError("A forecast used a state updated beyond its origin")
    fitted = ledger.model.ne("Naive")
    if not (pd.to_datetime(ledger.loc[fitted, "last_parameter_fit_date"]) == pd.Timestamp(train_end)).all():
        raise ValueError("ARIMA or drift fit date is inconsistent")
    if (pd.to_datetime(ledger.loc[fitted, "last_parameter_fit_date"]) > origins[fitted]).any():
        raise ValueError("Parameters were fitted after a forecast origin")
    naive = ledger.model.eq("Naive")
    drift_rows = ledger.model.eq("Drift")
    np.testing.assert_allclose(ledger.loc[naive, "forecast"], ledger.loc[naive, "origin_close"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        ledger.loc[drift_rows, "forecast"],
        ledger.loc[drift_rows, "origin_close"] + drift * ledger.loc[drift_rows, "horizon_sessions"],
        rtol=0, atol=1e-8,
    )
    arima = ledger.model.eq("ARIMA(1,1,1)")
    if ledger.loc[arima, ["lower95", "upper95"]].isna().any().any():
        raise ValueError("ARIMA interval is missing")
    if (ledger.loc[arima, "lower95"] > ledger.loc[arima, "upper95"]).any():
        raise ValueError("ARIMA interval bounds are reversed")
    if ledger.loc[~arima, ["lower95", "upper95"]].notna().any().any():
        raise ValueError("Baseline interval was labelled as an ARIMA interval")


def interval_quality(actual, lower, upper, alpha=0.05):
    """Coverage, mean width and Winkler score, all for the same target rows."""
    actual, lower, upper = (np.asarray(x, dtype=float) for x in (actual, lower, upper))
    if not (actual.shape == lower.shape == upper.shape) or actual.size == 0:
        raise ValueError("Interval arrays must be nonempty and aligned")
    if not 0 < alpha < 1 or not np.isfinite([actual, lower, upper]).all() or (lower > upper).any():
        raise ValueError("Invalid interval inputs")
    width = upper - lower
    score = width + (2 / alpha) * np.maximum(lower - actual, 0) + (2 / alpha) * np.maximum(actual - upper, 0)
    return {
        "coverage_pct": 100 * float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_width_points": float(np.mean(width)),
        "mean_winkler_score_points": float(np.mean(score)),
    }


def paired_one_step_check(ledger, metrics):
    """Recompute the primary MAE and RMSE from ledger rows independently."""
    one = ledger[(ledger.protocol == "Rolling origin") & (ledger.horizon_sessions == 1)]
    answer = {}
    for model in MODELS:
        rows = one[one.model == model].sort_values("target_date")
        errors = [float(y) - float(f) for y, f in zip(rows.actual, rows.forecast)]
        if not errors:
            raise ValueError("No one-session observations")
        mae = math.fsum(abs(e) for e in errors) / len(errors)
        rmse = math.sqrt(math.fsum(e * e for e in errors) / len(errors))
        expected = metrics[(metrics.protocol == "Rolling origin") &
                           (metrics.horizon_sessions == "1") & (metrics.model == model)].iloc[0]
        if len(errors) != expected.n or not math.isclose(mae, expected.mae_points, rel_tol=1e-12) or not math.isclose(rmse, expected.rmse_points, rel_tol=1e-12):
            raise ValueError(f"Independent one-session metric check failed for {model}")
        answer[model] = {"n": len(errors), "mae_points": mae, "rmse_points": rmse}
    return answer


def block_bootstrap_rmse_gain(rolling, block_lengths=(5, 20, 60), replicates=2000, seed=20261117):
    """Exploratory paired circular-block intervals for naive RMSE minus ARIMA RMSE."""
    one = rolling[rolling.Horizon == 1].sort_values("Date")
    actual = one.Actual.to_numpy(dtype=float)
    naive_error = actual - one.Naive.to_numpy(dtype=float)
    arima_error = actual - one["ARIMA(1,1,1)"].to_numpy(dtype=float)
    n = len(one)
    if n < max(block_lengths) or replicates < 100:
        raise ValueError("Too few observations or bootstrap replicates")
    point_gain = float(np.sqrt(np.mean(naive_error**2)) - np.sqrt(np.mean(arima_error**2)))
    results = []
    rng = np.random.default_rng(seed)
    for block in block_lengths:
        differences = np.empty(replicates)
        offsets = np.arange(block)
        for index in range(replicates):
            starts = rng.integers(0, n, size=math.ceil(n / block))
            chosen = ((starts[:, None] + offsets) % n).ravel()[:n]
            differences[index] = (np.sqrt(np.mean(naive_error[chosen] ** 2)) -
                                  np.sqrt(np.mean(arima_error[chosen] ** 2)))
        low, high = np.quantile(differences, [0.025, 0.975])
        results.append({"block_sessions": block, "replicates": replicates, "seed": seed,
                        "n_paired_tasks": n, "naive_minus_arima_rmse_points": point_gain,
                        "percentile_95_low_points": float(low), "percentile_95_high_points": float(high),
                        "interval_includes_zero": bool(low <= 0 <= high)})
    return pd.DataFrame(results)
