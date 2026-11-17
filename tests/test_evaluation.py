"""Checks for the comparisons most vulnerable to misalignment."""

import unittest

import numpy as np
import pandas as pd

from evaluation import (
    audit_forecast_ledger,
    block_bootstrap_rmse_gain,
    forecast_ledger,
    interval_quality,
)
from refit_experiment import compare_on_identical_tasks


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        dates = pd.to_datetime(["2013-12-30", "2013-12-31", "2014-01-02", "2014-01-03"])
        self.frame = pd.DataFrame({"Close": [98.0, 100.0, 103.0, 101.0]}, index=dates)
        self.fixed = pd.DataFrame({
            "Date": dates[2:], "Origin": dates[1], "Horizon": [1, 2],
            "Actual": [103.0, 101.0], "ARIMA(1,1,1)": [101.0, 102.0],
            "Lower95": [99.0, 98.0], "Upper95": [103.0, 106.0],
            "Naive": [100.0, 100.0], "Drift": [102.0, 104.0],
        })
        self.rolling = pd.DataFrame({
            "Date": dates[2:], "Origin": dates[1:3], "Horizon": [1, 1],
            "Actual": [103.0, 101.0], "ARIMA(1,1,1)": [101.0, 102.0],
            "Lower95": [99.0, 100.0], "Upper95": [103.0, 104.0],
            "Naive": [100.0, 103.0], "Drift": [102.0, 105.0],
        })

    def test_ledger_keeps_models_on_matching_tasks(self):
        ledger = forecast_ledger(self.fixed, self.rolling, self.frame, 2.0, "2013-12-31")
        self.assertEqual(len(ledger), 12)
        self.assertEqual(ledger.groupby(["protocol", "origin_date", "target_date", "horizon_sessions"]).size().tolist(), [3] * 4)

    def test_ledger_rejects_wrong_baseline_and_horizon(self):
        ledger = forecast_ledger(self.fixed, self.rolling, self.frame, 2.0, "2013-12-31")
        wrong_baseline = ledger.copy()
        wrong_baseline.loc[wrong_baseline.model.eq("Naive"), "forecast"] += 1
        with self.assertRaises(AssertionError):
            audit_forecast_ledger(wrong_baseline, self.frame, 2.0, "2013-12-31")
        wrong_horizon = ledger.copy()
        wrong_horizon.loc[0, "horizon_sessions"] = 20
        with self.assertRaises(ValueError):
            audit_forecast_ledger(wrong_horizon, self.frame, 2.0, "2013-12-31")

    def test_interval_score_penalizes_misses(self):
        quality = interval_quality([5, 0, 10], [3, 3, 3], [7, 7, 7])
        self.assertAlmostEqual(quality["coverage_pct"], 100 / 3)
        self.assertEqual(quality["mean_width_points"], 4)
        self.assertEqual(quality["mean_winkler_score_points"], 84)

    def test_paired_block_bootstrap_is_reproducible(self):
        n = 100
        rolling = pd.DataFrame({"Date": pd.date_range("2014-01-01", periods=n),
                                "Horizon": np.ones(n), "Actual": np.arange(n, dtype=float),
                                "Naive": np.arange(n, dtype=float) - 1,
                                "ARIMA(1,1,1)": np.arange(n, dtype=float) - 0.9})
        first = block_bootstrap_rmse_gain(rolling, block_lengths=(5,), replicates=100, seed=17)
        second = block_bootstrap_rmse_gain(rolling, block_lengths=(5,), replicates=100, seed=17)
        pd.testing.assert_frame_equal(first, second)
        self.assertAlmostEqual(first.iloc[0].naive_minus_arima_rmse_points, 0.1)

    def test_refit_comparison_rejects_shifted_or_future_informed_tasks(self):
        challenger = self.rolling[["Date", "Origin", "Horizon", "Actual"]].copy()
        challenger["RefitARIMA"] = [101.5, 102.5]
        challenger["RefitLower95"] = [99.0, 100.0]
        challenger["RefitUpper95"] = [104.0, 105.0]
        challenger["LastParameterFitDate"] = pd.Timestamp("2013-12-31")
        challenger["LastStateObservationDate"] = challenger.Origin
        self.assertEqual(len(compare_on_identical_tasks(self.rolling, challenger, self.frame)), 2)
        shifted = challenger.copy()
        shifted.loc[0, "Date"] = pd.Timestamp("2014-01-03")
        with self.assertRaises(ValueError):
            compare_on_identical_tasks(self.rolling, shifted, self.frame)
        future_fit = challenger.copy()
        future_fit.loc[0, "LastParameterFitDate"] = pd.Timestamp("2014-01-03")
        with self.assertRaises(ValueError):
            compare_on_identical_tasks(self.rolling, future_fit, self.frame)


if __name__ == "__main__":
    unittest.main()
