"""Fast checks for PBI 07 using a small synthetic batch (no MLflow, no real data)."""

import unittest

import numpy as np
import pandas as pd

from src.feature_engineering import INPUT_FEATURES, build_preprocessor
from src.data_access import VALIDATION_START
from src.train_nithish import (FEATURES, VERSIONS, best_f1_cutoff, boosting_pipeline, logistic_pipeline, prepare_data,
                               recent_training_window, train_nithish_models)


def synthetic_tables(rows=3000, seed=0):
    """Clean-looking tables where recent first-time customers are riskier."""
    rng = np.random.default_rng(seed)
    customer = rng.integers(0, 600, rows)
    order_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 100, rows)), unit="D")
    first_order = pd.Timestamp("2022-12-25") - pd.to_timedelta(rng.integers(0, 400, 600), unit="D")
    first_order = pd.Series(first_order)
    first_order[:300] = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 100, 300), unit="D")
    customers = pd.DataFrame({"country_code": "MY", "customer_id": [f"c{i}" for i in range(600)],
                              "first_order_datetime": first_order})
    orders = pd.DataFrame({
        "country_code": "MY", "order_id": [f"o{i}" for i in range(rows)],
        "collect_type": rng.choice(["delivery", "pickup"], rows),
        "payment_method": rng.choice(["credit card", "payment on Delivery", "payment on delivery", None], rows),
        "order_value": rng.gamma(2, 5, rows), "num_items_ordered": rng.integers(1, 6, rows).astype(float),
        "refund_value": 0.0, "order_date": order_date})
    days_since_first_order = (order_date - first_order[customer].to_numpy()).days
    risk = np.where((days_since_first_order >= 0) & (days_since_first_order < 30), 0.6, 0.08)
    labels = pd.DataFrame({"country_code": "MY", "order_id": orders["order_id"],
                           "customer_id": [f"c{i}" for i in customer],
                           "is_fraud": (rng.random(rows) < risk).astype(int)})
    return customers, orders, labels


class TrainNithishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = synthetic_tables()
        cls.data = prepare_data(*cls.tables)

    def test_schema_is_team_features_plus_first_order_history(self):
        x_train, _, x_val, _, _, _ = self.data
        self.assertEqual(FEATURES, INPUT_FEATURES)
        self.assertEqual(list(x_train.columns), FEATURES)
        self.assertEqual(list(x_val.columns), FEATURES)

    def test_team_model_pipelines_use_shared_preprocessor(self):
        shared_type = type(build_preprocessor())
        self.assertIsInstance(logistic_pipeline().named_steps["prepare"], shared_type)
        self.assertIsInstance(boosting_pipeline().named_steps["prepare"], shared_type)

    def test_days_since_first_order_matches_dates_and_is_never_negative(self):
        customers, orders, labels = self.tables
        x_train = self.data[0]
        row = x_train.index[0]                         # index = row position in labels
        first = customers.set_index("customer_id").loc[labels.loc[row, "customer_id"], "first_order_datetime"]
        expected = (orders.loc[row, "order_date"].normalize() - first.normalize()).days
        actual = x_train.loc[row, "days_since_first_order"]
        self.assertTrue(np.isnan(actual) if expected < 0 else actual == expected)
        self.assertTrue((x_train["days_since_first_order"].dropna() >= 0).all())
        self.assertTrue(x_train["days_since_first_order"].isna().any())   # contradictory dates -> missing

    def test_payment_method_casing_is_merged(self):
        methods = set(self.data[0]["payment_method"].dropna())
        self.assertNotIn("payment on Delivery", methods)
        self.assertIn("payment on delivery", methods)

    def test_split_is_chronological_and_test_is_not_returned(self):
        x_train, _, x_val, _, train_dates, report = self.data
        self.assertLess(report["train_last_date"], report["validation_first_date"])
        self.assertEqual(report["validation_first_date"], str(VALIDATION_START.date()))
        self.assertEqual(len(train_dates), len(x_train))
        self.assertEqual(len(x_train) + len(x_val), 3000)
        self.assertNotIn("test_rows", report)

    def test_cutoff_window_uses_latest_training_dates_only(self):
        train_dates = self.data[4]
        recent = recent_training_window(train_dates)
        self.assertGreater(train_dates[recent].min(), train_dates[~recent].max())

    def test_best_f1_cutoff_separates_clear_scores(self):
        cutoff = best_f1_cutoff(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertTrue(0.2 < cutoff <= 0.8)

    def test_three_versions_train_and_learn_the_first_order_history_signal(self):
        results = train_nithish_models(*self.tables, log=False)
        self.assertEqual([row["version"] for row in results], list(VERSIONS))
        self.assertEqual(results[0]["threshold"], 0.5)
        self.assertIs(results[0]["model"], results[1]["model"])      # v1 and v2 share one model
        for row in results:
            self.assertGreater(row["metrics"]["roc_auc"], 0.7)
            self.assertNotIn("run_id", row)                          # nothing logged when log=False


if __name__ == "__main__":
    unittest.main()
