import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy import sparse

from src.data_cleaning import clean_customers, clean_orders, clean_labels
from src.feature_engineering import INPUT_FEATURES, prepare_features, prepare_training_data, fit_preprocessor, training_sample_weights
from test_data_cleaning import customer, order, label


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.customers = clean_customers(pd.DataFrame([customer()]))[0]
        orders = []
        labels = []
        for i in range(20):
            date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)
            method = "cash" if i < 14 else "unseen"
            value = None if i == 1 else (1 if i < 14 else 1000)
            orders.append(order(order_id=str(i), order_date=date.strftime("%Y-%m-%d"),
                                order_value=value, payment_method=method))
            labels.append(label(order_id=str(i), is_fraud=i % 2))
        labels.append(label(order_id="missing"))
        self.orders = clean_orders(pd.DataFrame(orders))[0]
        self.labels = clean_labels(pd.DataFrame(labels))[0]

    def test_split_joins_leakage_and_reproducibility(self):
        x_train, y_train, x_val, y_val, x_test, y_test, report = prepare_training_data(
            self.customers, self.orders, self.labels
        )
        self.assertEqual(report["labels_without_orders"], 1)
        self.assertEqual((len(x_train), len(x_val), len(x_test)), (14, 3, 3))
        self.assertLess(report["train_last_date"], report["validation_first_date"])
        self.assertLess(report["validation_last_date"], report["test_first_date"])
        for features in (x_train, x_val, x_test):
            self.assertEqual(list(features.columns), INPUT_FEATURES)
        self.assertTrue(x_train.index.intersection(x_val.index).empty)
        self.assertTrue(x_train.index.intersection(x_test.index).empty)
        self.assertTrue(x_val.index.intersection(x_test.index).empty)
        for forbidden in ["refund_value", "is_fraud", "order_id", "customer_id",
                          "first_order_datetime", "num_refund_orders_last_50days"]:
            self.assertNotIn(forbidden, x_train.columns)
        changed_orders = self.orders.copy()
        changed_orders["refund_value"] = 999999
        changed_customers = self.customers.copy()
        changed_customers["num_refund_orders_last_50days"] = 999999
        repeated = prepare_training_data(changed_customers, changed_orders, self.labels)
        for index, features in [(0, x_train), (2, x_val), (4, x_test)]:
            pd.testing.assert_frame_equal(features, repeated[index])
        self.assertEqual(report, repeated[6])
        self.assertNotIn("test_fraud", report)
        duplicate = pd.concat([self.orders, self.orders.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate composite keys"):
            prepare_training_data(self.customers, duplicate, self.labels)
        self.assertEqual(len(training_sample_weights(y_train)), len(y_train))
        with self.assertRaises(ValueError):
            training_sample_weights([0, 0])

    def test_same_date_stays_together(self):
        orders = self.orders.copy()
        orders.loc[orders.order_id == "14", "order_date"] = orders.loc[orders.order_id == "13", "order_date"].iloc[0]
        x_train, _, x_val, _, x_test, _, report = prepare_training_data(self.customers, orders, self.labels)
        self.assertLess(report["train_last_date"], report["validation_first_date"])
        self.assertLess(report["validation_last_date"], report["test_first_date"])
        self.assertEqual(len(x_train) + len(x_val) + len(x_test), 20)
        # Both rows from the boundary date must belong to the same partition.
        row_13 = self.labels.index[self.labels.order_id == "13"][0]
        row_14 = self.labels.index[self.labels.order_id == "14"][0]
        self.assertTrue(any(row_13 in part.index and row_14 in part.index for part in (x_train, x_val, x_test)))

    def test_unseen_missing_train_only_fit_and_artifact(self):
        x_train, _, x_val, _, x_test, _, _ = prepare_training_data(self.customers, self.orders, self.labels)
        transformer = fit_preprocessor(x_train)
        train_matrix = transformer.transform(x_train)
        val_matrix = transformer.transform(x_val)
        self.assertEqual(train_matrix.shape[1], val_matrix.shape[1])
        self.assertEqual(list(x_test.columns), list(x_train.columns))
        train_values = train_matrix.toarray() if sparse.issparse(train_matrix) else train_matrix
        val_values = val_matrix.toarray() if sparse.issparse(val_matrix) else val_matrix
        self.assertTrue(np.isfinite(train_values).all())
        self.assertTrue(np.isfinite(val_values).all())
        imputer = transformer.named_transformers_["amounts"].named_steps["impute"]
        self.assertEqual(imputer.statistics_[0], 1)
        encoder = transformer.named_transformers_["categories"].named_steps["onehot"]
        self.assertNotIn("unseen", encoder.categories_[2])
        with tempfile.TemporaryDirectory() as path:
            filename = path + "/preprocessor.joblib"
            joblib.dump(transformer, filename)
            restored = joblib.load(filename).transform(x_val)
            if sparse.issparse(restored):
                restored = restored.toarray()
            assert_allclose(val_values, restored)

    def test_date_missing_and_invalid(self):
        missing_dates = self.orders.copy()
        missing_dates.loc[0, "order_date"] = pd.NaT
        *_, report = prepare_training_data(self.customers, missing_dates, self.labels)
        self.assertEqual(report["matched_orders_without_dates"], 1)
        self.assertTrue(np.isnan(prepare_features(missing_dates).iloc[0]["order_weekday"]))
        invalid = self.orders.copy()
        invalid.loc[0, "order_value"] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            prepare_features(invalid)
        same_date = self.orders.copy()
        same_date["order_date"] = pd.Timestamp("2023-01-01")
        with self.assertRaisesRegex(ValueError, "empty set"):
            prepare_training_data(self.customers, same_date, self.labels)

    def test_missing_customer_is_reported(self):
        labels = self.labels.copy()
        labels.loc[labels.order_id == "0", "customer_id"] = "unknown"
        *_, report = prepare_training_data(self.customers, self.orders, labels)
        self.assertEqual(report["labels_without_customers"], 1)
        self.assertEqual(report["training_rows"] + report["validation_rows"] + report["test_rows"], 20)


if __name__ == "__main__":
    unittest.main()
