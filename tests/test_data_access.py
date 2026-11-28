import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_access import load_development_tables, read_dataset_csv
from src.data_cleaning import clean_customers, clean_orders, clean_labels
from src.feature_engineering import prepare_training_data, fit_preprocessor


class DevelopmentDataAccessTests(unittest.TestCase):
    def test_schema_reader_preserves_keys_country_na_and_missing_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "orders_042.csv"
            path.write_text(
                "country_code,order_id,collect_type,payment_method,order_value,num_items_ordered,refund_value,order_date\n"
                "NA,0007,Delivery,,12,1,0,2023-01-01\n",
                encoding="utf-8",
            )
            rows = read_dataset_csv(path)
            self.assertEqual(rows.loc[0, "country_code"], "NA")
            self.assertEqual(rows.loc[0, "order_id"], "0007")
            self.assertTrue(pd.isna(rows.loc[0, "payment_method"]))
            cleaned, rejected, report = clean_orders(rows)
            self.assertEqual(cleaned.loc[0, "order_id"], "0007")
            self.assertEqual(cleaned.loc[0, "country_code"], "NA")
            self.assertEqual(len(rejected), 0)
            self.assertEqual(report["received"], 1)

    def test_reader_rejects_empty_missing_duplicate_and_malformed_csvs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            empty = root / "orders_1.csv"
            empty.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty or has no header"):
                read_dataset_csv(empty)

            missing = root / "orders_2.csv"
            missing.write_text("country_code,order_id\nNA,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                read_dataset_csv(missing)

            duplicate = root / "orders_3.csv"
            duplicate.write_text(
                "country_code,order_id,order_id,collect_type,payment_method,order_value,num_items_ordered,refund_value,order_date\n"
                "NA,1,1,delivery,cash,2,1,0,2023-01-01\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate column"):
                read_dataset_csv(duplicate)

            malformed = root / "orders_4.csv"
            malformed.write_text(
                "country_code,order_id,collect_type,payment_method,order_value,num_items_ordered,refund_value,order_date\n"
                "NA,1,delivery,cash,2,1,0,2023-01-01,extra\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Malformed CSV row"):
                read_dataset_csv(malformed)

            header_only = root / "orders_5.csv"
            header_only.write_text(
                "country_code,order_id,collect_type,payment_method,order_value,num_items_ordered,refund_value,order_date\n",
                encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "no data rows"):
                read_dataset_csv(header_only)

    def test_repeat_import_is_deterministic_and_batch_overlap_is_not_a_database_write(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels_1.csv"
            path.write_text(
                "country_code,order_id,customer_id,is_fraud\nNA,0008,0002,1\n",
                encoding="utf-8",
            )
            first = read_dataset_csv(path)
            second = read_dataset_csv(path)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(first.loc[0, "order_id"], "0008")

    def test_synthetic_batch_flows_through_features_and_all_seven_model_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            customer_path = root / "customers_demo.csv"
            orders_path = root / "orders_demo.csv"
            labels_path = root / "labels_demo.csv"
            pd.DataFrame([{
                "country_code": "NA", "customer_id": "0002", "mobile_verified": "true",
                "num_orders_last_50days": 8, "num_cancelled_orders_last_50days": 0,
                "num_refund_orders_last_50days": 0, "total_payment_last_50days": 20,
                "num_associated_customers": 1, "first_order_datetime": "2023-01-01",
            }]).to_csv(customer_path, index=False)
            order_rows = []
            label_rows = []
            for number in range(8):
                order_rows.append({
                    "country_code": "NA", "order_id": f"{number:04d}",
                    "collect_type": "Delivery", "payment_method": "new_wallet" if number > 4 else "cash",
                    "order_value": number + 1, "num_items_ordered": 1,
                    "refund_value": 0, "order_date": f"2023-01-{number + 1:02d}",
                })
                label_rows.append({
                    "country_code": "NA", "order_id": f"{number:04d}",
                    "customer_id": "0002", "is_fraud": number % 2,
                })
            pd.DataFrame(order_rows).to_csv(orders_path, index=False)
            pd.DataFrame(label_rows).to_csv(labels_path, index=False)

            customers = clean_customers(read_dataset_csv(customer_path))[0]
            orders = clean_orders(read_dataset_csv(orders_path))[0]
            labels = clean_labels(read_dataset_csv(labels_path))[0]
            x_train, y_train, x_val, y_val, _, _, report = prepare_training_data(
                customers, orders, labels
            )
            preprocessor = fit_preprocessor(x_train)
            train_matrix = preprocessor.transform(x_train)
            val_matrix = preprocessor.transform(x_val)
            self.assertEqual(train_matrix.shape[1], val_matrix.shape[1])
            self.assertEqual(report["training_rows"], len(y_train))
            self.assertEqual(report["validation_rows"], len(y_val))

            from src.train_hq import model_candidates
            model_names = set(model_candidates())
            self.assertEqual(model_names, {
                "logistic_regression", "random_forest", "xgboost", "lightgbm",
                "catboost", "hist_gradient_boosting", "gradient_boosting",
            })

    def test_csv_reader_skips_sealed_fields_and_preserves_identifiers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            customers_path = root / "customers.csv"
            orders_path = root / "orders.csv"
            labels_path = root / "labels.csv"
            manifest_path = root / "sealed.sha256"

            pd.DataFrame([
                {"country_code": "NA", "customer_id": "0007", "mobile_verified": "true",
                 "num_orders_last_50days": 1, "num_cancelled_orders_last_50days": 0,
                 "num_refund_orders_last_50days": 0, "total_payment_last_50days": 5,
                 "num_associated_customers": 1, "first_order_datetime": "2023-01-01"},
                {"country_code": "MY", "customer_id": "sealed-customer", "mobile_verified": "???",
                 "num_orders_last_50days": "not-a-number", "num_cancelled_orders_last_50days": 0,
                 "num_refund_orders_last_50days": 0, "total_payment_last_50days": 5,
                 "num_associated_customers": 1, "first_order_datetime": "bad-date"},
            ]).to_csv(customers_path, index=False)
            pd.DataFrame([
                {"country_code": "NA", "order_id": "0009", "collect_type": "delivery",
                 "payment_method": "cash", "order_value": 10, "num_items_ordered": 1,
                 "refund_value": 0, "order_date": "2023-03-01"},
                {"country_code": "MY", "order_id": "sealed-order", "collect_type": "must-not-load",
                 "payment_method": "must-not-load", "order_value": "invalid sealed feature",
                 "num_items_ordered": "invalid", "refund_value": "invalid", "order_date": "2023-04-14"},
            ]).to_csv(orders_path, index=False)
            labels_path.write_text(
                "country_code,order_id,customer_id,is_fraud\n"
                "NA,0009,0007,1\n"
                "MY,sealed-order,sealed-customer,not-a-label-to-read\n",
                encoding="utf-8",
            )

            customers, orders, labels = load_development_tables(
                customers_path, orders_path, labels_path, manifest_path
            )

            self.assertEqual(customers.loc[0, "country_code"], "NA")
            self.assertEqual(customers.loc[0, "customer_id"], "0007")
            self.assertEqual(orders.loc[0, "country_code"], "NA")
            self.assertEqual(orders.loc[0, "order_id"], "0009")
            self.assertEqual(labels.loc[0, "order_id"], "0009")
            self.assertEqual(len(orders), 1)
            self.assertEqual(len(labels), 1)
            self.assertNotIn("sealed-order", set(orders["order_id"]))
            self.assertNotIn("not-a-label-to-read", set(labels["is_fraud"]))
            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
