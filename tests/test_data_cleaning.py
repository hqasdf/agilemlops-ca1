import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data_cleaning import (clean_customers, clean_orders, clean_labels,
                               check_relationships, compare_batch_keys)


def customer(**changes):
    row = {"country_code": "my", "customer_id": "c1", "mobile_verified": "true",
           "num_orders_last_50days": 3, "num_cancelled_orders_last_50days": 0,
           "num_refund_orders_last_50days": 0, "total_payment_last_50days": 10,
           "num_associated_customers": 1, "first_order_datetime": "2022/01/02"}
    row.update(changes)
    return row


def order(**changes):
    row = {"country_code": "my", "order_id": "o1", "collect_type": "Delivery",
           "payment_method": None, "order_value": 10, "num_items_ordered": 1,
           "refund_value": 12, "order_date": "2023/01/02"}
    row.update(changes)
    return row


def label(**changes):
    row = {"country_code": "my", "order_id": "o1", "customer_id": "c1", "is_fraud": 1}
    row.update(changes)
    return row


class CleaningTests(unittest.TestCase):
    def test_schema_errors(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            clean_orders(pd.DataFrame({"order_id": ["a"]}))
        with self.assertRaisesRegex(TypeError, "DataFrame"):
            clean_customers([])

    def test_normalization_and_ambiguous_values(self):
        raw = pd.DataFrame([customer(mobile_verified="yes", num_orders_last_50days="bad",
                                     first_order_datetime="bad-date")])
        original = raw.copy(deep=True)
        cleaned, rejected, report = clean_customers(raw)
        assert_frame_equal(raw, original)
        self.assertEqual(cleaned.loc[0, "country_code"], "MY")
        self.assertTrue(pd.isna(cleaned.loc[0, "mobile_verified"]))
        self.assertTrue(pd.isna(cleaned.loc[0, "num_orders_last_50days"]))
        self.assertTrue(pd.isna(cleaned.loc[0, "first_order_datetime"]))
        self.assertEqual(report["issues"]["mobile_verified_unrecognized"], 1)
        self.assertEqual(report["issues"]["num_orders_last_50days_invalid"], 1)
        self.assertEqual(report["issues"]["first_order_datetime_invalid"], 1)
        self.assertEqual(len(rejected), 0)

    def test_numeric_dates_and_suspicious_refund(self):
        raw = pd.DataFrame([order(), order(order_id="o2", order_value=-2,
                                          num_items_ordered=1.5, order_date="2023-02-30")])
        cleaned, rejected, report = clean_orders(raw)
        self.assertEqual(cleaned.loc[0, "collect_type"], "delivery")
        self.assertEqual(cleaned.loc[0, "refund_value"], 12)
        self.assertTrue(pd.isna(cleaned.loc[1, "order_value"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "num_items_ordered"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "order_date"]))
        self.assertEqual(report["issues"]["refund_exceeds_order_value"], 1)
        self.assertEqual(len(rejected), 0)

    def test_duplicates_conflicts_and_country_keys(self):
        rows = [order(), order(), order(order_id="same", country_code="MY"),
                order(order_id="same", country_code="PH"),
                order(order_id="conflict", order_value=4),
                order(order_id="conflict", order_value=5), order(order_id=None)]
        cleaned, rejected, report = clean_orders(pd.DataFrame(rows))
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(len(rejected), 4)
        self.assertEqual(report["exact_duplicate_rows"], 1)
        self.assertEqual(report["conflicting_key_rows"], 2)
        self.assertEqual(report["invalid_key_rows"], 1)
        self.assertEqual(set(cleaned.loc[cleaned.order_id == "same", "country_code"]), {"MY", "PH"})

    def test_labels_and_relationships(self):
        customers, _, _ = clean_customers(pd.DataFrame([customer(), customer(customer_id="unused")]))
        orders, _, _ = clean_orders(pd.DataFrame([order(), order(order_id="unlabeled")]))
        labels, rejected, report = clean_labels(pd.DataFrame([
            label(), label(order_id="missing_order"), label(order_id="missing_customer", customer_id="other"),
            label(order_id="bad", is_fraud=2)]))
        self.assertEqual(len(rejected), 1)
        self.assertEqual(report["issues"]["invalid_or_missing_fraud_labels"], 1)
        links = check_relationships(customers, orders, labels)
        self.assertEqual(links["order_counts"]["left_only"], 2)
        self.assertEqual(links["order_counts"]["right_only"], 1)
        self.assertEqual(links["customer_counts"]["left_only"], 1)
        self.assertEqual(links["customer_counts"]["right_only"], 1)
        self.assertEqual(len(links["labels_without_customers"]), 1)

    def test_reproducible(self):
        raw = pd.DataFrame([order(), order(order_id="2", order_value="invalid")])
        first = clean_orders(raw)
        second = clean_orders(raw)
        assert_frame_equal(first[0], second[0])
        assert_frame_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_reports_missing_values_and_flagged_identifiers(self):
        cleaned, rejected, report = clean_orders(pd.DataFrame([
            order(order_id="flagged", refund_value=12, order_value=10,
                  payment_method="new_wallet"),
            order(order_id="missing", payment_method=None, order_date="bad date", refund_value=0),
        ]))
        self.assertEqual(report["missing_values"]["payment_method"], 1)
        self.assertEqual(report["category_values"]["payment_method"], ["new_wallet"])
        self.assertEqual(report["issues"]["refund_exceeds_order_value"], 1)
        self.assertEqual(report["flagged_records"][0]["order_id"], "flagged")
        self.assertIn("order_date_invalid", report["flagged_records"][1]["issues"])
        self.assertEqual(len(cleaned), 2)
        self.assertTrue(rejected.empty)

    def test_new_categories_are_retained(self):
        cleaned, _, report = clean_orders(pd.DataFrame([
            order(collect_type="Locker", payment_method="new_wallet")
        ]))
        self.assertEqual(cleaned.loc[0, "collect_type"], "locker")
        self.assertEqual(cleaned.loc[0, "payment_method"], "new_wallet")
        self.assertEqual(report["category_values"]["collect_type"], ["locker"])
        self.assertEqual(report["category_values"]["payment_method"], ["new_wallet"])

    def test_historical_overlap_is_reported_for_pipeline_reconciliation(self):
        incoming = pd.DataFrame([{"country_code": "NA", "order_id": "0001"},
                                 {"country_code": "MY", "order_id": "new"}])
        historical = pd.DataFrame([{"country_code": "NA", "order_id": "0001"}])
        result = compare_batch_keys(incoming, historical, ["country_code", "order_id"])
        self.assertEqual(result["already_in_history_count"], 1)
        self.assertEqual(result["new_key_count"], 1)
        self.assertEqual(result["already_in_history"].loc[0, "order_id"], "0001")
        self.assertEqual(result["new_keys"].loc[0, "order_id"], "new")
        with self.assertRaisesRegex(ValueError, "duplicate keys"):
            compare_batch_keys(pd.concat([incoming, incoming.iloc[[0]]]), historical,
                               ["country_code", "order_id"])


if __name__ == "__main__":
    unittest.main()
