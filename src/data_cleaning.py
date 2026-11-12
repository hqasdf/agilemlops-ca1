"""Reusable PBI 02 cleaning for customer, order and label DataFrames.

Each clean function returns (retained, rejected, report). Inputs are never
modified. Rejected rows contain their original values and a rejection reason.
Missing optional features are retained; invalid features become missing and are
counted in the report. Do not use these results as model-ready features.
"""

import numpy as np
import pandas as pd


CUSTOMER_COLUMNS = [
    "country_code", "customer_id", "mobile_verified", "num_orders_last_50days",
    "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
    "total_payment_last_50days", "num_associated_customers", "first_order_datetime",
]
ORDER_COLUMNS = [
    "country_code", "order_id", "collect_type", "payment_method", "order_value",
    "num_items_ordered", "refund_value", "order_date",
]
LABEL_COLUMNS = ["country_code", "order_id", "customer_id", "is_fraud"]


def _start(data, columns):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not data.columns.is_unique:
        raise ValueError("Input has duplicate column names")
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    # Positional indices make the output reliable even with duplicate input indices.
    return data.reset_index(drop=True).copy()


def _text(data, column, case=None):
    values = data[column].astype("string").str.strip().replace("", pd.NA)
    if case == "upper":
        values = values.str.upper()
    if case == "lower":
        values = values.str.lower()
    data[column] = values


def _number(data, column, issues, integer=False):
    original = data[column]
    values = pd.to_numeric(original, errors="coerce")
    invalid = original.notna() & (values.isna() | ~np.isfinite(values))
    invalid = invalid | (values < 0).fillna(False)
    if integer:
        invalid = invalid | ((values % 1 != 0) & values.notna())
        invalid = invalid | (values >= 2**63).fillna(False)
    issues[column + "_invalid"] = int(invalid.sum())
    data[column] = values.mask(invalid).astype("Float64")
    if integer:
        data[column] = data[column].astype("Int64")


def _date(data, column, issues):
    original = data[column]
    # Naive dates have no timezone assumption; timezone-bearing dates are not
    # silently shifted and are treated as invalid for this source schema.
    parsed = pd.to_datetime(original, errors="coerce", format="mixed")
    if not pd.api.types.is_datetime64_any_dtype(parsed) or isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{column} must contain timezone-naive dates")
    issues[column + "_invalid"] = int((original.notna() & parsed.isna()).sum())
    data[column] = parsed


def _finish(original, data, key, invalid_key, issues, invalid_target=None):
    # Only raw exact duplicates are dropped; other repeated keys are quarantined.
    exact = original.duplicated(keep="first")
    conflicts = data.loc[~exact].duplicated(key, keep=False).reindex(data.index, fill_value=False)
    if invalid_target is None:
        invalid_target = pd.Series(False, index=data.index)
    reasons = pd.Series("", index=data.index, dtype="string")
    reasons.loc[invalid_key] += "invalid key; "
    reasons.loc[invalid_target] += "invalid or missing fraud label; "
    reasons.loc[exact] += "exact duplicate; "
    reasons.loc[conflicts] += "conflicting composite key; "
    reject = reasons.ne("")
    rejected = original.loc[reject].copy()
    rejected["rejection_reason"] = reasons.loc[reject].str.rstrip("; ")
    retained = data.loc[~reject].copy()
    # Compare representations across all original columns; dtype-only changes that
    # preserve the same textual value do not count as a changed record.
    before = original.loc[~reject].astype("string").fillna("<missing>")
    after = retained.astype("string").fillna("<missing>")
    changed = before.ne(after).any(axis=1)
    report = {
        "received": len(original), "changed": int(changed.sum()),
        "rejected": int(reject.sum()), "retained": len(retained),
        "exact_duplicate_rows": int(exact.sum()),
        "conflicting_key_rows": int(conflicts.sum()),
        "invalid_key_rows": int(invalid_key.sum()),
        "issues": issues,
    }
    return retained.reset_index(drop=True), rejected.reset_index(drop=True), report


def clean_customers(data):
    """Normalize customers; preserve missing features and quarantine bad keys."""
    original = _start(data, CUSTOMER_COLUMNS)
    result = original.copy()
    issues = {}
    _text(result, "country_code", "upper")
    _text(result, "customer_id")
    mobile = result["mobile_verified"].astype("string").str.strip().str.lower()
    result["mobile_verified"] = mobile.map({"true": True, "false": False}).astype("boolean")
    issues["mobile_verified_unrecognized"] = int((mobile.notna() & result["mobile_verified"].isna()).sum())
    counts = ["num_orders_last_50days", "num_cancelled_orders_last_50days",
              "num_refund_orders_last_50days", "num_associated_customers"]
    for column in counts:
        _number(result, column, issues, integer=True)
    _number(result, "total_payment_last_50days", issues)
    _date(result, "first_order_datetime", issues)
    issues["cancellations_exceed_orders"] = int((result[counts[1]] > result[counts[0]]).fillna(False).sum())
    issues["refunds_exceed_orders"] = int((result[counts[2]] > result[counts[0]]).fillna(False).sum())
    invalid_key = result["customer_id"].isna() | ~result["country_code"].str.fullmatch(r"[A-Z]{2}", na=False)
    return _finish(original, result, ["country_code", "customer_id"], invalid_key, issues)


def clean_orders(data):
    """Normalize orders; keep suspicious refunds for business-rule review."""
    original = _start(data, ORDER_COLUMNS)
    result = original.copy()
    issues = {}
    _text(result, "country_code", "upper")
    _text(result, "order_id")
    _text(result, "collect_type", "lower")
    _text(result, "payment_method")
    _number(result, "order_value", issues)
    _number(result, "num_items_ordered", issues, integer=True)
    _number(result, "refund_value", issues)
    _date(result, "order_date", issues)
    issues["refund_exceeds_order_value"] = int((result["refund_value"] > result["order_value"]).fillna(False).sum())
    invalid_key = result["order_id"].isna() | ~result["country_code"].str.fullmatch(r"[A-Z]{2}", na=False)
    return _finish(original, result, ["country_code", "order_id"], invalid_key, issues)


def clean_labels(data):
    """Normalize labels; quarantine missing or non-binary fraud outcomes."""
    original = _start(data, LABEL_COLUMNS)
    result = original.copy()
    issues = {}
    _text(result, "country_code", "upper")
    _text(result, "order_id")
    _text(result, "customer_id")
    target = pd.to_numeric(result["is_fraud"], errors="coerce")
    invalid_target = ~target.isin([0, 1])
    issues["invalid_or_missing_fraud_labels"] = int(invalid_target.sum())
    result["is_fraud"] = target.where(~invalid_target).astype("Int64")
    invalid_key = result["order_id"].isna() | result["customer_id"].isna()
    invalid_key |= ~result["country_code"].str.fullmatch(r"[A-Z]{2}", na=False)
    return _finish(original, result, ["country_code", "order_id"], invalid_key, issues, invalid_target)


def check_relationships(customers, orders, labels):
    """Return composite-key match counts and unmatched source rows.

    Call with retained outputs of the three cleaners. Unmatched rows are not
    deleted: a future batch may be partial, and an unused customer is not an error.
    """
    customer_key = ["country_code", "customer_id"]
    order_key = ["country_code", "order_id"]
    for data, key, name in [(customers, customer_key, "customers"),
                            (orders, order_key, "orders"), (labels, order_key, "labels")]:
        if data[key].isna().any().any() or data.duplicated(key).any():
            raise ValueError(f"{name} contains missing or duplicate composite keys")
    if labels[customer_key].isna().any().any():
        raise ValueError("labels contains missing customer keys")
    order_matches = labels[order_key].merge(orders[order_key], on=order_key, how="outer", indicator=True, validate="one_to_one")
    customer_matches = labels[customer_key].merge(customers[customer_key], on=customer_key, how="outer", indicator=True, validate="many_to_one")
    return {
        "order_counts": order_matches["_merge"].value_counts().to_dict(),
        "customer_counts": customer_matches["_merge"].value_counts().to_dict(),
        "labels_without_orders": order_matches.loc[order_matches["_merge"] == "left_only", order_key],
        "orders_without_labels": order_matches.loc[order_matches["_merge"] == "right_only", order_key],
        "labels_without_customers": customer_matches.loc[customer_matches["_merge"] == "left_only", customer_key],
        "customers_without_labels": customer_matches.loc[customer_matches["_merge"] == "right_only", customer_key],
    }
