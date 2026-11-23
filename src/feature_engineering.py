"""Shared eight-feature preparation and training-only preprocessing for CA1."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight


ORDER_KEY = ["country_code", "order_id"]
CUSTOMER_KEY = ["country_code", "customer_id"]

NUMERIC_FEATURES = [
    "order_value", "num_items_ordered", "days_since_first_order",
]
DATE_FEATURES = ["order_weekday", "order_month"]
CATEGORICAL_FEATURES = ["country_code", "collect_type", "payment_method"]
INPUT_FEATURES = [
    "order_value", "num_items_ordered", "order_weekday", "order_month",
    "country_code", "collect_type", "payment_method", "days_since_first_order",
]


def _check_keys(data, keys, name):
    missing = [column for column in keys if column not in data.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if data[keys].isna().any().any() or data.duplicated(keys).any():
        raise ValueError(f"{name} has missing or duplicate composite keys: {keys}")


def _dates(values, column):
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{column} must contain timezone-naive dates")
    return parsed.dt.normalize()


def prepare_features(orders, customers):
    """Join cleaned customer history and return the shared ordered schema.

    A first-order timestamp is used only when it is on or before the order date.
    Missing, invalid and future timestamps become missing values. The feature is
    safe only if ``first_order_datetime`` is an as-of historical first-order
    value; confirm its source definition before production use.
    """
    required = [
        "country_code", "order_id", "customer_id", "order_value",
        "num_items_ordered", "collect_type", "payment_method", "order_date",
    ]
    missing = [column for column in required if column not in orders.columns]
    if missing:
        raise ValueError(f"Orders are missing required feature/join columns: {missing}")
    if "first_order_datetime" not in customers.columns:
        raise ValueError("Customers are missing required column: first_order_datetime")

    _check_keys(orders, ORDER_KEY, "orders")
    _check_keys(customers, CUSTOMER_KEY, "customers")
    order_rows = orders.reset_index(drop=True).copy()
    order_rows["_source_position"] = np.arange(len(order_rows))
    customer_history = customers[CUSTOMER_KEY + ["first_order_datetime"]]
    joined = order_rows.merge(
        customer_history, on=CUSTOMER_KEY, how="left", validate="many_to_one",
        indicator="customer_match",
    ).sort_values("_source_position")

    order_dates = _dates(joined["order_date"], "order_date")
    first_dates = _dates(joined["first_order_datetime"], "first_order_datetime")
    days = (order_dates - first_dates).dt.days.astype(float)
    joined["days_since_first_order"] = days.where(days >= 0)

    features = joined[[
        "order_value", "num_items_ordered", "order_date", "country_code",
        "collect_type", "payment_method", "days_since_first_order",
    ]].copy()
    for column in ("order_value", "num_items_ordered"):
        original = features[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & (numeric.isna() | ~np.isfinite(numeric))
        invalid |= (numeric < 0).fillna(False)
        if invalid.any():
            raise ValueError(f"{column} must be cleaned to nonnegative finite values before feature preparation")
        features[column] = numeric.astype(float)
    dates = _dates(features.pop("order_date"), "order_date")
    features["order_weekday"] = dates.dt.day_of_week.astype(float)
    features["order_month"] = dates.dt.month.astype(float)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("string").str.strip()
    features["country_code"] = features["country_code"].str.upper()
    features["collect_type"] = features["collect_type"].str.lower()
    features["payment_method"] = features["payment_method"].str.lower()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype(object).where(features[column].notna(), np.nan)
    features.index = orders.index
    return features[INPUT_FEATURES]


def prepare_training_data(customers, orders, labels, include_test=False,
                          include_training_dates=False, validation_start_date=None,
                          sealed_order_keys=None, new_order_keys=None):
    """Safely join development records and return chronological train/validation data.

    By default, reject all orders in the sealed period. The optional retraining
    path requires a trusted sealed-key manifest and an allowlist of confirmed
    new order keys; sealed orders are removed before any feature join. The
    caller owns manifest provenance and must persist previously accepted new
    keys. Whole dates stay in one partition. The historical validation boundary
    can be supplied to reproduce the saved validation period.
    """
    _check_keys(customers, CUSTOMER_KEY, "customers")
    _check_keys(orders, ORDER_KEY, "orders")
    _check_keys(labels, ORDER_KEY, "labels")
    if labels[CUSTOMER_KEY].isna().any().any():
        raise ValueError("labels has missing customer keys")
    if "is_fraud" not in labels:
        raise ValueError("labels must contain is_fraud")

    order_dates = _dates(orders["order_date"], "order_date")
    if sealed_order_keys is not None or new_order_keys is not None:
        from src.data_access import _key_hash
        if not sealed_order_keys or new_order_keys is None:
            raise ValueError('A verified historical holdout and new-order allowlist are required')
        if set(sealed_order_keys).intersection(new_order_keys):
            raise ValueError('A new order cannot be a sealed historical order')
        hashes = pd.Series([_key_hash(row.country_code, row.order_id) for row in orders.itertuples()], index=orders.index)
        unknown = (order_dates >= pd.Timestamp('2023-04-14')) & ~hashes.isin(sealed_order_keys) & ~hashes.isin(new_order_keys)
        if unknown.any():
            raise ValueError('Later orders include keys outside the verified new-order allowlist')
        allowed = ~hashes.isin(sealed_order_keys)
        orders = orders.loc[allowed].copy()
        order_dates = order_dates.loc[allowed]
        later_allowed = hashes.loc[allowed].isin(new_order_keys)
    else:
        later_allowed = pd.Series(False, index=orders.index)
    if ((order_dates >= pd.Timestamp("2023-04-14")) & ~later_allowed).any():
        raise ValueError("Input includes sealed test-period orders; use the development-only CSV loader")
    if include_test:
        raise ValueError("Test data are sealed and cannot be returned by this function")

    label_rows = labels.reset_index(drop=True).copy()
    order_rows = orders.reset_index(drop=True).copy()
    customer_rows = customers.reset_index(drop=True).copy()
    rows = label_rows.merge(
        order_rows, on=ORDER_KEY, how="left", validate="one_to_one", indicator="order_match",
    )
    customer_match = rows[CUSTOMER_KEY].merge(
        customer_rows[CUSTOMER_KEY], on=CUSTOMER_KEY, how="left", validate="many_to_one",
        indicator=True,
    )
    unmatched_orders = int((rows["order_match"] == "left_only").sum())
    unmatched_customers = int((customer_match["_merge"] == "left_only").sum())
    rows = rows.loc[rows["order_match"] == "both"].copy()
    dates = _dates(rows["order_date"], "order_date")
    missing_dates = int(dates.isna().sum())
    rows = rows.loc[dates.notna()].copy()
    rows["order_date"] = dates.loc[dates.notna()]
    if len(rows) < 3:
        raise ValueError("At least three dated matched orders are needed")

    sorted_dates = rows["order_date"].sort_values().reset_index(drop=True)
    if validation_start_date is None:
        validation_start = sorted_dates.iloc[int(len(sorted_dates) * 0.70)]
    else:
        validation_start = pd.Timestamp(validation_start_date)
    train_rows = rows.loc[rows["order_date"] < validation_start].copy()
    val_rows = rows.loc[rows["order_date"] >= validation_start].copy()
    if train_rows.empty or val_rows.empty:
        raise ValueError("The chronological train/validation split produced an empty set")
    if not train_rows["is_fraud"].isin([0, 1]).all() or not val_rows["is_fraud"].isin([0, 1]).all():
        raise ValueError("Training and validation labels must be nonmissing binary values")

    train_customers = pd.MultiIndex.from_frame(train_rows[CUSTOMER_KEY]).unique()
    val_customers = pd.MultiIndex.from_frame(val_rows[CUSTOMER_KEY]).unique()
    report = {
        "labels_received": len(labels), "labels_without_orders": unmatched_orders,
        "labels_without_customers": unmatched_customers, "matched_orders_without_dates": missing_dates,
        "training_rows": len(train_rows), "validation_rows": len(val_rows),
        "train_first_date": str(train_rows["order_date"].min().date()),
        "train_last_date": str(train_rows["order_date"].max().date()),
        "validation_first_date": str(val_rows["order_date"].min().date()),
        "validation_last_date": str(val_rows["order_date"].max().date()),
        "train_validation_customers": len(train_customers.intersection(val_customers)),
    }
    if new_order_keys is not None:
        report['new_training_rows'] = sum(_key_hash(row.country_code, row.order_id) in new_order_keys for row in train_rows.itertuples())
        report['new_validation_rows'] = sum(_key_hash(row.country_code, row.order_id) in new_order_keys for row in val_rows.itertuples())
    if include_training_dates:
        report["training_dates"] = train_rows["order_date"].copy()
    train_dates = train_rows["order_date"].copy()
    result = (
        prepare_features(train_rows, customer_rows), train_rows["is_fraud"].astype(int),
        prepare_features(val_rows, customer_rows), val_rows["is_fraud"].astype(int),
        None, None, report,
    )
    if include_training_dates:
        result += (train_dates,)
    return result


def build_preprocessor():
    """Return the single shared, unfitted PBI 03 transformation pipeline."""
    amounts = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", RobustScaler()),
    ])
    calendar = SimpleImputer(strategy="median", keep_empty_features=True)
    categories = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("amounts", amounts, NUMERIC_FEATURES),
        ("calendar", calendar, DATE_FEATURES),
        ("categories", categories, CATEGORICAL_FEATURES),
    ], sparse_threshold=1.0)


def fit_preprocessor(train_features):
    """Fit the shared imputer, transforms and encoder on training rows only."""
    if list(train_features.columns) != INPUT_FEATURES:
        raise ValueError(f"Expected feature columns in this order: {INPUT_FEATURES}")
    return build_preprocessor().fit(train_features)


def training_sample_weights(y_train):
    """Return balanced sample weights calculated from training labels only."""
    if set(pd.Series(y_train).dropna().unique()) != {0, 1} or pd.isna(y_train).any():
        raise ValueError("Training labels must contain both binary classes")
    return compute_sample_weight(class_weight="balanced", y=y_train)


