"""Shared, leakage-conscious feature preparation for fraud classification.

Inputs are retained DataFrames from PBI 02 cleaning. No model is trained here.
Use prepare_training_data, fit_preprocessor, then the returned preprocessor's
transform method. For later orders, call prepare_features before transform.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight


NUMERIC_FEATURES = ["order_value", "num_items_ordered"]
DATE_FEATURES = ["order_weekday", "order_month"]
CATEGORICAL_FEATURES = ["country_code", "collect_type", "payment_method"]
INPUT_FEATURES = NUMERIC_FEATURES + DATE_FEATURES + CATEGORICAL_FEATURES
ORDER_KEY = ["country_code", "order_id"]
CUSTOMER_KEY = ["country_code", "customer_id"]


def _check_keys(data, keys, name):
    missing = [column for column in keys if column not in data.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if data[keys].isna().any().any() or data.duplicated(keys).any():
        raise ValueError(f"{name} has missing or duplicate composite keys: {keys}")


def prepare_features(orders):
    """Build allowed order-time features for training or future inference.

    Pass cleaned orders. IDs, refund information and customer snapshots are
    never selected. Missing dates are allowed here and imputed by the fitted
    pipeline; training/evaluation splitting requires a known order date.
    """
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["order_date"]
    missing = [column for column in required if column not in orders.columns]
    if missing:
        raise ValueError(f"Orders are missing feature columns: {missing}")
    features = orders[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="raise").astype(float)
        values = features[column].dropna()
        if (values < 0).any() or not np.isfinite(values).all():
            raise ValueError(f"{column} must contain nonnegative finite values or missing values")
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype(object).where(features[column].notna(), np.nan)
    dates = pd.to_datetime(orders["order_date"], errors="raise")
    if not pd.api.types.is_datetime64_any_dtype(dates) or isinstance(dates.dtype, pd.DatetimeTZDtype):
        raise ValueError("order_date must contain timezone-naive dates")
    features["order_weekday"] = dates.dt.day_of_week.astype(float)
    features["order_month"] = dates.dt.month.astype(float)
    return features[INPUT_FEATURES]


def prepare_training_data(customers, orders, labels):
    """Join cleaned tables, split chronologically and return features and counts.

    Labels with no order or no valid order date are excluded and counted.
    Customers are checked but their snapshot values are never used as features.
    Rows with missing customer matches remain because order features are usable.
    Whole dates are assigned to training, validation or test at the 70th
    and 85th percentile rows. Fractions can differ when many orders share a date.
    Test labels are returned for final evaluation and are not summarized here.
    """
    _check_keys(customers, CUSTOMER_KEY, "customers")
    _check_keys(orders, ORDER_KEY, "orders")
    _check_keys(labels, ORDER_KEY, "labels")
    if labels[CUSTOMER_KEY].isna().any().any():
        raise ValueError("labels has missing customer keys")
    if "is_fraud" not in labels:
        raise ValueError("labels must contain is_fraud")

    rows = labels.merge(orders, on=ORDER_KEY, how="left", validate="one_to_one", indicator="order_match")
    rows = rows.merge(customers[CUSTOMER_KEY], on=CUSTOMER_KEY, how="left", validate="many_to_one", indicator="customer_match")
    unmatched_orders = int((rows["order_match"] == "left_only").sum())
    unmatched_customers = int((rows["customer_match"] == "left_only").sum())
    rows = rows.loc[rows["order_match"] == "both"].copy()
    dates = pd.to_datetime(rows["order_date"], errors="raise")
    if not pd.api.types.is_datetime64_any_dtype(dates) or isinstance(dates.dtype, pd.DatetimeTZDtype):
        raise ValueError("order_date must contain timezone-naive dates")
    missing_dates = int(dates.isna().sum())
    rows = rows.loc[dates.notna()].copy()
    rows["order_date"] = dates.loc[dates.notna()].dt.normalize()
    if len(rows) < 3:
        raise ValueError("At least three dated matched orders are needed")
    sorted_dates = rows["order_date"].sort_values().reset_index(drop=True)
    validation_start = sorted_dates.iloc[int(len(sorted_dates) * 0.70)]
    test_start = sorted_dates.iloc[int(len(sorted_dates) * 0.85)]
    train_rows = rows.loc[rows["order_date"] < validation_start]
    val_rows = rows.loc[(rows["order_date"] >= validation_start) & (rows["order_date"] < test_start)]
    test_rows = rows.loc[rows["order_date"] >= test_start]
    if train_rows.empty or val_rows.empty or test_rows.empty:
        raise ValueError("The 70/15/15 date split produced an empty set; more varied dates are needed")
    if not train_rows["is_fraud"].isin([0, 1]).all() or not val_rows["is_fraud"].isin([0, 1]).all():
        raise ValueError("Training and validation labels must be nonmissing binary values")

    train_customers = pd.MultiIndex.from_frame(train_rows[CUSTOMER_KEY]).unique()
    val_customers = pd.MultiIndex.from_frame(val_rows[CUSTOMER_KEY]).unique()
    test_customers = pd.MultiIndex.from_frame(test_rows[CUSTOMER_KEY]).unique()
    report = {
        "labels_received": len(labels), "labels_without_orders": unmatched_orders,
        "labels_without_customers": unmatched_customers, "matched_orders_without_dates": missing_dates,
        "training_rows": len(train_rows), "validation_rows": len(val_rows), "test_rows": len(test_rows),
        "train_first_date": str(train_rows["order_date"].min().date()),
        "train_last_date": str(train_rows["order_date"].max().date()),
        "validation_first_date": str(val_rows["order_date"].min().date()),
        "validation_last_date": str(val_rows["order_date"].max().date()),
        "test_first_date": str(test_rows["order_date"].min().date()),
        "test_last_date": str(test_rows["order_date"].max().date()),
        "train_validation_customers": len(train_customers.intersection(val_customers)),
        "train_test_customers": len(train_customers.intersection(test_customers)),
        "validation_test_customers": len(val_customers.intersection(test_customers)),
    }
    return (prepare_features(train_rows), train_rows["is_fraud"].astype(int),
            prepare_features(val_rows), val_rows["is_fraud"].astype(int),
            prepare_features(test_rows), test_rows["is_fraud"], report)


def fit_preprocessor(train_features):
    """Fit imputation, scaling and encoding on training features only."""
    if list(train_features.columns) != INPUT_FEATURES:
        raise ValueError(f"Expected feature columns in this order: {INPUT_FEATURES}")
    amounts = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", RobustScaler()),
    ])
    dates = SimpleImputer(strategy="median", keep_empty_features=True)
    categories = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    preprocessor = ColumnTransformer([
        ("amounts", amounts, NUMERIC_FEATURES),
        ("dates", dates, DATE_FEATURES),
        ("categories", categories, CATEGORICAL_FEATURES),
    ], sparse_threshold=1.0)
    preprocessor.fit(train_features)
    return preprocessor


def training_sample_weights(y_train):
    """Return balanced weights computed from training labels only."""
    if set(pd.Series(y_train).dropna().unique()) != {0, 1} or pd.isna(y_train).any():
        raise ValueError("Training labels must contain both binary classes")
    return compute_sample_weight(class_weight="balanced", y=y_train)
