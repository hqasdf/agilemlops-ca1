# PBI 03 — shared feature transformations

All three model-training PBIs use `src/feature_engineering.py`. Clean source tables with PBI 02 first. The main interface returns chronological training and validation features, labels, metadata, and no test feature/label arrays.

## Shared eight-feature schema

The raw model-input columns and their order are fixed:

1. `order_value`
2. `num_items_ordered`
3. `order_weekday`
4. `order_month`
5. `country_code`
6. `collect_type`
7. `payment_method`
8. `days_since_first_order`

Identifiers and `is_fraud` are join/target fields, never model inputs. `refund_value` and undated customer aggregates are excluded because they may reveal post-order outcomes or future customer history.

`prepare_training_data(customers, orders, labels)` validates unique composite keys, joins labels to orders one-to-one on `(country_code, order_id)`, and joins customer history many-to-one on `(country_code, customer_id)`. It rejects sealed-period orders before the feature join. Whole order dates remain in a single partition. The frozen validation boundary is 15 March 2023; training ends 14 March 2023 and validation ends 13 April 2023. Test features and labels are not returned or summarized.

`prepare_features(orders, customers)` is the inference-side builder. It requires each order row to include a customer ID obtained independently of the fraud-label file. The current order CSV does not have `customer_id`; the training labels provide that relation for historical examples. PBI 10 must supply a trusted order-to-customer mapping at scoring time. If it cannot, `days_since_first_order` is not available for prediction and the eight-feature pipeline cannot be served as specified.

Payment methods and collection types are lowercased in the shared feature builder. `days_since_first_order` is the normalized calendar-day difference between the order date and `first_order_datetime`. Missing, invalid, or future first-order timestamps become missing, never negative durations. The source calls this the first recorded order, not verified account-creation time. The mask is logically point-in-time safe if the source is a complete historical first-order timestamp; the dataset does not establish when that customer field becomes available. Confirm its provenance before production use.

## Shared preprocessing

`build_preprocessor()` creates the only preprocessing definition; `fit_preprocessor(train_features)` validates the column order and fits it on training records only. Every member must transform validation/inference data with that fitted artifact.

- Amount/count/history numeric features: training median imputation, `log1p`, then training-fitted `RobustScaler`.
- Weekday/month: training median imputation.
- Categorical values: explicit missing category and one-hot encoding fitted on training data; unseen categories are ignored safely.
- A dense matrix conversion is allowed only as a model input storage adaptation for estimators that cannot consume sparse matrices. It does not change the shared feature meaning.

Do not refit preprocessing on validation or future rows. For PBI 06/07 comparisons, the latest 20% of whole outer-training dates is reserved for threshold calibration; the model fits on the older dates, and that same fitted model scores calibration and validation. Thresholds maximize fraud-class F1 on training-only calibration predictions. Historical PBI 06 runs used their previously documented protocols and remain historical, not shared eight-feature results.

## Usage

```python
from src.data_access import load_development_tables, VALIDATION_START
from src.data_cleaning import clean_customers, clean_orders, clean_labels
from src.feature_engineering import INPUT_FEATURES, prepare_training_data, fit_preprocessor

customers_raw, orders_raw, labels_raw = load_development_tables()
customers = clean_customers(customers_raw)[0]
orders = clean_orders(orders_raw)[0]
labels = clean_labels(labels_raw)[0]
x_train, y_train, x_val, y_val, _, _, split_report, train_dates = prepare_training_data(
    customers, orders, labels, include_test=False, include_training_dates=True,
    validation_start_date=VALIDATION_START,
)
preprocessor = fit_preprocessor(x_train)
train_matrix = preprocessor.transform(x_train)
validation_matrix = preprocessor.transform(x_val)
assert list(x_train.columns) == INPUT_FEATURES
```

The latest 15% test partition remains unavailable to development. Do not use test labels or metrics during development. PBI 10 must resolve the customer-ID serving source and first-order field timing before claiming that the eighth feature is production-safe.
