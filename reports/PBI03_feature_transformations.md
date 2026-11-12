# PBI 03 — feature transformation record

Target: `is_fraud` (0 = non-fraud, 1 = fraud), from cleaned labels. One example is one labeled order. The code is in `src/feature_engineering.py`; apply PBI 02 cleaning before using it. The retained cleaner outputs are the only inputs to the shared interface.

| Input / derived columns | Decision and rationale | Code |
| --- | --- | --- |
| `country_code`, `order_id`, `customer_id` | Normalize/validate in PBI 02. Join labels to orders on country + order ID (one-to-one) and to customers on country + customer ID (many-to-one). IDs are excluded from features. Country is retained as a categorical feature, assuming it is known when the order is placed. | `prepare_training_data`, `prepare_features` |
| `order_value`, `num_items_ordered` | Numeric inputs known at checkout, subject to confirming the operational decision time. Reject negative/nonfinite inputs at this interface; use the training median for missing values, `log1p` for skew, then training-fitted robust scaling. Do not clip or drop valid extremes. | `prepare_features`, `fit_preprocessor` |
| `order_date` | Derive weekday (Monday = 0) and month. The raw date is used to split and is not passed to the model. Medians are learned from training for missing derived values at inference; undated records are excluded from the split and counted. These calendar features assume order date is known at prediction. | `prepare_training_data`, `prepare_features`, `fit_preprocessor` |
| `country_code`, `collect_type`, `payment_method` | Missing values get a distinct `__missing__` category; one-hot encoding is fitted only on training. Unseen categories produce all zeros for that field. Verify collection and payment method are known before review. | `prepare_features`, `fit_preprocessor` |
| `refund_value`, `is_fraud` | Exclude refund value because it may occur after prediction; the target is never an input feature. | `prepare_features` |
| All customer fields, including `first_order_datetime`, `mobile_verified` and 50-day aggregates | Exclude: this file is a snapshot without timestamps proving historical values were available per order. Even account-age features could leak if snapshot provenance is unknown. Joins are checked for coverage only. | `prepare_training_data`, `prepare_features` |
| Class imbalance | `training_sample_weights(y_train)` provides balanced weights using training labels only, for a later compatible estimator. There is no resampling, and validation and test keep their natural class distributions. Do not also apply separate class weights without deliberate review. Assess precision, recall and PR performance in the later model PBIs. | `training_sample_weights` |

## Reproducible split and base-data result

Sort matched labeled orders by date. The dates at the 70th and 85th percentile rows start validation and test respectively. All orders on the same date remain together, so proportions are approximate. A split that produces an empty set raises an error. Missing/unmatched orders and matched orders with missing dates are reported and excluded; missing customer matches are reported but retained because no customer values enter the model.

On the supplied data, 444 label rows had no order, 14 lacked a customer match and no matched orders lacked a date:

| Set | Rows | Share | Order dates |
| --- | ---: | ---: | --- |
| Training | 660,773 | 69.56% | 2022-11-12 to 2023-03-14 |
| Validation | 144,978 | 15.26% | 2023-03-15 to 2023-04-13 |
| Test | 144,123 | 15.17% | 2023-04-14 to 2023-05-11 |

Fit preprocessing and model parameters on training only. Use validation for model comparison, tuning and champion selection. Preserve test labels and the prepared test feature frame without inspection, fitted preprocessing or model evaluation until the final champion is selected; then use the already fitted preprocessing and chosen model once for final evaluation. The split report deliberately does not summarize test labels or metrics. PBI 02 validates the source labels as part of cleaning before the split.

There are 42,411 customers in both training and validation, 38,474 in both training and test, and 17,660 in both validation and test. This evaluates later orders from a mixture of existing and new customers, not generalization to entirely new customers; consider a separate customer-held-out evaluation. Label availability time is unknown, so temporal ordering alone cannot establish that training labels existed at each evaluation cutoff.

## Final base-data model-input schema

The unencoded interface is `order_value`, `num_items_ordered`, `order_weekday`, `order_month`, `country_code`, `collect_type`, `payment_method`, in that order. The fitted base-data artifact emits 31 columns in the order below. Category names are learned from training; a different training batch may produce a different schema, so all members must share the same fitted artifact for a given model.

```text
amounts__order_value
amounts__num_items_ordered
dates__order_weekday
dates__order_month
categories__country_code_BD
categories__country_code_MY
categories__country_code_PH
categories__country_code_PK
categories__country_code_TH
categories__collect_type_delivery
categories__collect_type_pickup
categories__payment_method_Online Banking
categories__payment_method___missing__
categories__payment_method_adyen hpp molpay
categories__payment_method_antfinancial bkash
categories__payment_method_antfinancial gcash
categories__payment_method_antfinancial truemoney
categories__payment_method_balance
categories__payment_method_boost wallet
categories__payment_method_credit card
categories__payment_method_cybersource
categories__payment_method_cybersource applepay
categories__payment_method_generic creditcard
categories__payment_method_invoice
categories__payment_method_jazzcash wallet
categories__payment_method_no payment
categories__payment_method_on Delivery
categories__payment_method_online banking
categories__payment_method_payment on Delivery
categories__payment_method_payment on delivery
categories__payment_method_paypal
```

Payment-method casing is kept as provided by PBI 02, so similar labels remain separate until their equivalence is confirmed. Some categories might be operationally unavailable at review time; confirm provenance before production use. The fitted artifact contains learned categories and statistics and should be handled as potentially sensitive. Only load trusted joblib artifacts, and retain the artifact and dependency versions with each future model.
