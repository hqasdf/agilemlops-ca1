# PBI 07 — Fraud model comparison

**Owner:** Nithish (`@nith-ish22`)
**Purpose:** Give Trust & Safety evidence for prioritising orders for human review. The model does not approve or reject orders.

## Status

The notebook and training code were rerun on **17 November 2026** with the fixed CA1 base files. Hyperparameter search, calibration, validation, charts and local MLflow logging completed. The newest test period was not used.

The runs below are in the local MLflow store because `MLFLOW_TRACKING_URI` was not set during execution. They are reproducible local evidence; they do not demonstrate logging to the shared AWS MLflow service. A teammate review and a same-protocol comparison with PBI 06 are also still needed.

## Data and evaluation

The routine applies the PBI 02 cleaners and the PBI 03 whole-date chronological split:

| Period | Rows | Dates | Fraud rate | Use |
| --- | ---: | --- | ---: | --- |
| Training | 660,773 | 2022-11-12 to 2023-03-14 | 10.1% | Model selection, fitting and threshold calibration |
| Validation | 144,978 | 2023-03-15 to 2023-04-13 | 13.6% | Compare the three versions once |
| Test | 144,123 | 2023-04-14 to 2023-05-11 | Not examined | Hold for PBI 09 |

Within outer training, the latest 20% of whole dates is held for threshold calibration. Models are fitted on the earlier dates. Hyperparameters are selected inside those model-fit dates: candidates train on the earlier 80% of model-fit dates and are scored on the later 20%. The selected settings are then refit on all model-fit dates. The same final fitted model scores the calibration and validation periods.

Hyperparameters are chosen by **average precision (AP)**. AP compares fraud ranking across thresholds and is useful when fraud is the minority class. The decision threshold is selected separately by maximum F1 on calibration data. Neither validation nor test chooses hyperparameters or thresholds.

## Search results

| Model family | Candidates | Selected settings | Best tuning AP | Tuning period |
| --- | ---: | --- | ---: | --- |
| Logistic regression | 3 | `C=0.1` | 0.4043 | 2023-01-28 to 2023-02-17 |
| Histogram gradient boosting | 8 | `learning_rate=0.10`, `max_leaf_nodes=15`, `min_samples_leaf=100`, `l2_regularization=1.0` | 0.4548 | 2023-01-28 to 2023-02-17 |

Logistic candidates `C=0.1`, `1.0` and `10.0` scored AP 0.40433, 0.40429 and 0.40431. The differences are tiny, so the search did not find a meaningful regularisation advantage.

The eight boosting candidates covered learning rates 0.05/0.10, 7/15 leaves and minimum leaf sizes 100/300. The search paired L2 regularisation 0.0 with learning rate 0.05 and 1.0 with learning rate 0.10. The best candidate scored 0.45482; the next best scored 0.45471. This is a narrow grid and a single chronological tuning slice, so the selected settings are the best of these candidates on that period, not a guarantee of future performance.

## Validation results

| Version | Settings / threshold | F1 | Precision | Recall | AP | ROC-AUC | False alarms | Missed fraud | Orders reviewed | Local MLflow run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v1 logistic | `C=0.1`, threshold 0.5000 | 0.4863 | 0.3528 | 0.7820 | 0.5080 | 0.8417 | 28,317 | 4,303 | 30.2% | `b4779bbc719f4d55b800ae66385db6b3` |
| v2 same logistic | `C=0.1`, threshold 0.6421 | 0.5194 | 0.4788 | 0.5676 | 0.5080 | 0.8417 | 12,196 | 8,536 | 16.1% | `9ca9906392244bfd92509c8e7bc093e1` |
| v3 histogram gradient boosting | tuned settings above, threshold 0.7331 | 0.5236 | 0.5310 | 0.5164 | 0.5265 | 0.8488 | 9,003 | 9,547 | 13.2% | `b8e97b8368f84085a55abef3ac77d6e8` |

v1 and v2 share the same fitted logistic model and have identical ranking metrics. Their difference is only the decision threshold. v3 has the best validation AP and slightly higher F1 than v2. At its selected threshold it flags fewer orders and has higher precision, while missing 1,011 more fraud cases than v2. This is a business trade-off, not an automatic model-selection decision.

## Features and safeguards

The models use the seven PBI 03 order-time features plus `days_since_first_order`, calculated from the order date and `first_order_datetime`. This is a first-recorded-purchase history proxy, not verified account age. It is missing if the first order date is absent or after the scored order.

| Field | Decision | Reason |
| --- | --- | --- |
| Order value, item count, weekday, month, country, collection type, payment method | Use | Available at order time; payment text is lowercased to merge casing variants |
| Days since first recorded order | Use cautiously | Historical proxy; field meaning and monotonic direction need business confirmation |
| Refund value | Exclude | Recorded after checkout and can reveal a later outcome |
| Undated 50-day customer totals/snapshots | Exclude | Could contain activity after the order being scored |
| IDs | Join only | Identifiers are not model inputs |

Imputation, encoding and scaling are inside each fitted pipeline. Unknown categories are handled without requiring advance knowledge of validation categories. Balanced class weights address the minority fraud class. Country-level results are reported with support; investigate market differences and fairness before any operational use. Use scores only to prioritise human review.

## Reproducibility and hand-off

The callable entry point is `train_nithish_models(customers, orders, labels)`. It accepts cleaned PBI 02 tables, repeats the search and split-safe calibration, evaluates on validation and logs the chosen settings, every candidate's tuning AP, dates, metrics, charts and a retrievable preprocessing-plus-model artifact to MLflow.

To produce shared-service evidence, set `MLFLOW_TRACKING_URI` to the active team MLflow service and rerun the notebook. Do not describe the current local runs as shared runs.

## Remaining PBI 07 evidence

- [x] Tune logistic and boosted-tree settings using training dates only.
- [x] Keep the original outer 70/15/15 split and leave test data untouched.
- [x] Refit the selected models, calibrate thresholds separately and rerun the notebook.
- [x] Record candidate settings, tuning AP, validation metrics and local MLflow run IDs.
- [ ] Rerun with the shared MLflow tracking URI and retain the shared run/artifact links.
- [ ] Obtain a teammate review and record it.
- [ ] Recompute PBI 06 with the same fit/calibration protocol before making a strict cross-model comparison.
- [ ] Use the held-out test period only in PBI 09 after selecting a candidate and operating policy.
