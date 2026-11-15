# PBI 07 — Nithish's fraud-model experiments

**Owner:** Nithish (`@nith-ish22`)

**Purpose:** Give Trust & Safety evidence about which orders to send for human review. This work is an experiment, not permission to automatically reject orders or customers.

## Evidence status

The supplied notebook contained a pre-correction run. Its metric tables and charts are not final evidence: the tuned cut-offs had been selected using models fitted on 80% of the training period, while validation used separately refitted models trained on 100%. That made the v1/v2 comparison invalid. Those outputs have been cleared from the notebook. **Rerun the notebook after this correction before quoting any model metrics or claiming a winning model.**

The shared MLflow URL was not available during this correction because the AWS service had been stopped. Local MLflow runs are not evidence for the shared-service acceptance criterion. The final shared run links and a teammate's review remain to be added when the service is available.

## Evaluation protocol

1. PBI 02 cleaning is applied to customers, orders and labels.
2. PBI 03 assigns whole dates to an outer chronological split: oldest 70% of matched, dated orders for training, next 15% for validation, newest 15% held out for final testing. The split helper's default behavior is unchanged for existing callers.
3. Inside the training period, the newest 20% of whole dates is reserved for threshold calibration. The model is fitted on the older dates; that **same fitted model** produces calibration and validation scores. This removes the previous refit/threshold mismatch and keeps validation out of threshold selection.
4. v1 and v2 share one fitted logistic model and differ only in operating threshold: fixed 0.50 versus the maximum-F1 threshold selected on the calibration slice. v3 uses the same feature set and split policy with a histogram gradient-boosting model and its own calibration threshold.
5. Validation is the only period used for reported model comparison. The newest 15% test period is not materialised as feature/label arrays by this training path and must remain untouched until PBI 09 chooses a champion.

Because threshold calibration uses part of the outer training period, PBI 07's models fit on fewer rows than a model trained on the full outer training period. The shared validation dates and metric definitions remain the same, but comparisons with PBI 06 are **directional unless PBI 06 is recomputed with the same inner calibration protocol**. The notebook reports the exact fit/calibration counts and dates for each run so this limitation is visible.

Metrics reported: precision, recall, F1, average precision, ROC-AUC and the confusion-matrix counts. F1 balances precision and recall; average precision and ROC-AUC describe ranking quality independently of a selected operating threshold. Accuracy is not used as a headline metric because fraud is the minority class.

## Model versions and question answered

| Version | Model and policy | What the comparison can establish |
| --- | --- | --- |
| v1 | Logistic regression, threshold 0.50 | Baseline operating point for the shared logistic model |
| v2 | Same fitted logistic model, threshold selected on calibration data for maximum F1 | Effect of changing the review threshold only; this is a policy variant, not a second trained model |
| v3 | Histogram gradient boosting, threshold selected by the same rule | Whether a nonlinear model improves validation ranking and review outcomes |

Do not present v1 and v2 as independent model families or register them as separate model candidates in PBI 09. They share identical fitted model weights; the threshold is separately logged with each run.

## Feature and leakage decisions

The model uses the seven PBI 03 order-time features plus `days_since_first_order`, derived as order date minus `first_order_datetime`.

`first_order_datetime` records a customer's first recorded order, not necessarily account registration. The feature is therefore named **days since first recorded order**, not “account age.” It is a historical-behavior proxy. It is set to missing when the first-order date is absent or after the current order date. Before treating it as an account-age feature, the group must verify the field definition in the data dictionary.

| Data | Decision | Reason |
| --- | --- | --- |
| Order value, item count, weekday, month, country, collection type and payment method | Use | Available by the time the order is placed; lowercase payment text to merge casing variants |
| Days since first recorded order | Use cautiously | Available only if the first-order timestamp is valid and precedes the order; it is not verified account-registration age |
| Refund value | Exclude | Occurs after checkout and can reveal a later outcome |
| Undated 50-day customer totals and other undated customer snapshots | Exclude | Their values may include behavior after the scored order |
| IDs | Join only | Identifiers are not model inputs |

Imputation, encoding, scaling and model fitting are inside each model pipeline and are fitted only on the model-fit rows. Unknown categories are handled without requiring a validation/test category to be known in advance. Balanced class weights address the rare fraud class without resampling.

## Business interpretation and safeguards

The threshold chosen for maximum F1 is a reproducible comparison rule, **not** an approved business operating point. F1 does not encode review-team capacity, the cost of missed fraud, customer harm, or investigation time. Trust & Safety must choose an operating threshold using those costs and the number of cases analysts can review.

Country is a model input and can create materially different flag rates and error rates between markets. The notebook therefore reports per-country support, fraud prevalence, flag rate, precision, recall and false-positive rate after rerun. Small groups must be interpreted cautiously. Country must not be used for an automatic decline; use the output only to prioritise human review and have the business assess fairness, local rules and customer impact before operational use.

## MLflow, reproducibility and PBI 10 hand-off

The callable entry point is:

```python
from src.train_nithish import train_nithish_models
results = train_nithish_models(customers, orders, labels, batch_name="base")
```

Inputs are the cleaned DataFrames returned by PBI 02. To use the shared server, set `MLFLOW_TRACKING_URI` (or pass `tracking_uri`) before running. If neither is set, the routine warns and uses the local `artifacts/mlflow/` store; a local run must not be described as a shared run. Each run records member/PBI tags, model/version, features, preprocessing and model parameters, outer and inner split dates/counts, threshold rule, metrics, diagnostic figures, inference metadata and a retrievable model artifact containing preprocessing.

Install the bounded PBI 07 runtime dependencies from `requirements-pbi07.txt`. The model artifact also records the exact active NumPy, pandas, scikit-learn and MLflow versions. Invocation, model output fields and the threshold are demonstrated in `PBI07_Nithish_models.ipynb`.

## Completion checklist

- [x] Correct the threshold/model mismatch so calibration and validation scores come from the same fitted model.
- [x] Preserve the existing PBI 03 default interface while adding opt-in training dates and an option not to materialise test features/labels.
- [x] Remove the PBI 06 module import that is absent from the integration branch; calculate the documented standard metrics locally.
- [x] Name the first-order-date feature accurately and document its semantic limitation.
- [x] Make local MLflow fallback explicit and warn that it does not satisfy shared-service evidence.
- [ ] Rerun the corrected notebook on the authorised base data and review every generated metric/chart.
- [ ] Log each final run to the shared MLflow service and retain run links plus artifact retrieval evidence.
- [ ] Have a teammate review the result and retain the review/PR evidence.
- [ ] Compare with PBI 06 on the identical fit/calibration protocol before making a cross-member champion claim.

Only close PBI 07 after the unchecked evidence items are satisfied.
