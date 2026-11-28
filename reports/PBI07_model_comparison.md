# PBI 07 — Nithish's fraud-model experiments

**Owner:** Nithish (`@nith-ish22`)

**Purpose:** Give Trust & Safety evidence about which orders to send for human review. This work is an experiment, not permission to automatically reject orders or customers.

## Evidence status

The final results are the three runs logged to the shared MLflow service on 22 November 2026 (see "Shared MLflow runs" below). Each run has its own logged model, and the experiment, run links and metrics are retrievable by the whole team. The earlier local-only runs are kept further down as history and are not used for any comparison.

All three runs use the shared CA1 protocol: 525,226 fit rows (12 Nov 2022 to 17 Feb 2023), 135,547 threshold-calibration rows (18 Feb to 14 Mar 2023) and 144,978 validation orders (15 Mar to 13 Apr 2023), seed 42. These are the same fit, calibration and validation counts and dates logged by Han Qian's PBI 06 runs, so the results can be compared directly (see `PBI09_model_comparison.md`).

## Evaluation protocol

1. PBI 02 cleaning is applied to customers, orders and labels.
2. PBI 03 assigns whole dates to an outer chronological split: oldest 70% of matched, dated orders for training, next 15% for validation, newest 15% held out for final testing. The split helper's default behavior is unchanged for existing callers.
3. Inside the training period, the newest 20% of whole dates is reserved for threshold calibration. The model is fitted on the older dates; that **same fitted model** produces calibration and validation scores. This removes the previous refit/threshold mismatch and keeps validation out of threshold selection.
4. v1 and v2 share one fitted logistic model and differ only in operating threshold: fixed 0.50 versus the maximum-F1 threshold selected on the calibration slice. v3 uses the same feature set and split policy with a histogram gradient-boosting model and its own calibration threshold.
5. Validation is the only period used for reported model comparison. The newest 15% test period is not materialised as feature/label arrays by this training path and must remain untouched until PBI 09 chooses a champion.

Because threshold calibration uses part of the outer training period, PBI 07's models fit on fewer rows than a model trained on the full outer training period. The shared validation dates and metric definitions remain the same, but comparisons with PBI 06 are **directional unless PBI 06 is recomputed with the same inner calibration protocol**. Each MLflow run records exact fit/calibration counts and dates; the notebook displays the outer split and explains this limitation.

Metrics reported: precision, recall, F1, average precision, ROC-AUC and the confusion-matrix counts. F1 balances precision and recall; average precision and ROC-AUC describe ranking quality independently of a selected operating threshold. Accuracy is not used as a headline metric because fraud is the minority class.

## Historical local validation results (15 November 2026, superseded)

These figures came from an earlier local notebook run, before the shared feature set and the shared server. They are kept as history only. They are validation results, not test-set results or production guarantees.

| Version | Cut-off | F1 | Precision | Recall | AP | ROC-AUC | False alarms | Missed fraud | Orders reviewed | Local MLflow run ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v1 logistic, fixed threshold | 0.5000 | 0.4867 | 0.3535 | 0.7806 | 0.5079 | 0.8416 | 28,177 | 4,331 | 30.1% | `72e67b546e8546faa76a3f60479ac739` |
| v2 same logistic model, tuned threshold | 0.6425 | 0.5194 | 0.4797 | 0.5662 | 0.5079 | 0.8416 | 12,122 | 8,564 | 16.1% | `007bb93e5b34487a9e35aef4ba8130de` |
| v3 histogram gradient boosting | 0.6940 | 0.5390 | 0.5032 | 0.5803 | 0.5259 | 0.8499 | 11,311 | 8,285 | 15.7% | `7598ab24dbe14dfab74eeece9222efdf` |

v1 and v2 use the same logistic model and have identical AP and ROC-AUC. Moving the cut-off reduces review volume from 30.1% to 16.1%, but recall falls from 78.1% to 56.6%. v3 improves F1/AP over v2 on this validation period and sends 15.7% of orders to review. That does not establish a production winner: threshold choice still needs review-capacity and cost information, and the model's country-level error rates need business and fairness review.

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
- [x] Rerun the corrected notebook on the fixed CA1 base data; all 16 code cells completed without notebook errors and the corrected validation table and charts were reviewed.
- [x] Log each final run to the shared MLflow service and retain run links (see "Shared MLflow runs").
- [x] Log a retrievable model for each final run (three logged models, status `LOGGED_MODEL_READY`).
- [x] Compare with PBI 06 on the identical fit/calibration protocol (same fit rows, calibration rows, validation window and seed); the comparison is in `PBI09_model_comparison.md`.
- [ ] Show the three models loading back from the shared server (run `scripts/verify_pbi07_models.py` and keep the output).
- [ ] Have a teammate review the result and retain the review or pull-request evidence (PR #83).

Close PBI 07 after the two unchecked items are done.

## Shared eight-feature rerun (local runs, 17 November 2026, superseded)

The historical results above remain unchanged. The three run IDs in the historical table were not present in the local MLflow database available at the start of this change; their reported metrics remain in this report, but their artifacts could not be checked or reloaded from this checkout. The updated notebook loads only development-eligible rows with `load_development_tables()` and uses the shared PBI 03 feature/preprocessing implementation. It logs a separate `shared-eight-feature` batch. The frozen outer split is the same as PBI 06: training 660,773 rows (12 Nov 2022–14 Mar 2023), validation 144,978 rows (15 Mar–13 Apr 2023). The latest 20% of whole training dates is used for threshold selection; the held-out test data are not returned or evaluated.

| New version | Threshold | F1 | Precision | Recall | AP | ROC-AUC | FP | FN | Fit seconds | Local MLflow run ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v1 shared logistic, fixed | 0.5000 | 0.4782 | 0.3413 | 0.7984 | 0.5082 | 0.8418 | 30,412 | 3,979 | 2.10 | `08048524c3cf4b85a2ad45d032d4dbcc` |
| v2 shared logistic, calibrated | 0.6476 | 0.5221 | 0.4714 | 0.5850 | 0.5082 | 0.8418 | 12,950 | 8,192 | 0.10 | `39bda182bac844c992e8a4da522f7d07` |
| v3 shared HistGradientBoosting | 0.7123 | 0.5223 | 0.5288 | 0.5159 | 0.5193 | 0.8475 | 9,075 | 9,556 | 3.50 | `fef5227f0a084e78b22db05cc2c27c97` |

Against the historical PBI 07 validation table, the fixed-threshold logistic F1 changes from 0.4867 to 0.4782 (-0.0085); calibrated logistic changes from 0.5194 to 0.5221 (+0.0027); and HGB changes from 0.5390 to 0.5223 (-0.0167). These are observed differences, not isolated causal effects: the common preprocessing, case normalization and HGB estimator settings changed together. The threshold increases logistic F1 and sharply reduces false positives but catches fewer fraud orders. The new HGB has a slightly higher F1 than calibrated logistic (0.5223 vs 0.5221), but lower recall (0.5159 vs 0.5850). Its AP is 0.5193 compared with logistic 0.5082. The choice depends on costs and review capacity.

The prior HGB native categorical and monotonic constraints were removed to use the common one-hot representation. This changes model behavior and is a new model version, not a like-for-like repeat. The first-order timestamp's point-in-time provenance and an independent order-to-customer mapping remain production blockers. PBI 08 has not been integrated because Ephraim's notebook is not present in this local checkout.

During verification, the notebook was first run with the split helper's row-quantile default, which began validation on 25 February instead of the frozen 15 March boundary. Those three preliminary `shared-eight-feature` runs remain in the local experiment store with their split dates and metrics intact; they are superseded and must not be used for team comparison. After detecting this, `prepare_data()` was corrected to pass the shared `VALIDATION_START`, and the three run IDs listed in the table above were produced on the aligned dates. No existing runs were deleted or edited.

## Final results on the shared server (22 November 2026)

| Version | Cut-off | F1 | Precision | Recall | AP | ROC-AUC | False alarms | Missed fraud | Orders reviewed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 logistic, fixed cut-off | 0.5000 | 0.4782 | 0.3413 | 0.7983 | 0.5082 | 0.8418 | 30,411 | 3,981 | 31.8% |
| v2 same logistic model, calibrated cut-off | 0.6477 | 0.5222 | 0.4720 | 0.5845 | 0.5082 | 0.8418 | 12,908 | 8,202 | 16.9% |
| v3 histogram gradient boosting, calibrated cut-off | 0.7113 | 0.5221 | 0.5256 | 0.5187 | 0.5197 | 0.8477 | 9,244 | 9,500 | 13.4% |

Each run has a logged model in the shared MLflow service (status `LOGGED_MODEL_READY`):

| Version | Logged model ID |
| --- | --- |
| v1 | `m-141667ee048f42b7973e5a5d9d5dc005` |
| v2 | `m-fbbd22521e3a4c978c77671c0ec5b2cd` |
| v3 | `m-216bea3fda1e46b99b662e2ab7205d42` |

Each run also stores `confusion_matrix.png`, `precision_recall.png` and `inference.json`. Run parameters record the fit rows, calibration rows and dates, the cut-off rule, the feature list, preprocessing and seed. The run tags record the member (`nith-ish22`), the model name and the change from the previous version.

**What changed between versions, and why.**

1. v1 is an explainable logistic regression at the default 0.5 cut-off. Diagnosis: it catches 79.8% of fraud, but 30,411 genuine orders are flagged (precision 0.34), and nearly a third of all orders would go to Trust & Safety reviewers, who have limited time.
2. v2 keeps the same fitted model and moves only the cut-off, chosen as the maximum F1 on the newest 20% of training dates. Result: false alarms fall from 30,411 to 12,908 and review load from 31.8% to 16.9%, at the cost of recall (79.8% to 58.4%). AP and ROC-AUC are identical to v1, which confirms that only the operating point changed, so v1 and v2 are not two independent models.
3. v3 changes the model family to histogram gradient boosting, with the same features and cut-off rule as v2. Diagnosis: logistic regression cannot capture non-linear patterns. Result: AP rises from 0.5082 to 0.5197 and false alarms fall to 9,244, but recall drops to 51.9%, so F1 is unchanged (0.5221 against 0.5222). The choice between v2 and v3 depends on whether reviewers value fewer false alarms (v3) or more caught fraud (v2).

In the team comparison (`PBI09_model_comparison.md`) neither v2 nor v3 reaches the champion or runner-up positions. Han Qian's calibrated Random Forest (F1 0.5435) and calibrated CatBoost (F1 0.5208, AP 0.5250) are registered instead.

## Shared MLflow runs (22 November 2026)

The notebook was rerun from top to bottom against the shared MLflow server (Elastic Beanstalk environment `mlflow-pbi12`, RDS MySQL metadata, S3 artifacts; see `PBI12_MLFLOW_STORAGE.md`). All cells completed with no errors. The runs are in experiment 3 on the shared server.

| Version | Shared MLflow run |
| --- | --- |
| v1 logistic, fixed cut-off | [`c10fc24f606349d489314e67f665b3d1`](http://mlflow-pbi12.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/c10fc24f606349d489314e67f665b3d1) |
| v2 logistic, calibrated cut-off | [`5c40eaf968a6471297d39f7ce9704917`](http://mlflow-pbi12.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/5c40eaf968a6471297d39f7ce9704917) |
| v3 histogram gradient boosting, calibrated cut-off | [`b19132dd57a74c638697e6a86d1bf1a4`](http://mlflow-pbi12.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/b19132dd57a74c638697e6a86d1bf1a4) |

The server is stopped when the AWS Learner Lab session ends. The runs and models are stored in RDS and S3, so they are still there after the lab is started again.

The "Local MLflow run ID" values in the earlier tables refer to local runs and are kept as history. The shared links above are the ones to use for team comparison.

### Retrieving the models from the shared server

`scripts/verify_pbi07_models.py` loads each of the three logged models from the shared server and prints its type and input schema. Run it from the repository root:

```powershell
$env:MLFLOW_TRACKING_URI="http://mlflow-pbi12.eba-9bnrxizu.us-east-1.elasticbeanstalk.com"
python scripts/verify_pbi07_models.py
```

Keep the printed output as the artifact-retrieval evidence for the acceptance criteria.
