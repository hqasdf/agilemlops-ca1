# PBI 06 — Six-model baseline comparison

The six baseline models in `CA1.ipynb` use the same cleaned records, chronological 70/15/15 split and preprocessor fitted on training only. Training has **660,773 rows** (12 November 2022–14 March 2023); validation has **144,978 rows** (15 March–13 April 2023). The held-out test partition is not passed to the models or preprocessor and no test metrics are calculated. All baseline classification thresholds are **0.5**; the separate training-only tuning phase below also keeps that threshold fixed.

**Historical limitation:** PBI 01 examined fraud outcomes across the full historical dataset before the holdout was established. The current benchmark does not inspect test-specific outcomes, but the holdout is not completely untouched by prior analysis. Customer snapshots are excluded because their historical availability at prediction time is unverified. Confirm payment and collection feature timing before operational use.

## Validation leaderboard

Sorted by fraud-class F1; AP and ROC-AUC use probability scores. FPR is FP / (FP + TN), the fraction of legitimate orders flagged. Fit and prediction times are individual local measurements; they exclude cleaning, preprocessing and MLflow logging.

| Model | F1 | Precision | Recall | AP | ROC-AUC | Accuracy | FPR | TN | FP | FN | TP | Fit s | Predict s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CatBoost | 0.4443 | 0.3096 | 0.7865 | 0.4688 | 0.8224 | 0.7321 | 0.2765 | 90,612 | 34,626 | 4,215 | 15,525 | 2.080 | 0.0061 |
| Random forest | 0.4411 | 0.3055 | 0.7931 | 0.4755 | 0.8237 | 0.7263 | 0.2842 | 89,646 | 35,592 | 4,085 | 15,655 | 3.009 | 0.0388 |
| XGBoost | 0.4410 | 0.3055 | 0.7924 | 0.4756 | 0.8238 | 0.7264 | 0.2839 | 89,677 | 35,561 | 4,098 | 15,642 | 0.753 | 0.0412 |
| LightGBM | 0.4409 | 0.3054 | 0.7925 | 0.4784 | 0.8243 | 0.7263 | 0.2841 | 89,659 | 35,579 | 4,097 | 15,643 | 0.678 | 0.0723 |
| HistGradientBoosting | 0.4398 | 0.3045 | 0.7914 | 0.4737 | 0.8226 | 0.7254 | 0.2850 | 89,548 | 35,690 | 4,117 | 15,623 | 2.578 | 0.0751 |
| Logistic regression | 0.4246 | 0.2906 | 0.7876 | 0.4340 | 0.8108 | 0.7094 | 0.3030 | 87,294 | 37,944 | 4,193 | 15,547 | 2.644 | 0.0012 |

CatBoost leads on F1 and has the lowest FPR, but misses 130 more fraud orders than random forest. LightGBM has the highest AP and ROC-AUC; random forest has the highest recall. F1 differences among the tree models are small, and all send more than 34,000 legitimate orders for review. They are not evidence of stable production superiority or causality.

Before the training-only search below, the baseline candidates recommended for further study were **CatBoost** (best F1 and fewer false positives), **LightGBM** (best AP/ROC-AUC and fastest fit in this run), and **random forest** (second F1 and strongest recall). XGBoost is a close alternative to the random forest with a faster fit in this run. Consider review capacity, missed-fraud cost, repeatability and different time periods before choosing; simply trying more algorithms is not strategic optimization.

## Baseline configurations and class imbalance

All use seed 42; the existing three configurations were not changed. Logistic regression uses balanced class weights and `max_iter=200`. Random forest uses balanced class weights, 20 trees, depth 12, minimum leaf 30 and four jobs. HistGradientBoosting uses 80 maximum iterations, 15 leaves and learning rate 0.1. XGBoost uses 80 trees, depth 5, learning rate 0.1, histogram tree method and four jobs. LightGBM uses 80 trees, 31 leaves, learning rate 0.1 and four jobs. CatBoost uses 80 iterations, depth 5, learning rate 0.1 and four threads. Histogram boosting, XGBoost, LightGBM and CatBoost each receive balanced training sample weights once; no second model-level class weighting is enabled. These are starting configurations, not tuned optima. HistGradientBoosting retains its library-default automatic early stopping.

The notebook displays the full parameter dictionaries, confusion matrices, six precision–recall curves, a metrics comparison and a recall-versus-FPR chart. Every model is scored on the same validation rows, and the same fitted preprocessor is saved with each model. HistGradientBoosting and CatBoost receive a dense representation of those identical transformed features.

## Timing and MLflow evidence

The measured six-model benchmark, including transforming training/validation data and logging runs, took **13.46 seconds**. The complete notebook took **34.04 seconds** wall-clock on this machine, including EDA and cleaning. These are single local runs; the three-minute PBI 10 requirement also includes incoming URLs, RDS operations, all team models and shared AWS MLflow, none of which was measured here. The end-to-end requirement remains unverified.

All runs went to the local `artifacts/mlflow/` backend; no shared AWS service was contacted. Each run has parameters, metrics, confusion matrix, precision–recall curve, saved model, fitted preprocessor and inference metadata. All six saved model/preprocessor pairs were reloaded and reproduced their complete validation metrics. Earlier experiments, including the historical tuned run, remain in the local store. Start the local UI as described in README; latest run IDs are:

- CatBoost: `b8e6f0e9663d4e8bbc5fe04a92c2edb2`
- Random forest: `b6d9e8d4791f47cebdb2a2332a680453`
- XGBoost: `e35157efc8754e929678097d19338a40`
- LightGBM: `c65a2b1e8df74457980a15c8c390e5b0`
- HistGradientBoosting: `1eb492a42fdb430aaedd6e5d8c820e71`
- Logistic regression: `bc95e5c0fbfb4a79ad3b3290882d98cc`

## Remaining integration

The shared AWS MLflow service, RDS updates, incoming CSV URL processing, integration with the PBI 10 owner and end-to-end demo runtime still require team work. PBI 06's customer-information criterion also remains unresolved until point-in-time customer data can be verified. Do not register a champion or evaluate test data before team selection based on validation.

## Training-only tuning phase

A small fixed search tried three configurations for each of CatBoost, LightGBM and random forest on two expanding chronological folds **inside the outer training partition**. Fold 1 trained on 12 November 2022–22 January 2023 and scored 23 January–17 February 2023. Fold 2 trained on 12 November 2022–17 February 2023 and scored 18 February–14 March 2023. Whole dates stayed together. Each fold's preprocessor was fitted only on that fold's training rows, and imbalance weights were calculated from those rows only. Mean inner-fold fraud F1 chose each model's configuration; outer validation labels and scores did not participate. The threshold stayed at 0.5.

All three selected their third candidate, which combines model-parameter changes with a **0.5 weighting strength**: `1 + 0.5 × (balanced_weight - 1)`. This is less aggressive than fully balanced weights. Random forest's selected model sets `class_weight=None` and receives these sample weights, so it is not double-weighted. After selection, each preprocessor and model was fitted on the full training partition, then scored on outer validation.

| Model | Selected changes from baseline | Inner mean F1 | Validation F1 baseline → tuned | Precision baseline → tuned | Recall baseline → tuned | FP baseline → tuned | FN baseline → tuned |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 120 estimators, 15 leaves, learning rate 0.08, L2 5; weight strength 0.5 | 0.3841 | 0.4409 → 0.4963 | 0.3054 → 0.4623 | 0.7925 → 0.5358 | 35,579 → 12,303 | 4,097 → 9,163 |
| CatBoost | 120 iterations, depth 5, learning rate 0.08, L2 leaf regularization 5; weight strength 0.5 | 0.3857 | 0.4443 → 0.4941 | 0.3096 → 0.4644 | 0.7865 → 0.5278 | 34,626 → 12,015 | 4,215 → 9,322 |
| Random forest | 40 trees, depth 12, minimum leaf 40, `class_weight=None`; weight strength 0.5 | 0.3836 | 0.4411 → 0.4885 | 0.3055 → 0.4694 | 0.7931 → 0.5094 | 35,592 → 11,368 | 4,085 → 9,685 |

| Tuned model | AP | ROC-AUC | Accuracy | FPR | TN | FP | FN | TP | Full-training fit s | Validation predict s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.4828 | 0.8226 | 0.8519 | 0.0982 | 112,935 | 12,303 | 9,163 | 10,577 | 0.743 | 0.0853 |
| CatBoost | 0.4737 | 0.8226 | 0.8528 | 0.0959 | 113,223 | 12,015 | 9,322 | 10,418 | 2.826 | 0.0079 |
| Random forest | 0.4797 | 0.8246 | 0.8548 | 0.0908 | 113,870 | 11,368 | 9,685 | 10,055 | 6.377 | 0.0631 |

The reduced imbalance weighting is consistent with the baseline's high false-positive rates, but these candidates also changed tree settings, so the comparison does **not isolate weighting as the cause**. Precision rose and false alarms fell substantially, at the cost of missing roughly 5,000–5,600 additional fraud orders per model. Operational review capacity and the cost of missed fraud are needed before treating the trade-off as an improvement in practice.

**LightGBM is the strongest tuned candidate by outer-validation F1 (0.4963)**, narrowly ahead of CatBoost; it is a candidate for team review, not an automatically registered champion. The inner search including MLflow logging took **35.10 s**; search plus selected-model preprocessing, refits, validation and logging took **46.20 s**. The full notebook took **76.12 s** locally. These timings are separate from ordinary baseline training and do not prove the PBI 10 end-to-end three-minute requirement.

Eighteen inner-fold trial runs and three selected-model runs were logged locally, each with parameters, metrics, preprocessor and model artifacts. The selected model/preprocessor pairs were reloaded and reproduced their validation metrics. Latest selected runs: LightGBM `e229abb1b96c41cabcf1eaf6bd085762`, CatBoost `d91606601ea440d590bf68072ac16565`, random forest `034f3237a9ed4adc9bc7b30859de4a60`. All prior runs remain available. The exploratory search is not called by the normal `train_benchmark` retraining path; PBI 10 integration still requires the team to decide how to use selected configurations.

Repeated comparisons on the same outer validation period can make model selection optimistic. In addition, the historical EDA already examined full-data labels before the holdout existed. Once the team freezes a champion and evaluation protocol, any final test assessment requires separate authorization and should be interpreted in light of that history. No test-set metrics were produced in this tuning phase.

## Advanced training-only search

The advanced search used **20 configurations per model**, each evaluated on both existing chronological inner folds: 120 fold fits. The previous tuned configuration was included as candidate 1; the other 19 per model were sampled reproducibly from bounded discrete parameter grids with seed 42. The grids vary estimator count, depth or leaves, learning rate where applicable, regularization, row/feature sampling and training-only class-weight strength. The threshold remained **0.5** throughout; no threshold tuning was performed.

All algorithms used the same inner date boundaries and fold-specific preprocessing. Weights were derived from each inner training fold, without double-weighting. Mean inner-fold fraud F1 selected each configuration before any outer-validation scoring. The selected configurations were refitted on the complete outer training partition. The test set was not used for selection or evaluation. The initial and advanced searches reuse the same inner folds; a broader search and repeated outer-validation review can lead to optimistic selection, even though validation was not used to choose hyperparameters.

### Combined outer-validation leaderboard

All metrics below are from one notebook run, ranked by fraud F1. FPR is FP / (FP + TN). Fit and prediction times exclude preprocessing and MLflow logging.

| Model | Version | F1 | Precision | Recall | AP | ROC-AUC | Accuracy | FPR | TN | FP | FN | TP | Fit s | Predict s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | advanced | 0.5103 | 0.4052 | 0.6890 | 0.4683 | 0.8217 | 0.8200 | 0.1594 | 105,276 | 19,962 | 6,139 | 13,601 | 0.722 | 0.0812 |
| CatBoost | advanced | 0.5007 | 0.4590 | 0.5507 | 0.4813 | 0.8250 | 0.8504 | 0.1023 | 112,425 | 12,813 | 8,869 | 10,871 | 3.174 | 0.0065 |
| LightGBM | initial tuned | 0.4963 | 0.4623 | 0.5358 | 0.4828 | 0.8226 | 0.8519 | 0.0982 | 112,935 | 12,303 | 9,163 | 10,577 | 0.633 | 0.0677 |
| Random forest | advanced | 0.4951 | 0.4546 | 0.5435 | 0.4777 | 0.8224 | 0.8491 | 0.1028 | 112,369 | 12,869 | 9,012 | 10,728 | 9.999 | 0.0510 |
| CatBoost | initial tuned | 0.4941 | 0.4644 | 0.5278 | 0.4737 | 0.8226 | 0.8528 | 0.0959 | 113,223 | 12,015 | 9,322 | 10,418 | 2.544 | 0.0060 |
| Random forest | initial tuned | 0.4885 | 0.4694 | 0.5094 | 0.4797 | 0.8246 | 0.8548 | 0.0908 | 113,870 | 11,368 | 9,685 | 10,055 | 5.186 | 0.0621 |
| CatBoost | baseline | 0.4443 | 0.3096 | 0.7865 | 0.4688 | 0.8224 | 0.7321 | 0.2765 | 90,612 | 34,626 | 4,215 | 15,525 | 1.795 | 0.0055 |
| Random forest | baseline | 0.4411 | 0.3055 | 0.7931 | 0.4755 | 0.8237 | 0.7263 | 0.2842 | 89,646 | 35,592 | 4,085 | 15,655 | 2.374 | 0.0364 |
| XGBoost | baseline | 0.4410 | 0.3055 | 0.7924 | 0.4756 | 0.8238 | 0.7264 | 0.2839 | 89,677 | 35,561 | 4,098 | 15,642 | 0.618 | 0.0378 |
| LightGBM | baseline | 0.4409 | 0.3054 | 0.7925 | 0.4784 | 0.8243 | 0.7263 | 0.2841 | 89,659 | 35,579 | 4,097 | 15,643 | 0.579 | 0.0649 |
| HistGradientBoosting | baseline | 0.4398 | 0.3045 | 0.7914 | 0.4737 | 0.8226 | 0.7254 | 0.2850 | 89,548 | 35,690 | 4,117 | 15,623 | 2.289 | 0.0653 |
| Logistic regression | baseline | 0.4246 | 0.2906 | 0.7876 | 0.4340 | 0.8108 | 0.7094 | 0.3030 | 87,294 | 37,944 | 4,193 | 15,547 | 2.444 | 0.0011 |

### Selected configurations and trade-offs

- **LightGBM:** 120 estimators, 15 leaves, learning rate 0.04, L2 10, minimum child samples 50, row subsample 1.0, feature subsample 0.7, subsample frequency 1 and weight strength 0.75. Mean inner F1 **0.3871**; outer F1 improved from **0.4963 to 0.5103** compared with initial tuning. Recall increased from 0.5358 to 0.6890, while false positives rose from 12,303 to 19,962.
- **CatBoost:** 160 iterations, depth 6, learning rate 0.12, L2 3, Bernoulli row subsample 0.7 and weight strength 0.5. Mean inner F1 **0.3880**; outer F1 improved **0.4941 → 0.5007**, with false positives **12,015 → 12,813**.
- **Random forest:** 30 trees, depth 10, minimum leaf 50, row sample 0.85, feature sample 1.0, `class_weight=None` and weight strength 0.5. Mean inner F1 **0.3886**; outer F1 improved **0.4885 → 0.4951**, with false positives **11,368 → 12,869**.

**Advanced LightGBM** leads on outer-validation F1, but its greater recall comes with more false alarms than the initially tuned version. Its AP also fell from 0.4828 to 0.4683. Treat it as a candidate for team review, not an automatically registered champion. The 0.5 threshold was not optimized; threshold optimization, if authorized later, is a separate phase.

The advanced inner search including local MLflow logging took **558.90 s**; search plus three refits, outer validation and logging took **573.79 s**. The complete notebook took **646.08 s** on this machine. This extensive exploration is not part of the normal `train_benchmark` retraining path and exceeds three minutes on its own; it must not be included in PBI 10's three-minute demo. The actual PBI 10 end-to-end duration remains unverified.

All 120 fold trials and three advanced refits were logged locally with configurations, metrics, model and preprocessor artifacts. The three advanced model/preprocessor pairs were reloaded and reproduced their outer-validation metrics. Latest advanced runs: LightGBM `d6e45399a8934446aadeb01caddd7849`, CatBoost `12c49c06191b47e2898a126c9f722e11`, random forest `a8998c12d8004c8e9f47db017db1f2f5`. Existing baseline and earlier tuning experiments remain intact. The complete per-fold results and combined leaderboard are displayed in `CA1.ipynb`.

## Training-only threshold optimization

The fixed advanced configurations were refitted separately on each of the two chronological inner-training folds. Each fold used its own training-fitted preprocessor and class weights. Predictions from the two subsequent training-period validation windows were pooled (**269,125 rows**); the F1-maximizing threshold was selected with higher threshold breaking exact ties. No model was refitted per threshold. Outer validation and test outcomes did not select thresholds.

The hyperparameters were previously selected using these same folds, so pooled out-of-fold F1 is optimistic as an estimate for the combined selection process. An independent, newly collected calibration period would provide stronger evidence. The earlier full-data EDA also limits the interpretation of the historical holdout.

| Model | Threshold | Training OOF F1 | Validation F1 at 0.5 → optimized | Precision at 0.5 → optimized | Recall at 0.5 → optimized | FP at 0.5 → optimized | FN at 0.5 → optimized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CatBoost | 0.456841 | 0.3914 | 0.5007 → **0.5115** | 0.4590 → 0.4380 | 0.5507 → 0.6147 | 12,813 → 15,570 | 8,869 → 7,605 |
| LightGBM | 0.558408 | 0.3905 | 0.5103 → **0.5036** | 0.4052 → 0.4449 | 0.6890 → 0.5802 | 19,962 → 14,287 | 6,139 → 8,287 |
| Random forest | 0.461245 | 0.3899 | 0.4951 → **0.5032** | 0.4546 → 0.4518 | 0.5435 → 0.5678 | 12,869 → 13,598 | 9,012 → 8,531 |

CatBoost has the highest observed validation F1 at **0.5115**, only 0.0012 above advanced LightGBM at threshold 0.5. LightGBM’s training-selected threshold reduced its outer-validation F1; it is not an improvement on this validation period. AP and ROC-AUC stay unchanged within each model because thresholds do not change probability rankings. Repeatedly reviewing the same outer validation period creates further selection optimism; confirm operational costs and freeze the candidate before any final assessment.

### Complete validation leaderboard

| Model | Version | Threshold | F1 | Precision | Recall | AP | ROC-AUC | Accuracy | FPR | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| catboost | threshold_optimized | 0.456841 | 0.5115 | 0.4380 | 0.6147 | 0.4813 | 0.8250 | 0.8401 | 0.1243 | 109,668 | 15,570 | 7,605 | 12,135 |
| lightgbm | advanced_tuned | 0.500000 | 0.5103 | 0.4052 | 0.6890 | 0.4683 | 0.8217 | 0.8200 | 0.1594 | 105,276 | 19,962 | 6,139 | 13,601 |
| lightgbm | threshold_optimized | 0.558408 | 0.5036 | 0.4449 | 0.5802 | 0.4683 | 0.8217 | 0.8443 | 0.1141 | 110,951 | 14,287 | 8,287 | 11,453 |
| random_forest | threshold_optimized | 0.461245 | 0.5032 | 0.4518 | 0.5678 | 0.4777 | 0.8224 | 0.8474 | 0.1086 | 111,640 | 13,598 | 8,531 | 11,209 |
| catboost | advanced_tuned | 0.500000 | 0.5007 | 0.4590 | 0.5507 | 0.4813 | 0.8250 | 0.8504 | 0.1023 | 112,425 | 12,813 | 8,869 | 10,871 |
| lightgbm | tuned | 0.500000 | 0.4963 | 0.4623 | 0.5358 | 0.4828 | 0.8226 | 0.8519 | 0.0982 | 112,935 | 12,303 | 9,163 | 10,577 |
| random_forest | advanced_tuned | 0.500000 | 0.4951 | 0.4546 | 0.5435 | 0.4777 | 0.8224 | 0.8491 | 0.1028 | 112,369 | 12,869 | 9,012 | 10,728 |
| catboost | tuned | 0.500000 | 0.4941 | 0.4644 | 0.5278 | 0.4737 | 0.8226 | 0.8528 | 0.0959 | 113,223 | 12,015 | 9,322 | 10,418 |
| random_forest | tuned | 0.500000 | 0.4885 | 0.4694 | 0.5094 | 0.4797 | 0.8246 | 0.8548 | 0.0908 | 113,870 | 11,368 | 9,685 | 10,055 |
| catboost | baseline | 0.500000 | 0.4443 | 0.3096 | 0.7865 | 0.4688 | 0.8224 | 0.7321 | 0.2765 | 90,612 | 34,626 | 4,215 | 15,525 |
| random_forest | baseline | 0.500000 | 0.4411 | 0.3055 | 0.7931 | 0.4755 | 0.8237 | 0.7263 | 0.2842 | 89,646 | 35,592 | 4,085 | 15,655 |
| xgboost | baseline | 0.500000 | 0.4410 | 0.3055 | 0.7924 | 0.4756 | 0.8238 | 0.7264 | 0.2839 | 89,677 | 35,561 | 4,098 | 15,642 |
| lightgbm | baseline | 0.500000 | 0.4409 | 0.3054 | 0.7925 | 0.4784 | 0.8243 | 0.7263 | 0.2841 | 89,659 | 35,579 | 4,097 | 15,643 |
| hist_gradient_boosting | baseline | 0.500000 | 0.4398 | 0.3045 | 0.7914 | 0.4737 | 0.8226 | 0.7254 | 0.2850 | 89,548 | 35,690 | 4,117 | 15,623 |
| logistic_regression | baseline | 0.500000 | 0.4246 | 0.2906 | 0.7876 | 0.4340 | 0.8108 | 0.7094 | 0.3030 | 87,294 | 37,944 | 4,193 | 15,547 |

### Reuse, runtime and artifacts

Normal Run All reuses the saved advanced configurations and their 120 trial records from local MLflow. To rerun the expensive advanced search, set `RUN_ADVANCED_SEARCH = True` in its notebook cell and execute it; the newly selected run IDs are saved locally for later runs. Source-file fingerprints and split metadata are checked before reuse; changed input files require a fresh advanced search. The saved-model store is Git-ignored and must be retained.

This full Run All took **98.19 seconds** locally with advanced search disabled. Threshold selection and validation logging took **21.49 seconds**, including three pairs of fold fits. These local times are not an end-to-end PBI 10 measurement. Threshold optimization is a separate exploratory function and is not called by normal `train_benchmark` retraining.

Each threshold-optimized local MLflow run stores the fitted model, its preprocessor and `model/inference.json` with the selected threshold and feature order. The saved artifacts were reloaded and reproduced the full validation metrics. Latest runs: CatBoost `ad724329187b4f99a3845cf8abadb0aa`, LightGBM `fad3c3f16b9a4476aecab034eb002831`, random forest `740d1cfc49be43808a6ca270d4fffdfe`. Earlier runs remain intact. Only load trusted artifacts.

## PBI 06 finalization and acceptance status

The frozen notebook comparison retains all 15 original runs and does not run hyperparameter or threshold searches on Run All. It requires the original local MLflow database and artifacts and reports an error if they are missing. The current historical leader is CatBoost at threshold 0.45684071728727366 (validation F1 0.5115); LightGBM at 0.5 is close at 0.5103 and has higher recall with more false positives. Neither is automatically registered as champion.

| Issue #6 acceptance criterion | Status and evidence |
| --- | --- |
| At least two versions using customer and order information | **Partial:** multiple distinct models and versions were trained, but customer snapshot fields are excluded until point-in-time availability is proven. |
| Reproducible split and justified fraud metrics | **Met locally:** whole-date chronological 70/15/15, fraud F1, precision/recall, AP, ROC-AUC, accuracy, confusion counts and timings are documented. Historical full-data EDA limits holdout independence. |
| Each version logged to shared MLflow with retrievable artifact | **Blocked:** all 15 original versions have locally verified parameters, metrics, identity, model, preprocessor, threshold, confusion matrix and PR plot. Shared AWS MLflow has not been verified; local run IDs are not shared-service evidence. |
| Compare versions and explain diagnosis and change | **Met locally:** baseline, initial tuning, advanced tuning and training-only threshold comparisons and their trade-offs are documented above. |
| Callable routine, dependencies and invocation for PBI 10 | **Met locally, integration pending:** `train_selected_models` accepts prepared training/validation frames, refits the two fixed configurations, applies fixed thresholds and logs artifacts. PBI 10 owner agreement and end-to-end invocation remain to be verified. |

The fixed retraining interface was exercised on the base split without accessing test features or labels. Local runs **CatBoost `a32ff6b4cdae4476ac3be5b79e0ed56a`** and **LightGBM `9d929ed60986442c85394bf37444ca22`** reproduced the historical validation F1 scores of 0.5115 and 0.5103 respectively. Model, preprocessor and inference-threshold artifacts were reloaded; all validation predictions and probabilities matched the corresponding original runs exactly. Individual fit times were 3.24 s and 0.77 s; this is not the full PBI 10 pipeline time. The interface can accept an explicitly configured tracking URI, but no AWS connection was attempted.
