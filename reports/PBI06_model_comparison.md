# PBI 06 — Seven-family model comparison

The rebuilt workflow evaluated all seven model families on the frozen chronological development split. Local MLflow parent run `939506bd3eb6434c8489d5871ea9778e` in `PBI06_HanQian`.

## Ranked validation leaderboard

| Family | Baseline F1 | Ordinary F1 | Advanced F1 | Selected | F1 at 0.5 | Threshold | Optimized F1 | Precision | Recall | AP | ROC-AUC | TN | FP | FN | TP | Fit s | Predict s | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForestClassifier | 0.540459 | 0.543441 | 0.546150 | advanced | 0.546150 | 0.510997 | 0.543502 | 0.491539 | 0.607751 | 0.523255 | 0.851193 | 112,828 | 12,410 | 7,743 | 11,997 | 8.457462 | 0.113309 | `adc11d5eedb54dc9a3086cad7e6c4270` |
| XGBClassifier | 0.543524 | 0.544585 | 0.543889 | advanced | 0.543889 | 0.547957 | 0.528035 | 0.521436 | 0.534802 | 0.523368 | 0.848380 | 115,549 | 9,689 | 9,183 | 10,557 | 0.786290 | 0.027484 | `289945f86f2449ea8fe7ff2421249eaa` |
| GradientBoostingClassifier | 0.532694 | 0.525190 | 0.537374 | advanced | 0.537374 | 0.528222 | 0.526210 | 0.511904 | 0.541337 | 0.523780 | 0.847646 | 115,049 | 10,189 | 9,054 | 10,686 | 17.316130 | 0.082492 | `5e4b1bd4257f4e98920647519f59550c` |
| LGBMClassifier | 0.540652 | 0.545418 | 0.545418 | ordinary | 0.545418 | 0.580598 | 0.522158 | 0.528457 | 0.516008 | 0.521865 | 0.847562 | 116,149 | 9,089 | 9,554 | 10,186 | 0.803293 | 0.130568 | `8fd15510396444e29e0308f423fe97df` |
| CatBoostClassifier | 0.538475 | 0.546434 | 0.546434 | ordinary | 0.546434 | 0.576291 | 0.520806 | 0.533043 | 0.509119 | 0.524951 | 0.850076 | 116,434 | 8,804 | 9,690 | 10,050 | 3.006833 | 0.007761 | `652ffc4da1894319aa859515e14c2fce` |
| LogisticRegression | 0.477997 | 0.517770 | 0.517770 | ordinary | 0.517770 | 0.489344 | 0.520114 | 0.472318 | 0.578673 | 0.507008 | 0.840845 | 112,476 | 12,762 | 8,317 | 11,423 | 1.713572 | 0.001077 | `2a91d0ecab36428989a0efecc96c80e6` |
| HistGradientBoostingClassifier | 0.539990 | 0.537184 | 0.537184 | ordinary | 0.537184 | 0.545706 | 0.517620 | 0.526572 | 0.508967 | 0.515904 | 0.848068 | 116,205 | 9,033 | 9,693 | 10,047 | 1.890102 | 0.082498 | `bec883e5805e4614aaabe32dc0db1f15` |

## Training-selected configurations

| Family | Selected stage | Parameters / weight | Mean inner F1 | Std F1 | Mean inner AP | Highest observed F1 stage on validation |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | ordinary | `{"C": 20, "max_iter": 200}` / weight 0.6 | 0.405487 | 0.020625 | 0.374099 | ordinary |
| RandomForestClassifier | advanced | `{"bootstrap": true, "criterion": "entropy", "max_depth": 19, "max_features": "sqrt", "max_samples": 0.55, "min_samples_leaf": 15, "min_samples_split": 160, "n_estimators": 55}` / weight 0.6 | 0.450711 | 0.011909 | 0.431987 | advanced |
| XGBClassifier | advanced | `{"colsample_bytree": 0.85, "gamma": 1, "learning_rate": 0.07, "max_bin": 256, "max_depth": 5, "min_child_weight": 1, "n_estimators": 110, "reg_alpha": 3, "reg_lambda": 4, "subsample": 0.65}` / weight 0.6 | 0.450620 | 0.012722 | 0.440680 | ordinary |
| LGBMClassifier | ordinary | `{"learning_rate": 0.07, "n_estimators": 120, "num_leaves": 40}` / weight 0.6 | 0.448232 | 0.014751 | 0.442298 | ordinary |
| CatBoostClassifier | ordinary | `{"depth": 7, "iterations": 155, "learning_rate": 0.08}` / weight 0.6 | 0.448758 | 0.012091 | 0.440850 | ordinary |
| HistGradientBoostingClassifier | ordinary | `{"learning_rate": 0.04, "max_iter": 70, "max_leaf_nodes": 24}` / weight 0.6 | 0.449409 | 0.015719 | 0.430314 | baseline |
| GradientBoostingClassifier | advanced | `{"learning_rate": 0.075, "loss": "log_loss", "max_depth": 3, "max_features": 0.4, "min_samples_leaf": 45, "min_samples_split": 90, "n_estimators": 110, "subsample": 1.0}` / weight 0.6 | 0.443597 | 0.016140 | 0.425411 | advanced |

## Threshold effects on validation

| Family | F1 at 0.5 | Optimized F1 | F1 change | FP at 0.5 | FP optimized | FN at 0.5 | FN optimized | Calibration F1 at 0.5 | Calibration F1 optimized |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForestClassifier | 0.546150 | 0.543502 | -0.002647 | 12,968 | 12,410 | 7,453 | 7,743 | 0.447635 | 0.449287 |
| XGBClassifier | 0.543889 | 0.528035 | -0.015854 | 12,550 | 9,689 | 7,679 | 9,183 | 0.446089 | 0.449784 |
| GradientBoostingClassifier | 0.537374 | 0.526210 | -0.011164 | 12,791 | 10,189 | 7,788 | 9,054 | 0.441153 | 0.446861 |
| LGBMClassifier | 0.545418 | 0.522158 | -0.023260 | 12,919 | 9,089 | 7,494 | 9,554 | 0.446799 | 0.450477 |
| CatBoostClassifier | 0.546434 | 0.520806 | -0.025628 | 13,176 | 8,804 | 7,366 | 9,690 | 0.446149 | 0.448648 |
| LogisticRegression | 0.517770 | 0.520114 | 0.002344 | 11,936 | 12,762 | 8,675 | 8,317 | 0.426202 | 0.427372 |
| HistGradientBoostingClassifier | 0.537184 | 0.517620 | -0.019564 | 12,145 | 9,033 | 8,031 | 9,693 | 0.443122 | 0.446246 |

## Interpretation

- Highest optimized validation F1 was **RandomForestClassifier**, 0.543502.
- Highest validation AP was **CatBoostClassifier**, 0.524951.
- Highest recall was **RandomForestClassifier**, 0.607751.
- Highest precision was **CatBoostClassifier**, 0.533043.
- Lowest FPR was **CatBoostClassifier**, 0.070298.
- Fastest fit was **XGBClassifier**, 0.786290.
- Best improvement over baseline was **LogisticRegression**, +0.042117 F1.
- Highest comparable training-fold F1 was **RandomForestClassifier**, 0.450711.
- Highest observed F1 across the displayed stages was **CatBoostClassifier**, 0.546434 at cutoff 0.500000 (ordinary).

Cutoff calibration reduced outer-validation F1 for RandomForestClassifier, XGBClassifier, GradientBoostingClassifier, LGBMClassifier, CatBoostClassifier, HistGradientBoostingClassifier.

The top optimized model would flag 24,407 of 144,978 validation orders, of which 12,410 were legitimate. It missed 7,743 fraud orders. Review staffing and the cost of missing fraud should inform the PBI 09 comparison.

The earlier XGBoost F1 was **0.5561** at threshold 0.5. The rebuilt training-selected XGBoost reached **0.543889** at 0.5 and **0.528035** at the training-selected cutoff. Both studies used eight features, the frozen outer validation dates, the same 525,226 fitting rows and the same 135,547 calibration rows. The earlier study used fixed parameters inherited from seven-feature evidence and sparse XGBoost inputs. Its 0.5561 was the fixed-cutoff result before calibration. This rebuild selects among eight-feature trials using folds entirely before calibration and supplies dense inputs so encoded zeros keep their numeric meaning. The observed difference spans these changes.

Full model metrics, exact search spaces and evidence are in [the implementation report](PBI06_training_rebuild_report.md).

<details>
<summary>Historical PBI 06 evidence before the rebuild</summary>

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

## Final seven-model baseline, tuning and threshold evaluation

This additive experiment ran all seven requested model families. The original 15 model versions and their runs above remain unchanged. The final search used **three configurations per model and two expanding chronological whole-date folds** (42 candidate fits). Mean inner-fold fraud F1 at threshold 0.5 selected hyperparameters. The selected model was refit on the full training partition, then its threshold was selected from pooled training-only OOF probabilities. The selected model’s folds were also used for hyperparameter selection, so the OOF threshold estimate is optimistic. Outer validation was used only for final comparisons.

The frozen dates and sizes were training **660,773** (2022-11-12–2023-03-14) and validation **144,978** (2023-03-15–2023-04-13). No test feature or label columns were loaded. The local one-time membership manifest contains only SHA-256 hashes of order keys; order keys and dates were read as split metadata. Earlier PBI 01 full-history EDA is retained as historical evidence and was not test-blind.

The corrected final experiment is `19583bb26bce4c79ba8604592944f624` in local MLflow experiment `PBI06_FinalOptimization`. It has **7 baseline runs, 42 inner-fold candidate runs, 7 tuned-at-0.5 runs and 7 threshold-optimized runs**. Candidate model-fit time summed to **325.09 seconds**; the search loop took **331.56 seconds** including preprocessing and MLflow logging. The complete experiment took **453.95 seconds**.

| Model | Baseline F1 | Tuned F1 @ 0.5 | Optimized F1 | Δ vs baseline | Threshold | Precision | Recall | AP | ROC-AUC | Accuracy | TN | FP | FN | TP | Final fit s | Candidate fit s | Predict s | Baseline run | Tuned run | Threshold run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| xgboost | 0.4410 | 0.5119 | **0.5129** | +0.0719 | 0.560731 | 0.4442 | 0.6068 | 0.4779 | 0.8242 | 0.8431 | 110,248 | 14,990 | 7,762 | 11,978 | 0.889 | 3.43 | 0.058 | `f7fae4b7e9314fe8b4da5b3ea52d85d9` | `5917b71a9dac4e91a6374b0057dea984` | `b9f84bf7c76f4a1f9131f0b7ff444230` |
| lightgbm | 0.4409 | 0.5114 | **0.5106** | +0.0697 | 0.560070 | 0.4429 | 0.6027 | 0.4851 | 0.8246 | 0.8427 | 110,271 | 14,967 | 7,842 | 11,898 | 0.887 | 4.55 | 0.095 | `3db0a44d3a5c40bfa3c1a3a3656c4db4` | `8a9e89cab6d445ee89737e130294e01c` | `1846ad55ad974483b5258b3a155272df` |
| random_forest | 0.4411 | 0.5010 | **0.5101** | +0.0690 | 0.448313 | 0.4405 | 0.6057 | 0.4919 | 0.8245 | 0.8416 | 110,050 | 15,188 | 7,783 | 11,957 | 22.562 | 161.18 | 0.099 | `23959e83f5334153bc7f3f572572da74` | `c0177ca17cc5472487b92e61abc8510a` | `eabb91ccc7f045c28d3cc664b656e793` |
| gradient_boosting | 0.4361 | 0.5095 | **0.5080** | +0.0720 | 0.542369 | 0.4400 | 0.6009 | 0.4629 | 0.8199 | 0.8415 | 110,142 | 15,096 | 7,878 | 11,862 | 37.638 | 120.91 | 0.074 | `df9f31a0d8e8428881c362dbb16bf20c` | `1b23e11e3f1b43a4bfdfa202786b339e` | `b7fd78632a3d42c3b6c253503ebbf5a8` |
| hist_gradient_boosting | 0.4398 | 0.5122 | **0.5074** | +0.0677 | 0.570476 | 0.4479 | 0.5852 | 0.4775 | 0.8242 | 0.8453 | 111,002 | 14,236 | 8,189 | 11,551 | 4.278 | 15.14 | 0.147 | `fb73b04d9a5f408889a7e1f4b84e049b` | `13bb250dbac1448f9d106f6a3694e9e5` | `f0cdea3f63fb4e12a36f719086a377a6` |
| catboost | 0.4443 | 0.5090 | **0.5068** | +0.0626 | 0.573579 | 0.4492 | 0.5814 | 0.4734 | 0.8228 | 0.8459 | 111,167 | 14,071 | 8,263 | 11,477 | 2.563 | 10.85 | 0.006 | `35ab93b45e3d4e75b7ab24f585638683` | `8a9a8964522345c98e8cc572acf6811e` | `65281b45bf024333be4fe5ac48645265` |
| logistic_regression | 0.4246 | 0.4765 | **0.4799** | +0.0553 | 0.557376 | 0.4186 | 0.5623 | 0.4344 | 0.8111 | 0.8341 | 109,821 | 15,417 | 8,641 | 11,099 | 2.407 | 9.03 | 0.001 | `bfcb1f4cddd74871a3847cc86f86d453` | `64f24ab20fad4040b8b3cd43afbf24bb` | `306ea24b2edb44e194b8c5a5cf73faf4` |

### Selected configurations

- **xgboost:** `n_estimators=120`, `max_depth=5`, `learning_rate=0.08`, `reg_lambda=5`, `subsample=0.9`, `colsample_bytree=0.8`. Selected threshold **0.560731**.

- **lightgbm:** `n_estimators=120`, `num_leaves=23`, `learning_rate=0.08`, `reg_lambda=5`, `min_child_samples=50`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.8`. Selected threshold **0.560070**.

- **random_forest:** `n_estimators=60`, `max_depth=12`, `min_samples_leaf=30`, `max_features=0.8`. Selected threshold **0.448313**.

- **gradient_boosting:** `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`, `subsample=0.8`, `min_samples_leaf=40`. Selected threshold **0.542369**.

- **hist_gradient_boosting:** `max_iter=120`, `max_leaf_nodes=31`, `learning_rate=0.05`, `l2_regularization=5.0`. Selected threshold **0.570476**.

- **catboost:** `iterations=120`, `depth=5`, `learning_rate=0.06`, `l2_leaf_reg=5`, `subsample=0.9`, `bootstrap_type=Bernoulli`. Selected threshold **0.573579**.

- **logistic_regression:** `C=1.0`, `max_iter=500`. Selected threshold **0.557376**.

### Result and limitations

**XGBoost ranks first** at optimized validation fraud F1 **0.5129**, improving from baseline 0.4410 and tuned-at-0.5 0.5119. LightGBM is second at 0.5106 and Random Forest third at 0.5101. Recommend XGBoost and LightGBM for the next team review; keep Random Forest as a close third and note that it has the highest AP (0.4919) of these final thresholded candidates. Threshold optimization slightly improved XGBoost, Random Forest and Logistic Regression, and slightly lowered F1 for the other four. Report the measurements as observed; do not assume threshold tuning always helps.

The 0.5 baseline class weighting yields high fraud recall but many false positives. The tuned/optimized candidates improve precision and reduce false positives at the cost of some recall; review capacity and missed-fraud costs should guide any operational choice. AP and ROC-AUC measure ranking quality independently of the selected threshold.

The seven-model comparison is based on a single outer validation period, and the same inner folds informed hyperparameter and threshold selection. Treat the close model ranking and training OOF F1 as exploratory, not as an unbiased final performance estimate. The held-out test data remains sealed. Shared AWS MLflow is unverified; point-in-time-safe customer-history features remain unresolved and excluded; PBI 10 integration and its three-minute demonstration are still pending.

## Shared eight-feature comparison (additive, 17 November 2026)

This is a new experiment family logged to `PBI06_SharedEightFeature`; no historical seven-feature runs or metrics were changed. It reuses the selected seven-model configurations for XGBoost, LightGBM and HistGradientBoosting without searching. Preprocessing and models are fit on the oldest 80% of the outer training dates; the newest 20% of whole training dates selects each threshold by fraud F1. The same fitted model scores calibration and outer validation. Validation metrics do not select thresholds, and sealed test records are not passed to the training function. The older full-training threshold-0.5 baseline numbers below are historical context and are not a controlled single-variable comparison because fitting rows and features differ.

| Model | Historical 7-feature validation F1 | New 8-feature F1 at 0.5 | New 8-feature threshold / F1 | Precision | Recall | AP | ROC-AUC | FP | FN | Fit seconds | Prediction seconds | New local MLflow run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| XGBoost | 0.5129 | 0.5561 | 0.6170 / 0.5283 | 0.5179 | 0.5392 | 0.5245 | 0.8482 | 9,909 | 9,097 | 0.759 | 0.051 | `3ee48762b5dc47328044962b586f3e61` |
| LightGBM | 0.5106 | 0.5533 | 0.6314 / 0.5238 | 0.5208 | 0.5268 | 0.5227 | 0.8483 | 9,568 | 9,341 | 0.808 | 0.109 | `5bb07b0208f244cd882ae044e205122b` |
| HistGradientBoosting | 0.5074 | 0.5546 | 0.6321 / 0.5210 | 0.5279 | 0.5143 | 0.5211 | 0.8476 | 9,079 | 9,588 | 2.998 | 0.145 | `48dcd210896d420990e3b8ca41ed53da` |

At fixed 0.5, F1 increased by 0.0432, 0.0427 and 0.0472 respectively relative to the historical seven-feature runs. With the training-only thresholds, F1 is higher than those historical numbers by 0.0154, 0.0132 and 0.0136, respectively, but lower than each new run's own 0.5 result. This highlights calibration/validation drift and the tradeoff from a threshold selected on a separate, earlier training period. These are validation comparisons, not test results. XGBoost has the highest new calibrated validation F1 (0.5283), narrowly ahead of LightGBM (0.5238); neither is a champion without team review and point-in-time feature verification.

The current HGB version receives shared one-hot encoded inputs. Its prior native categorical handling and monotonic constraint were removed because the shared transformed layout does not retain a safe mapping of original-feature constraints to encoded columns. This changes the estimator's inductive bias; the measured result above is a new version and must not be compared as though only the eighth feature changed.

The new results are stored additively with model and preprocessor artifacts, inference threshold metadata, parameters, validation metrics, and confusion/precision-recall figures. The ordinary notebook does not run tuning searches. The first-order timestamp's provenance and the order-to-customer key needed at inference remain unresolved; see the PBI 03 report.

</details>
