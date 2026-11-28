# PBI 06 — Training rebuild implementation report

Local MLflow research run `939506bd3eb6434c8489d5871ea9778e`. Evidence fingerprint `c52849e7a9a22643c6a2ce9a0305ee8aa34f8363ff0a4029e1e8cd39b6cbd1e3`. Search spaces, stage evidence and validation scores are available under this run as `search_spaces.json`, `evidence.json` and `predictions.npz`. Each trial row in the evidence contains its configuration, per-fold scores, runtimes and MLflow IDs. Each fold run contains the estimator and a reference to its training-fitted shared preprocessor.

## A. Cleanup and interfaces

The working tree, branch references, Git history, local MLflow SQLite metadata and artifact inventory were inspected. Source and documentation inspected included the main CA1 notebook, `src/train_hq.py`, `src/feature_engineering.py`, `src/model_evaluation.py`, `src/data_access.py`, `tests/test_train_hq.py`, `tests/test_data_access.py`, `tests/test_feature_engineering.py`, `README.md`, `requirements.txt`, `reports/PBI03_feature_transformations.md` and the previous `reports/PBI06_model_comparison.md`. The PBI 08 branch tree and PBI 07 artifact metadata were checked for integration scope.

Changed files are `ca1 hq file.ipynb`, `src/train_hq.py`, `tests/test_train_hq.py`, `README.md`, `reports/PBI06_model_comparison.md` and this report. `src/pbi06_research.py` is the new research implementation. No files were removed.

The notebook's former seven-feature benchmark, old initial tuning, old advanced tuning, old threshold optimization, historical seven-family comparison, partial three-family eight-feature comparison, associated charts and switching logic, and old PBI 10 handoff cells were replaced with the staged seven-family workflow and its current handoff. Cells through PBI 03 and notebook kernel settings were preserved. Feature and cleaning code, teammate files, requirements and the sealed manifest were preserved. Superseded training and loading functions remain accessible in `train_hq.py` for legacy callers and compatibility tests. The rebuilt notebook invokes only the canonical research entry point.

The PBI 10 callable `train_selected_models(x_train, y_train, x_val, y_val, split_report, tracking_uri=None, experiment_name=..., member=..., batch_name=..., *, configurations=None)` preserves its default fixed CatBoost and LightGBM call. To retrain an agreed rebuilt policy, pass `configurations={family: {"params": selected_params, "weight": selected_weight, "threshold": selected_threshold}}`. The caller supplies verified chronological prepared partitions. The new fixed-fit path uses the same weighting and model input format as research. The refit can shift calibrated probabilities, so its output reports performance at the supplied policy cutoff. Team ETL integration remains separate.

## B. Data leakage and dataset isolation

- The existing development-only CSV loader filters records via composite keys, dates and the sealed manifest before reading full records. PBI 03 runs on already filtered development data. The loader scans key/date metadata to apply the boundary. It supplies no sealed full feature rows or fraud labels.
- PBI 03 joins cleaned orders, customers and labels using validated composite keys. The exact schema is `order_value`, `num_items_ordered`, `order_weekday`, `order_month`, `country_code`, `collect_type`, `payment_method` and `days_since_first_order`. IDs and dates used for joins or splitting stay outside model inputs. Refund values and undated customer totals remain excluded.
- PBI 03 normalizes category text, including payment casing, and masks absent, invalid and future first-order timestamps. Its preprocessor uses median imputation, log1p and robust scaling for amount/count/history, calendar median imputation and one-hot encoding with unseen categories ignored. The point-in-time source and an independent inference-time customer mapping still require verification.
- The research entry point validates aligned schemas, row identities and date metadata, checks the frozen 15 March cutoff, and rejects any dates on or after 14 April. It requires disjoint training and validation row indexes and dates.
- One separate shared preprocessor fits on the fitting rows of each chronological fold and another fits on the earlier outer training rows. Calibration and validation only call transform on the corresponding training-fitted object. The preserved PBI 03 demonstration preprocessor fits on all outer-training rows for its schema display. The PBI 06 experiments fit fresh preprocessors and do not consume that demonstration object. Sample weights are based on each fit's own binary labels. When weight strength is zero all rows receive weight one. Other strengths interpolate between unit and balanced weights. Estimator-level class weights are disabled for these research fits.
- All model choices come from fold F1, then fold AP, then smaller iteration/tree counts and fewer changed parameters. With exact ties, the first candidate wins. Validation scores are descriptive. Calibration uses only the latest whole training dates, which were reserved before search. The same fitted model produces both calibration and validation probabilities. Threshold ties favor the higher cutoff.
- The earlier full-history PBI 01 EDA examined labels before the holdout was established. This rebuild cannot retroactively restore independence from that historical exposure.

| Period | First | Last | Rows | Non-fraud | Fraud |
| --- | --- | --- | --- | --- | --- |
| fit | 2022-11-12 | 2023-02-17 | 525,226 | 472,966 | 52,260 |
| calibration | 2023-02-18 | 2023-03-14 | 135,547 | 120,960 | 14,587 |
| validation | 2023-03-15 | 2023-04-13 | 144,978 | 125,238 | 19,740 |

The development loader supplied 544,534 customer rows, 805,763 order rows and 805,751 label rows. PBI 02 retained all of these rows. Twelve development orders lacked labels. One label had no matched customer and its unavailable history value was handled as missing.

Outer training contained 660,773 matched rows, including 66,847 fraud orders. The research folds were:

| Fold | Fit from | Fit to | Fit rows | Assess from | Assess to | Assess rows |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-11-12 | 2023-01-07 | 312,916 | 2023-01-08 | 2023-01-27 | 103,784 |
| 2 | 2022-11-12 | 2023-01-27 | 416,700 | 2023-01-28 | 2023-02-17 | 108,526 |

## C. Baseline diagnosis

| Family | Precision | Recall | False alarms | Missed fraud |
| --- | --- | --- | --- | --- |
| RandomForestClassifier | 0.413126 | 0.781256 | 21,908 | 4,318 |
| XGBClassifier | 0.418809 | 0.774012 | 21,203 | 4,461 |
| GradientBoostingClassifier | 0.403904 | 0.782067 | 22,784 | 4,302 |
| LGBMClassifier | 0.415569 | 0.773455 | 21,472 | 4,472 |
| CatBoostClassifier | 0.410798 | 0.781307 | 22,121 | 4,317 |
| LogisticRegression | 0.341103 | 0.798430 | 30,445 | 3,979 |
| HistGradientBoostingClassifier | 0.414875 | 0.773151 | 21,525 | 4,478 |

Logistic regression sent the most legitimate orders to review (30,445). Random Forest and CatBoost recovered more fraud than most baselines, with over 21,000 false alarms each. XGBoost had the strongest baseline F1, while XGBoost, LightGBM and HistGradientBoosting still sent over 21,000 legitimate orders to review and missed over 4,400 fraud orders each. GradientBoosting missed fewer fraud orders than these three and had more false alarms. These observations motivate examining weighting and capacity. The PBI 06 notebook displays baseline ranking, PR curves and confusion matrices.

## D–E. Ordinary and advanced searches

Seed 42 controls estimator and parameter sampling. Each ordinary search starts with the matching baseline. Each advanced search starts with the ordinary winner. An exact repeated reference uses the same fold result. The search samples joint parameter combinations rather than enumerating a Cartesian product. Parameter values and compatibility rules are below. Weight strengths are [0.0, 0.35, 0.6, 0.8, 1.0] for both search stages. The ordinary selected configuration used strength 0.6 for all seven families. The advanced space tested adjacent weighting values together with new parameter combinations. Search scope was established from training-only evidence and the approved budget.

- Logistic regression uses `l1_ratio` for regularization mixing in the installed scikit-learn, with SAGA for nonzero ratios and valid L2 solvers otherwise.
- Random forest only sets `max_samples` when bootstrap is enabled.
- LightGBM enables `subsample_freq` when rows are subsampled and caps leaves for depth-limited trees.
- CatBoost uses bagging temperature for Bayesian bootstrap or subsampling for Bernoulli bootstrap.
- HistGradientBoosting uses explicit iteration counts without randomized early stopping.
- GradientBoosting explores loss rather than criterion because the installed version reports criterion as deprecated with no effect.
- XGBoost receives dense transformed features so one-hot zeros retain their numeric meaning. The other estimators receive the same values with appropriate sparse or dense representation.

### Ordinary search space

- **LogisticRegression** `{"C": [0.03, 0.2, 1, 5, 20], "max_iter": [200, 350]}`
- **RandomForestClassifier** `{"max_depth": [8, 12, 16], "min_samples_leaf": [25, 65], "n_estimators": [25, 45, 65]}`
- **XGBClassifier** `{"learning_rate": [0.04, 0.08, 0.13], "max_depth": [3, 5, 7], "n_estimators": [65, 100, 145]}`
- **LGBMClassifier** `{"learning_rate": [0.035, 0.07, 0.12], "n_estimators": [70, 120, 170], "num_leaves": [12, 24, 40]}`
- **CatBoostClassifier** `{"depth": [4, 5, 7], "iterations": [75, 115, 155], "learning_rate": [0.04, 0.08, 0.13]}`
- **HistGradientBoostingClassifier** `{"learning_rate": [0.04, 0.08, 0.13], "max_iter": [70, 110, 150], "max_leaf_nodes": [12, 24, 40]}`
- **GradientBoostingClassifier** `{"learning_rate": [0.045, 0.09, 0.15], "max_depth": [2, 3, 4], "n_estimators": [45, 65, 90]}`

### Advanced search space

- **LogisticRegression** `{"C": [0.005, 0.05, 0.4, 2, 12, 40], "l1_ratio": [0.0, 0.25, 0.65, 1.0], "max_iter": [180, 350, 500], "tol": [3e-05, 0.0001, 0.0005]}`
- **RandomForestClassifier** `{"bootstrap": [true, false], "criterion": ["gini", "entropy"], "max_depth": [8, 13, 19], "max_features": ["sqrt", 0.6, 1.0], "max_samples": [0.55, 0.8, null], "min_samples_leaf": [15, 40, 85], "min_samples_split": [30, 80, 160], "n_estimators": [30, 55, 85]}`
- **XGBClassifier** `{"colsample_bytree": [0.65, 0.85, 1.0], "gamma": [0, 0.25, 1], "learning_rate": [0.03, 0.07, 0.14], "max_bin": [96, 192, 256], "max_depth": [3, 5, 7], "min_child_weight": [1, 12, 45], "n_estimators": [60, 110, 170], "reg_alpha": [0, 0.5, 3], "reg_lambda": [1, 4, 10], "subsample": [0.65, 0.82, 1.0]}`
- **LGBMClassifier** `{"colsample_bytree": [0.6, 0.8, 1.0], "learning_rate": [0.03, 0.065, 0.13], "max_depth": [5, 8, -1], "min_child_samples": [25, 65, 130], "min_child_weight": [0.001, 0.1, 1.0], "min_split_gain": [0, 0.1, 0.5], "n_estimators": [70, 120, 180], "num_leaves": [12, 24, 44], "reg_alpha": [0, 0.3, 3], "reg_lambda": [0, 3, 10], "subsample": [0.65, 0.85, 1.0]}`
- **CatBoostClassifier** `{"bagging_temperature": [0.2, 1.0, 2.0], "bootstrap_type": ["Bayesian", "Bernoulli"], "border_count": [48, 96, 160], "depth": [4, 6, 8], "iterations": [75, 125, 180], "l2_leaf_reg": [2, 6, 12], "learning_rate": [0.035, 0.075, 0.13], "random_strength": [0.2, 1, 3], "subsample": [0.65, 0.8, 0.95]}`
- **HistGradientBoostingClassifier** `{"l2_regularization": [0, 2, 10], "learning_rate": [0.03, 0.065, 0.13], "max_bins": [64, 128, 255], "max_depth": [4, 7, null], "max_iter": [70, 120, 180], "max_leaf_nodes": [12, 24, 48], "min_samples_leaf": [30, 80, 160]}`
- **GradientBoostingClassifier** `{"learning_rate": [0.035, 0.075, 0.145], "loss": ["log_loss", "exponential"], "max_depth": [2, 3, 4], "max_features": [0.4, 0.7, null], "min_samples_leaf": [15, 45, 90], "min_samples_split": [30, 90, 180], "n_estimators": [45, 75, 110], "subsample": [0.6, 0.8, 1.0]}`

| Stage | Family | Configurations | New configurations | New fold fits | Failed configs | Mean fold F1 | Std F1 | Mean fold AP | Validation F1 | Summed fold fit s | Winning parameters / weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | RandomForestClassifier | 1 | 1 | 2 | 0 | 0.400879 | 0.015267 | 0.423441 | 0.540459 | 2.772014 | `{}` / weight 1.0 |
| baseline | XGBClassifier | 1 | 1 | 2 | 0 | 0.408996 | 0.014202 | 0.439196 | 0.543524 | 0.774049 | `{}` / weight 1.0 |
| baseline | GradientBoostingClassifier | 1 | 1 | 2 | 0 | 0.397851 | 0.014125 | 0.426453 | 0.532694 | 40.368795 | `{}` / weight 1.0 |
| baseline | LGBMClassifier | 1 | 1 | 2 | 0 | 0.407935 | 0.015431 | 0.438648 | 0.540652 | 0.918116 | `{}` / weight 1.0 |
| baseline | CatBoostClassifier | 1 | 1 | 2 | 0 | 0.398807 | 0.015854 | 0.427417 | 0.538475 | 2.154386 | `{}` / weight 1.0 |
| baseline | LogisticRegression | 1 | 1 | 2 | 0 | 0.354272 | 0.013754 | 0.374179 | 0.477997 | 2.508223 | `{}` / weight 1.0 |
| baseline | HistGradientBoostingClassifier | 1 | 1 | 2 | 0 | 0.406723 | 0.016489 | 0.435772 | 0.539990 | 2.280373 | `{}` / weight 1.0 |
| ordinary | RandomForestClassifier | 10 | 9 | 18 | 0 | 0.448739 | 0.012149 | 0.424616 | 0.543441 | 56.129495 | `{"max_depth": 12, "min_samples_leaf": 65, "n_estimators": 25}` / weight 0.6 |
| ordinary | XGBClassifier | 10 | 9 | 18 | 0 | 0.449504 | 0.013006 | 0.442235 | 0.544585 | 9.339011 | `{"learning_rate": 0.08, "max_depth": 7, "n_estimators": 100}` / weight 0.6 |
| ordinary | GradientBoostingClassifier | 10 | 9 | 18 | 0 | 0.441277 | 0.013976 | 0.418229 | 0.525190 | 315.368818 | `{"learning_rate": 0.09, "max_depth": 3, "n_estimators": 45}` / weight 0.6 |
| ordinary | LGBMClassifier | 10 | 9 | 18 | 0 | 0.448232 | 0.014751 | 0.442298 | 0.545418 | 9.109252 | `{"learning_rate": 0.07, "n_estimators": 120, "num_leaves": 40}` / weight 0.6 |
| ordinary | CatBoostClassifier | 10 | 9 | 18 | 0 | 0.448758 | 0.012091 | 0.440850 | 0.546434 | 28.074478 | `{"depth": 7, "iterations": 155, "learning_rate": 0.08}` / weight 0.6 |
| ordinary | LogisticRegression | 10 | 9 | 18 | 0 | 0.405487 | 0.020625 | 0.374099 | 0.517770 | 21.383309 | `{"C": 20, "max_iter": 200}` / weight 0.6 |
| ordinary | HistGradientBoostingClassifier | 10 | 9 | 18 | 0 | 0.449409 | 0.015719 | 0.430314 | 0.537184 | 25.698478 | `{"learning_rate": 0.04, "max_iter": 70, "max_leaf_nodes": 24}` / weight 0.6 |
| advanced | RandomForestClassifier | 24 | 23 | 46 | 0 | 0.450711 | 0.011909 | 0.431987 | 0.546150 | 469.385337 | `{"bootstrap": true, "criterion": "entropy", "max_depth": 19, "max_features": "sqrt", "max_samples": 0.55, "min_samples_leaf": 15, "min_samples_split": 160, "n_estimators": 55}` / weight 0.6 |
| advanced | XGBClassifier | 24 | 23 | 46 | 0 | 0.450620 | 0.012722 | 0.440680 | 0.543889 | 27.566407 | `{"colsample_bytree": 0.85, "gamma": 1, "learning_rate": 0.07, "max_bin": 256, "max_depth": 5, "min_child_weight": 1, "n_estimators": 110, "reg_alpha": 3, "reg_lambda": 4, "subsample": 0.65}` / weight 0.6 |
| advanced | GradientBoostingClassifier | 24 | 23 | 46 | 0 | 0.443597 | 0.016140 | 0.425411 | 0.537374 | 444.475450 | `{"learning_rate": 0.075, "loss": "log_loss", "max_depth": 3, "max_features": 0.4, "min_samples_leaf": 45, "min_samples_split": 90, "n_estimators": 110, "subsample": 1.0}` / weight 0.6 |
| advanced | LGBMClassifier | 24 | 23 | 46 | 0 | 0.448232 | 0.014751 | 0.442298 | 0.545418 | 26.162299 | `{"learning_rate": 0.07, "n_estimators": 120, "num_leaves": 40}` / weight 0.6 |
| advanced | CatBoostClassifier | 24 | 23 | 46 | 0 | 0.448758 | 0.012091 | 0.440850 | 0.546434 | 72.983299 | `{"depth": 7, "iterations": 155, "learning_rate": 0.08}` / weight 0.6 |
| advanced | LogisticRegression | 24 | 23 | 46 | 0 | 0.405487 | 0.020625 | 0.374099 | 0.517770 | 215.390537 | `{"C": 20, "max_iter": 200}` / weight 0.6 |
| advanced | HistGradientBoostingClassifier | 24 | 23 | 46 | 0 | 0.449409 | 0.015719 | 0.430314 | 0.537184 | 83.562294 | `{"learning_rate": 0.04, "max_iter": 70, "max_leaf_nodes": 24}` / weight 0.6 |

Total independent fold fits attempted **462**.

The full approved budget ran without reduction. The training-fold winners stayed at interior weight strength 0.6 and the nearest alternative settings were evaluated within the advanced budget. The broader search selected new versions for Random Forest, XGBoost and GradientBoosting and retained the ordinary configuration for the other four families. These training results supported stopping at the approved budget.

No failed trial fits were recorded.

### Fold metrics for the stage winners

| Stage | Family | Fold | F1 | AP | Fit seconds | Run |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | LogisticRegression | 1 | 0.340518 | 0.343549 | 1.190622 | `a0edd2343b7c472e809d1736cfcf3322` |
| baseline | LogisticRegression | 2 | 0.368027 | 0.404808 | 1.317601 | `6d5382784b8c41088a41abc84c6ebc61` |
| baseline | RandomForestClassifier | 1 | 0.385612 | 0.395842 | 1.107856 | `5406b52b1096483cb41f65604c79f0be` |
| baseline | RandomForestClassifier | 2 | 0.416146 | 0.451040 | 1.664158 | `4cd7522e14ee420c862286f45dd7612d` |
| baseline | XGBClassifier | 1 | 0.394794 | 0.418553 | 0.333379 | `f9d98c9b67a048ffbd773888d05664ce` |
| baseline | XGBClassifier | 2 | 0.423199 | 0.459839 | 0.440671 | `cae5cfadccfe45929119d84122010a02` |
| baseline | LGBMClassifier | 1 | 0.392504 | 0.418581 | 0.412652 | `409cb5caaa944e75926f840f739952bc` |
| baseline | LGBMClassifier | 2 | 0.423366 | 0.458716 | 0.505464 | `03725cab58794a5eaf3335c7fe373b4c` |
| baseline | CatBoostClassifier | 1 | 0.382953 | 0.407991 | 0.961481 | `c8b080384ab94408b47bb2426ae7a79c` |
| baseline | CatBoostClassifier | 2 | 0.414661 | 0.446842 | 1.192905 | `daac6d28caec494cbfc8949e27ee9efb` |
| baseline | HistGradientBoostingClassifier | 1 | 0.390235 | 0.415993 | 0.955587 | `93fc0d46791443babb50480bd4234f2c` |
| baseline | HistGradientBoostingClassifier | 2 | 0.423212 | 0.455552 | 1.324786 | `2ae5d588d607411bbf7405bf2e01e923` |
| baseline | GradientBoostingClassifier | 1 | 0.383726 | 0.409564 | 16.662093 | `09160511190d4556aa4d0241336cc13c` |
| baseline | GradientBoostingClassifier | 2 | 0.411976 | 0.443341 | 23.706702 | `95096d0754614c22a39a17990bf7526f` |
| ordinary | LogisticRegression | 1 | 0.384862 | 0.343313 | 1.060796 | `2d6bf0563dfa408fb0d5be59f7637572` |
| ordinary | LogisticRegression | 2 | 0.426111 | 0.404884 | 1.381696 | `f2b53801cbd74e2c85e59df166089b0d` |
| ordinary | RandomForestClassifier | 1 | 0.436590 | 0.403880 | 1.504428 | `6e907340543347a481e3690af29eeb07` |
| ordinary | RandomForestClassifier | 2 | 0.460888 | 0.445352 | 2.120504 | `a7f4f55d118b4aecb2d15d0f6ecb30bf` |
| ordinary | XGBClassifier | 1 | 0.436497 | 0.420543 | 0.522195 | `2f9ae773f9714260b15fd542188028a2` |
| ordinary | XGBClassifier | 2 | 0.462510 | 0.463927 | 0.679989 | `b44c83cd400e43808b57c8ac1241e4a0` |
| ordinary | LGBMClassifier | 1 | 0.433481 | 0.423648 | 0.569410 | `3e440a5861a4453d999b46e84316acae` |
| ordinary | LGBMClassifier | 2 | 0.462982 | 0.460948 | 0.683612 | `da7aa814b177476db8c4f457942771a4` |
| ordinary | CatBoostClassifier | 1 | 0.436667 | 0.421245 | 1.831903 | `4df49d74d7404bdcb52927166ef0616d` |
| ordinary | CatBoostClassifier | 2 | 0.460849 | 0.460455 | 2.389798 | `7ee5f5ac484e4ab29d0e45ac916d671a` |
| ordinary | HistGradientBoostingClassifier | 1 | 0.433690 | 0.413144 | 1.120436 | `0be4d93998004851bbe1822de5fdf328` |
| ordinary | HistGradientBoostingClassifier | 2 | 0.465128 | 0.447484 | 1.474330 | `dfed3a6ade3642faa041d2365742fcc4` |
| ordinary | GradientBoostingClassifier | 1 | 0.427301 | 0.399169 | 9.207171 | `ceecc4a7074547cd913b43a482e75932` |
| ordinary | GradientBoostingClassifier | 2 | 0.455253 | 0.437290 | 13.067785 | `c0d0bc22b374490eaee10c4577af4ecc` |
| advanced | LogisticRegression | 1 | 0.384862 | 0.343313 | 1.060796 | `2d6bf0563dfa408fb0d5be59f7637572` |
| advanced | LogisticRegression | 2 | 0.426111 | 0.404884 | 1.381696 | `f2b53801cbd74e2c85e59df166089b0d` |
| advanced | RandomForestClassifier | 1 | 0.438802 | 0.412152 | 3.481905 | `f9cce4eb6ad44d03ab831156d75c0470` |
| advanced | RandomForestClassifier | 2 | 0.462620 | 0.451821 | 6.461359 | `6aa13035736b447f800094ea08de1015` |
| advanced | XGBClassifier | 1 | 0.437899 | 0.423393 | 0.548577 | `04480897f261406583ac181b1df2bcc0` |
| advanced | XGBClassifier | 2 | 0.463342 | 0.457968 | 0.753049 | `6b9d91f525724a8f928ed4360fc1f4cc` |
| advanced | LGBMClassifier | 1 | 0.433481 | 0.423648 | 0.569410 | `3e440a5861a4453d999b46e84316acae` |
| advanced | LGBMClassifier | 2 | 0.462982 | 0.460948 | 0.683612 | `da7aa814b177476db8c4f457942771a4` |
| advanced | CatBoostClassifier | 1 | 0.436667 | 0.421245 | 1.831903 | `4df49d74d7404bdcb52927166ef0616d` |
| advanced | CatBoostClassifier | 2 | 0.460849 | 0.460455 | 2.389798 | `7ee5f5ac484e4ab29d0e45ac916d671a` |
| advanced | HistGradientBoostingClassifier | 1 | 0.433690 | 0.413144 | 1.120436 | `0be4d93998004851bbe1822de5fdf328` |
| advanced | HistGradientBoostingClassifier | 2 | 0.465128 | 0.447484 | 1.474330 | `dfed3a6ade3642faa041d2365742fcc4` |
| advanced | GradientBoostingClassifier | 1 | 0.427457 | 0.404127 | 9.319242 | `6879d1c3966d4465a16686de9c5b2e13` |
| advanced | GradientBoostingClassifier | 2 | 0.459737 | 0.446696 | 13.013861 | `d1a3047298704e98b79672b2742945a6` |

Full effective base parameters for each family are saved in its baseline MLflow run under `parameters.json`. The tables show overrides and class weight strength. The detailed metrics follow.

### Baseline metrics

| Family | f1 | precision | recall | average_precision | roc_auc | accuracy | false_positive_rate | tn | fp | fn | tp | training_seconds | prediction_seconds | Threshold | Mean inner F1 | MLflow run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForestClassifier | 0.540459 | 0.413126 | 0.781256 | 0.526523 | 0.847699 | 0.819104 | 0.174931 | 103,330 | 21,908 | 4,318 | 15,422 | 2.296275 | 0.038531 | 0.500000 | 0.400879 | `11ad74bc96334e31b93f6b4527e3c544` |
| XGBClassifier | 0.543524 | 0.418809 | 0.774012 | 0.523596 | 0.847153 | 0.822980 | 0.169302 | 104,035 | 21,203 | 4,461 | 15,279 | 0.574243 | 0.017007 | 0.500000 | 0.408996 | `68247f279d9e4ec4b9b6f6c882f255f6` |
| GradientBoostingClassifier | 0.532694 | 0.403904 | 0.782067 | 0.518083 | 0.847198 | 0.813172 | 0.181926 | 102,454 | 22,784 | 4,302 | 15,438 | 32.997962 | 0.061774 | 0.500000 | 0.397851 | `840ba683b76844bea8808dddbc714608` |
| LGBMClassifier | 0.540652 | 0.415569 | 0.773455 | 0.522166 | 0.847642 | 0.821049 | 0.171450 | 103,766 | 21,472 | 4,472 | 15,268 | 0.557900 | 0.080486 | 0.500000 | 0.407935 | `6a6564a5d525432f8a39ec5dc6f72f0a` |
| CatBoostClassifier | 0.538475 | 0.410798 | 0.781307 | 0.524681 | 0.849816 | 0.817641 | 0.176632 | 103,117 | 22,121 | 4,317 | 15,423 | 1.438070 | 0.005274 | 0.500000 | 0.398807 | `1003dca400344927aca47c9e89f61bd2` |
| LogisticRegression | 0.477997 | 0.341103 | 0.798430 | 0.508129 | 0.841734 | 0.762557 | 0.243097 | 94,793 | 30,445 | 3,979 | 15,761 | 1.445832 | 0.001108 | 0.500000 | 0.354272 | `41c73e06a7c44321af16bd66ad2ac612` |
| HistGradientBoostingClassifier | 0.539990 | 0.414875 | 0.773151 | 0.518977 | 0.847360 | 0.820642 | 0.171873 | 103,713 | 21,525 | 4,478 | 15,262 | 1.651477 | 0.069512 | 0.500000 | 0.406723 | `3a4da8756d6f4a6f9fb542949da3b9c7` |

### Ordinary metrics

| Family | f1 | precision | recall | average_precision | roc_auc | accuracy | false_positive_rate | tn | fp | fn | tp | training_seconds | prediction_seconds | Threshold | Mean inner F1 | MLflow run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForestClassifier | 0.543441 | 0.480517 | 0.625329 | 0.506899 | 0.849854 | 0.856937 | 0.106557 | 111,893 | 13,345 | 7,396 | 12,344 | 2.956274 | 0.038458 | 0.500000 | 0.448739 | `d483cbecfa624b8f9cd31c6bf5fa1625` |
| XGBClassifier | 0.544585 | 0.487054 | 0.617528 | 0.524144 | 0.846734 | 0.859372 | 0.102509 | 112,400 | 12,838 | 7,550 | 12,190 | 0.843347 | 0.031582 | 0.500000 | 0.449504 | `dd07c62c0e0e43a18d42fc379cb2d85e` |
| GradientBoostingClassifier | 0.525190 | 0.511199 | 0.539970 | 0.516911 | 0.849099 | 0.867063 | 0.081381 | 115,046 | 10,192 | 9,081 | 10,659 | 16.618223 | 0.038415 | 0.500000 | 0.441277 | `c90bd18e4e4447b8b6a4fd51f3f80664` |
| LGBMClassifier | 0.545418 | 0.486628 | 0.620365 | 0.521865 | 0.847562 | 0.859199 | 0.103156 | 112,319 | 12,919 | 7,494 | 12,246 | 0.803293 | 0.130568 | 0.500000 | 0.448232 | `2a1056b2c0bc4759b5c8213d7a0bd443` |
| CatBoostClassifier | 0.546434 | 0.484305 | 0.626849 | 0.524951 | 0.850076 | 0.858310 | 0.105208 | 112,062 | 13,176 | 7,366 | 12,374 | 3.006833 | 0.007761 | 0.500000 | 0.448758 | `8c3b22e3730746a891f589c26940091c` |
| LogisticRegression | 0.517770 | 0.481066 | 0.560537 | 0.507008 | 0.840845 | 0.857834 | 0.095307 | 113,302 | 11,936 | 8,675 | 11,065 | 1.713572 | 0.001077 | 0.500000 | 0.405487 | `95ba837d12ad4f868762a44500e65c88` |
| HistGradientBoostingClassifier | 0.537184 | 0.490861 | 0.593161 | 0.515904 | 0.848068 | 0.860834 | 0.096975 | 113,093 | 12,145 | 8,031 | 11,709 | 1.890102 | 0.082498 | 0.500000 | 0.449409 | `459a36a85b36415d8dd81c2fe5f898e7` |

### Advanced metrics

| Family | f1 | precision | recall | average_precision | roc_auc | accuracy | false_positive_rate | tn | fp | fn | tp | training_seconds | prediction_seconds | Threshold | Mean inner F1 | MLflow run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForestClassifier | 0.546150 | 0.486518 | 0.622442 | 0.523255 | 0.851193 | 0.859144 | 0.103547 | 112,270 | 12,968 | 7,453 | 12,287 | 8.457462 | 0.113309 | 0.500000 | 0.450711 | `231ad02c1df54fe38c2463efe8416cba` |
| XGBClassifier | 0.543889 | 0.490065 | 0.610993 | 0.523368 | 0.848380 | 0.860468 | 0.100209 | 112,688 | 12,550 | 7,679 | 12,061 | 0.786290 | 0.027484 | 0.500000 | 0.450620 | `01a3a6d07d3246d09f809867cab8fe8c` |
| GradientBoostingClassifier | 0.537374 | 0.483046 | 0.605471 | 0.523780 | 0.847646 | 0.858054 | 0.102134 | 112,447 | 12,791 | 7,788 | 11,952 | 17.316130 | 0.082492 | 0.500000 | 0.443597 | `a4598a508f7e4040962f97548e5d7c0b` |
| LGBMClassifier | 0.545418 | 0.486628 | 0.620365 | 0.521865 | 0.847562 | 0.859199 | 0.103156 | 112,319 | 12,919 | 7,494 | 12,246 | 0.803293 | 0.130568 | 0.500000 | 0.448232 | `3831ea7f26f642c0aa31434d956e8663` |
| CatBoostClassifier | 0.546434 | 0.484305 | 0.626849 | 0.524951 | 0.850076 | 0.858310 | 0.105208 | 112,062 | 13,176 | 7,366 | 12,374 | 3.006833 | 0.007761 | 0.500000 | 0.448758 | `33e0bce9dccf44ad8bfe44aa50c958b6` |
| LogisticRegression | 0.517770 | 0.481066 | 0.560537 | 0.507008 | 0.840845 | 0.857834 | 0.095307 | 113,302 | 11,936 | 8,675 | 11,065 | 1.713572 | 0.001077 | 0.500000 | 0.405487 | `63db305ddd604befac13c12759a1a010` |
| HistGradientBoostingClassifier | 0.537184 | 0.490861 | 0.593161 | 0.515904 | 0.848068 | 0.860834 | 0.096975 | 113,093 | 12,145 | 8,031 | 11,709 | 1.890102 | 0.082498 | 0.500000 | 0.449409 | `dd0d4c6d1e7b41f3b695396819def5af` |

### Optimized metrics

| Family | f1 | precision | recall | average_precision | roc_auc | accuracy | false_positive_rate | tn | fp | fn | tp | training_seconds | prediction_seconds | Threshold | Mean inner F1 | MLflow run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForestClassifier | 0.543502 | 0.491539 | 0.607751 | 0.523255 | 0.851193 | 0.860993 | 0.099091 | 112,828 | 12,410 | 7,743 | 11,997 | 8.457462 | 0.113309 | 0.510997 | 0.450711 | `adc11d5eedb54dc9a3086cad7e6c4270` |
| XGBClassifier | 0.528035 | 0.521436 | 0.534802 | 0.523368 | 0.848380 | 0.869829 | 0.077365 | 115,549 | 9,689 | 9,183 | 10,557 | 0.786290 | 0.027484 | 0.547957 | 0.450620 | `289945f86f2449ea8fe7ff2421249eaa` |
| GradientBoostingClassifier | 0.526210 | 0.511904 | 0.541337 | 0.523780 | 0.847646 | 0.867270 | 0.081357 | 115,049 | 10,189 | 9,054 | 10,686 | 17.316130 | 0.082492 | 0.528222 | 0.443597 | `5e4b1bd4257f4e98920647519f59550c` |
| LGBMClassifier | 0.522158 | 0.528457 | 0.516008 | 0.521865 | 0.847562 | 0.871408 | 0.072574 | 116,149 | 9,089 | 9,554 | 10,186 | 0.803293 | 0.130568 | 0.580598 | 0.448232 | `8fd15510396444e29e0308f423fe97df` |
| CatBoostClassifier | 0.520806 | 0.533043 | 0.509119 | 0.524951 | 0.850076 | 0.872436 | 0.070298 | 116,434 | 8,804 | 9,690 | 10,050 | 3.006833 | 0.007761 | 0.576291 | 0.448758 | `652ffc4da1894319aa859515e14c2fce` |
| LogisticRegression | 0.520114 | 0.472318 | 0.578673 | 0.507008 | 0.840845 | 0.854606 | 0.101902 | 112,476 | 12,762 | 8,317 | 11,423 | 1.713572 | 0.001077 | 0.489344 | 0.405487 | `2a91d0ecab36428989a0efecc96c80e6` |
| HistGradientBoostingClassifier | 0.517620 | 0.526572 | 0.508967 | 0.515904 | 0.848068 | 0.870836 | 0.072127 | 116,205 | 9,033 | 9,693 | 10,047 | 1.890102 | 0.082498 | 0.545706 | 0.449409 | `bec883e5805e4614aaabe32dc0db1f15` |

## F. Threshold calibration and drift

| Family | F1 at 0.5 | Optimized F1 | F1 change | FP at 0.5 | FP optimized | FN at 0.5 | FN optimized | Calibration F1 at 0.5 | Calibration F1 optimized |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForestClassifier | 0.546150 | 0.543502 | -0.002647 | 12,968 | 12,410 | 7,453 | 7,743 | 0.447635 | 0.449287 |
| XGBClassifier | 0.543889 | 0.528035 | -0.015854 | 12,550 | 9,689 | 7,679 | 9,183 | 0.446089 | 0.449784 |
| GradientBoostingClassifier | 0.537374 | 0.526210 | -0.011164 | 12,791 | 10,189 | 7,788 | 9,054 | 0.441153 | 0.446861 |
| LGBMClassifier | 0.545418 | 0.522158 | -0.023260 | 12,919 | 9,089 | 7,494 | 9,554 | 0.446799 | 0.450477 |
| CatBoostClassifier | 0.546434 | 0.520806 | -0.025628 | 13,176 | 8,804 | 7,366 | 9,690 | 0.446149 | 0.448648 |
| LogisticRegression | 0.517770 | 0.520114 | 0.002344 | 11,936 | 12,762 | 8,675 | 8,317 | 0.426202 | 0.427372 |
| HistGradientBoostingClassifier | 0.537184 | 0.517620 | -0.019564 | 12,145 | 9,033 | 8,031 | 9,693 | 0.443122 | 0.446246 |

The search enumerated the full precision–recall threshold sequence, evaluating one F1 per distinct score cutoff. Model scores are generated once per partition. Calibration diagnostics below split the reserved dates into two chronological halves for description. The halves never choose another cutoff.

| Family | Period | Fraud rate | Mean score | F1 at chosen cutoff |
| --- | --- | --- | --- | --- |
| RandomForestClassifier | {'first': '2023-02-18', 'last': '2023-03-02', 'rows': 69987} | 0.099590 | 0.278856 | 0.432494 |
| RandomForestClassifier | {'first': '2023-03-03', 'last': '2023-03-14', 'rows': 65560} | 0.116184 | 0.288927 | 0.465522 |
| XGBClassifier | {'first': '2023-02-18', 'last': '2023-03-02', 'rows': 69987} | 0.099590 | 0.278486 | 0.434224 |
| XGBClassifier | {'first': '2023-03-03', 'last': '2023-03-14', 'rows': 65560} | 0.116184 | 0.288746 | 0.464683 |
| GradientBoostingClassifier | {'first': '2023-02-18', 'last': '2023-03-02', 'rows': 69987} | 0.099590 | 0.281682 | 0.427432 |
| GradientBoostingClassifier | {'first': '2023-03-03', 'last': '2023-03-14', 'rows': 65560} | 0.116184 | 0.291905 | 0.465445 |
| LGBMClassifier | {'first': '2023-02-18', 'last': '2023-03-02', 'rows': 69987} | 0.099590 | 0.278880 | 0.436804 |
| LGBMClassifier | {'first': '2023-03-03', 'last': '2023-03-14', 'rows': 65560} | 0.116184 | 0.288877 | 0.463562 |
| CatBoostClassifier | {'first': '2023-02-18', 'last': '2023-03-02', 'rows': 69987} | 0.099590 | 0.278402 | 0.433002 |
| CatBoostClassifier | {'first': '2023-03-03', 'last': '2023-03-14', 'rows': 65560} | 0.116184 | 0.288294 | 0.463544 |
| LogisticRegression | {'first': '2023-02-18', 'last': '2023-03-02', 'rows': 69987} | 0.099590 | 0.286923 | 0.401074 |
| LogisticRegression | {'first': '2023-03-03', 'last': '2023-03-14', 'rows': 65560} | 0.116184 | 0.298820 | 0.452520 |
| HistGradientBoostingClassifier | {'first': '2023-02-18', 'last': '2023-03-02', 'rows': 69987} | 0.099590 | 0.283805 | 0.432350 |
| HistGradientBoostingClassifier | {'first': '2023-03-03', 'last': '2023-03-14', 'rows': 65560} | 0.116184 | 0.293312 | 0.459511 |

## G. Selection and final ranking

| Family | Selected stage | Parameters / weight | Mean inner F1 | Std F1 | Mean inner AP | Highest observed F1 stage on validation |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | ordinary | `{"C": 20, "max_iter": 200}` / weight 0.6 | 0.405487 | 0.020625 | 0.374099 | ordinary |
| RandomForestClassifier | advanced | `{"bootstrap": true, "criterion": "entropy", "max_depth": 19, "max_features": "sqrt", "max_samples": 0.55, "min_samples_leaf": 15, "min_samples_split": 160, "n_estimators": 55}` / weight 0.6 | 0.450711 | 0.011909 | 0.431987 | advanced |
| XGBClassifier | advanced | `{"colsample_bytree": 0.85, "gamma": 1, "learning_rate": 0.07, "max_bin": 256, "max_depth": 5, "min_child_weight": 1, "n_estimators": 110, "reg_alpha": 3, "reg_lambda": 4, "subsample": 0.65}` / weight 0.6 | 0.450620 | 0.012722 | 0.440680 | ordinary |
| LGBMClassifier | ordinary | `{"learning_rate": 0.07, "n_estimators": 120, "num_leaves": 40}` / weight 0.6 | 0.448232 | 0.014751 | 0.442298 | ordinary |
| CatBoostClassifier | ordinary | `{"depth": 7, "iterations": 155, "learning_rate": 0.08}` / weight 0.6 | 0.448758 | 0.012091 | 0.440850 | ordinary |
| HistGradientBoostingClassifier | ordinary | `{"learning_rate": 0.04, "max_iter": 70, "max_leaf_nodes": 24}` / weight 0.6 | 0.449409 | 0.015719 | 0.430314 | baseline |
| GradientBoostingClassifier | advanced | `{"learning_rate": 0.075, "loss": "log_loss", "max_depth": 3, "max_features": 0.4, "min_samples_leaf": 45, "min_samples_split": 90, "n_estimators": 110, "subsample": 1.0}` / weight 0.6 | 0.443597 | 0.016140 | 0.425411 | advanced |

| Family | Baseline F1 | Ordinary F1 | Advanced F1 | Selected | F1 at 0.5 | Threshold | Optimized F1 | Precision | Recall | AP | ROC-AUC | TN | FP | FN | TP | Fit s | Predict s | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForestClassifier | 0.540459 | 0.543441 | 0.546150 | advanced | 0.546150 | 0.510997 | 0.543502 | 0.491539 | 0.607751 | 0.523255 | 0.851193 | 112,828 | 12,410 | 7,743 | 11,997 | 8.457462 | 0.113309 | `adc11d5eedb54dc9a3086cad7e6c4270` |
| XGBClassifier | 0.543524 | 0.544585 | 0.543889 | advanced | 0.543889 | 0.547957 | 0.528035 | 0.521436 | 0.534802 | 0.523368 | 0.848380 | 115,549 | 9,689 | 9,183 | 10,557 | 0.786290 | 0.027484 | `289945f86f2449ea8fe7ff2421249eaa` |
| GradientBoostingClassifier | 0.532694 | 0.525190 | 0.537374 | advanced | 0.537374 | 0.528222 | 0.526210 | 0.511904 | 0.541337 | 0.523780 | 0.847646 | 115,049 | 10,189 | 9,054 | 10,686 | 17.316130 | 0.082492 | `5e4b1bd4257f4e98920647519f59550c` |
| LGBMClassifier | 0.540652 | 0.545418 | 0.545418 | ordinary | 0.545418 | 0.580598 | 0.522158 | 0.528457 | 0.516008 | 0.521865 | 0.847562 | 116,149 | 9,089 | 9,554 | 10,186 | 0.803293 | 0.130568 | `8fd15510396444e29e0308f423fe97df` |
| CatBoostClassifier | 0.538475 | 0.546434 | 0.546434 | ordinary | 0.546434 | 0.576291 | 0.520806 | 0.533043 | 0.509119 | 0.524951 | 0.850076 | 116,434 | 8,804 | 9,690 | 10,050 | 3.006833 | 0.007761 | `652ffc4da1894319aa859515e14c2fce` |
| LogisticRegression | 0.477997 | 0.517770 | 0.517770 | ordinary | 0.517770 | 0.489344 | 0.520114 | 0.472318 | 0.578673 | 0.507008 | 0.840845 | 112,476 | 12,762 | 8,317 | 11,423 | 1.713572 | 0.001077 | `2a91d0ecab36428989a0efecc96c80e6` |
| HistGradientBoostingClassifier | 0.539990 | 0.537184 | 0.537184 | ordinary | 0.537184 | 0.545706 | 0.517620 | 0.526572 | 0.508967 | 0.515904 | 0.848068 | 116,205 | 9,033 | 9,693 | 10,047 | 1.890102 | 0.082498 | `bec883e5805e4614aaabe32dc0db1f15` |

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

## H. Verification and runtime

- The full-mode notebook executed the entire research workflow and logged a verified complete parent run in **2010.49 seconds** of parent-run wall time. Its presentation cell then failed because the confusion matrix was passed as a list. After converting the matrix to an array, fresh-kernel reuse completed **44** cells in **29.34 seconds** with zero cell errors and reproduced the recorded evidence. The completed training fits were reused for this presentation check.
- The full test suite reported **44 passed, zero failed**. The rebuilt tests flip outer labels and confirm identical configuration and threshold decisions, reject sealed dates, check train-only preprocessing, verify artifact round-trip, log and exclude failed folds, exercise fixed-configuration retraining, and reject mismatched cached evidence.
- Seven final model, preprocessor and cutoff combinations reloaded and reproduced the saved probabilities within rtol 1e-10 and atol 1e-12, with identical classification decisions and matching logged metrics. A synthetic test exposed a maximum difference of about 3.3e-16, prompting this tolerance. The interrupted first research parent `12a3805439204ddaa4d9b89c10cc2d4b` remains in MLflow and was excluded from selection.
- Historic snapshot checks confirmed all **426** run rows, metrics, parameters and tags were preserved, along with **1320** tracked artifact sizes and modification timestamps.
- Stage durations include local artifact logging. Per-model fit and prediction time describe the corresponding fitted model. Reused fits retain their original timing.

| Stage | Wall seconds |
| --- | --- |
| setup | 1.296100 |
| baseline | 94.763586 |
| ordinary | 505.412317 |
| advanced | 1399.579794 |
| threshold_and_logging | 2.214214 |
| artifact_verification | 1.218438 |

Reproduce the two notebook modes from the project root with the project virtual environment active.

```bash
source .venv/bin/activate
PBI06_MODE=full jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=14400 "ca1 hq file.ipynb"
PBI06_MODE=reuse PBI06_RUN_ID=939506bd3eb6434c8489d5871ea9778e jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1200 "ca1 hq file.ipynb"
python -m unittest discover -s tests -q
```

Installed research packages: {'numpy': '2.5.4', 'pandas': '3.1.1', 'scipy': '1.18.1', 'scikit-learn': '1.9.1', 'mlflow': '3.18.0', 'xgboost': '3.4.1', 'lightgbm': '4.7.0', 'catboost': '1.2.10', 'joblib': '1.6.0', 'threadpoolctl': '3.8.0'}.

## I. Remaining work

The eighth feature still needs data source confirmation and a trusted customer mapping available at scoring time. The prior reported XGBoost score used the same outer dates, but had a different search and representation. Current ranking relies on one late validation period and should be compared with the other team models in PBI 09. Ephraim's PBI 08 notebook is present only on its branch, and its protocol was not rerun during this PBI 06 work. Shared AWS MLflow, team ETL integration and the three-minute end-to-end runtime remain unverified. The reports and notebook provide local evidence.
