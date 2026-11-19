# PBI 06 direct MLflow logging

AWS experiment [PBI06_HanQian](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3) contains 28 individual runs. 26 inference packages were downloaded and reloaded successfully. 2 runs failed the artifact upload. The existing measured metrics and effective parameters matched the AWS records for all 28 runs.

The previous workflow copied a research tree with hundreds of fold fits and used a custom publisher to handle large files. PBI 07 and PBI 08 use one named run per model version and log metrics and fitted models through standard MLflow interfaces. The PBI 06 notebook now uses the same flat structure. It keeps the research history in local MLflow and uses the MLflow Python model flavor to package the saved estimator, fitted shared preprocessor and threshold as one inference interface. The model returns probabilities and fraud decisions. The evaluation uses existing validation probabilities and thresholds.

A separate client verified the Logistic Regression baseline before the other runs were logged. The first attempt encountered a path type incompatibility in the MLflow client. The next notebook execution reused the same pilot run ID. A query sorted on the standard `f1` metric returned the pilot and the final audit found one shared run per source version. A separate fresh-kernel repeat of the completed pilot reused its AWS run and left all 28 AWS run IDs unchanged. The MLflow client was 3.18.0 and the AWS server reported 3.17.2. Standard tracking, the run table query and model artifact retrieval worked against that pair.

A fresh kernel completed 46 code cells in 325.1 seconds without a cell error. The artifact errors were reported in notebook output and marked failed in MLflow. All 15 existing focused tracking tests passed. A focused routing check confirmed that unset and SQLite destinations are rejected before a direct logging write. The sealed features and labels were untouched. No models were retrained. PBI 07 and PBI 08 experiments remained present on the server.

Files updated were `ca1 hq file.ipynb`, `src/pbi06_mlflow.py`, `src/pbi06_research.py`, `src/pbi06_tracking.py`, `README.md` and this report. The notebook drops the old recovery and parent run lookup cells. The active research code no longer calls the old publisher's artifact functions. The historical publisher and evidence remain available for existing legacy callers.

## Run results

| Run | F1 | Precision | Recall | AP | ROC-AUC | Estimator bytes | Artifact result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [LogisticRegression_Baseline](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/3fc34dc9507e44ff9a9d3c762787c9d7) | 0.47800 | 0.34110 | 0.79843 | 0.50813 | 0.84173 | 682 | verified |
| [RandomForest_Baseline](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/14cf9900b98144778960817b224b01fa) | 0.54046 | 0.41313 | 0.78126 | 0.52652 | 0.84770 | 841081 | verified |
| [XGBoost_Baseline](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/39f9be296fca415f9e5a393333224680) | 0.54352 | 0.41881 | 0.77401 | 0.52360 | 0.84715 | 69685 | verified |
| [LightGBM_Baseline](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/56ae0b17f27c4591a6cd867de8662dc9) | 0.54065 | 0.41557 | 0.77345 | 0.52217 | 0.84764 | 122503 | verified |
| [CatBoost_Baseline](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/f0e30aab134f4d98bd69bb6c0f304488) | 0.53847 | 0.41080 | 0.78131 | 0.52468 | 0.84982 | 33017 | verified |
| [HistGradientBoosting_Baseline](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/8d7fb96ccf904d58a89dc8abeb36d70e) | 0.53999 | 0.41487 | 0.77315 | 0.51898 | 0.84736 | 61592 | verified |
| [GradientBoosting_Baseline](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/eafff5d939bf45bfa82a1cf36bd37604) | 0.53269 | 0.40390 | 0.78207 | 0.51808 | 0.84720 | 43655 | verified |
| [LogisticRegression_Tuned](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/5bf37fa57995460eb9d7f3cf39032d6b) | 0.51777 | 0.48107 | 0.56054 | 0.50701 | 0.84084 | 680 | verified |
| [RandomForest_Tuned](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/6ab8f1222a7a46969ec87ae42f701936) | 0.54344 | 0.48052 | 0.62533 | 0.50690 | 0.84985 | 842682 | verified |
| [XGBoost_Tuned](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/0a0fe500db1848d793fd6bf564412814) | 0.54459 | 0.48705 | 0.61753 | 0.52414 | 0.84673 | 283220 | verified |
| [LightGBM_Tuned](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/0357479efe8a4b628d4ca06b6e9531af) | 0.54542 | 0.48663 | 0.62036 | 0.52187 | 0.84756 | 231916 | verified |
| [CatBoost_Tuned](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/eea19edc60394d819cabbcfdd7a1b522) | 0.54643 | 0.48431 | 0.62685 | 0.52495 | 0.85008 | 148054 | verified |
| [HistGradientBoosting_Tuned](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/3493d6e4f25d4c5fbef050a8ce0abeaf) | 0.53718 | 0.49086 | 0.59316 | 0.51590 | 0.84807 | 82086 | verified |
| [GradientBoosting_Tuned](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/2de9e95ffe424e65be68c1ce99603de6) | 0.52519 | 0.51120 | 0.53997 | 0.51691 | 0.84910 | 27833 | verified |
| [LogisticRegression_Advanced](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/7a5bd1b54c8044e4981a426cc10e0f7f) | 0.51777 | 0.48107 | 0.56054 | 0.50701 | 0.84084 | 680 | verified |
| [RandomForest_Advanced](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/15d9b9545ee8499182a770fb9696b60e) | 0.54615 | 0.48652 | 0.62244 | 0.52326 | 0.85119 | 2599084 | FAILED |
| [XGBoost_Advanced](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/53bc4bd923cc404eaa9fbeee0fe4cf07) | 0.54389 | 0.49007 | 0.61099 | 0.52337 | 0.84838 | 92779 | verified |
| [LightGBM_Advanced](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/8aa5cccf978f4a36aa3d4038966f67e0) | 0.54542 | 0.48663 | 0.62036 | 0.52187 | 0.84756 | 231916 | verified |
| [CatBoost_Advanced](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/4674133a6f2942608e47c4985d49d8b8) | 0.54643 | 0.48431 | 0.62685 | 0.52495 | 0.85008 | 148054 | verified |
| [HistGradientBoosting_Advanced](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/341a64b0d97149ef90ed777e71630dcb) | 0.53718 | 0.49086 | 0.59316 | 0.51590 | 0.84807 | 82086 | verified |
| [GradientBoosting_Advanced](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/8d6cd79f2d3544d58704407175a0cea0) | 0.53737 | 0.48305 | 0.60547 | 0.52378 | 0.84765 | 58579 | verified |
| [RandomForest_ThresholdOptimized](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/37e38be72d8e4c6f99f0011815036ad6) | 0.54350 | 0.49154 | 0.60775 | 0.52326 | 0.85119 | 2730440 | FAILED |
| [XGBoost_ThresholdOptimized](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/90c5ffc9a9cb4d42ac1c8f179a76d9e4) | 0.52803 | 0.52144 | 0.53480 | 0.52337 | 0.84838 | 92779 | verified |
| [GradientBoosting_ThresholdOptimized](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/87b85f782f2244509293f36cfd9b5006) | 0.52621 | 0.51190 | 0.54134 | 0.52378 | 0.84765 | 58579 | verified |
| [LightGBM_ThresholdOptimized](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/4217d680aec4419297efc6cec698a71b) | 0.52216 | 0.52846 | 0.51601 | 0.52187 | 0.84756 | 231916 | verified |
| [CatBoost_ThresholdOptimized](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/3e384177cd3845bc854178b69cac23a6) | 0.52081 | 0.53304 | 0.50912 | 0.52495 | 0.85008 | 148054 | verified |
| [LogisticRegression_ThresholdOptimized](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/1123e25d9b2d433bbb0e8f3e4e67125d) | 0.52011 | 0.47232 | 0.57867 | 0.50701 | 0.84084 | 680 | verified |
| [HistGradientBoosting_ThresholdOptimized](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/3/runs/43972b696dd94bf2b6af873c82151c24) | 0.51762 | 0.52657 | 0.50897 | 0.51590 | 0.84807 | 82086 | verified |

## Artifact upload errors

- RandomForest_Advanced returned `HTTP 413 for the lossless XZ model (2599084 bytes): 413 Client Error: Request Entity Too Large for url: http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/api/2.0/mlflow-artifacts/artifacts/3/15d9b9545ee8499182a770fb9696b60e/artifacts/model/artifacts/estimator.joblib.`.
- RandomForest_ThresholdOptimized returned `HTTP 413 for the lossless XZ model (2730440 bytes): 413 Client Error: Request Entity Too Large for url: http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/api/2.0/mlflow-artifacts/artifacts/3/37e38be72d8e4c6f99f0011815036ad6/artifacts/model/artifacts/estimator.joblib.`.

For the advanced and threshold-optimized Random Forest versions, standard MLflow uploads of 4,692,636 and 4,844,057 bytes respectively returned HTTP 413. The notebook retried with lossless Joblib XZ compression. The compressed files produced matching probabilities within numerical tolerance and identical decisions on the validation probes. The proxy must accept request bodies larger than the measured rejected model payloads. A scoped nginx `client_max_body_size 5m` limit would accommodate the tested files if nginx is the limiting proxy. The AWS environment owner should confirm its effective configuration. See the [nginx directive](https://nginx.org/en/docs/http/ngx_http_core_module.html#client_max_body_size). This publication does not establish storage persistence across server replacement.

## Complete metrics and effective parameters

### LogisticRegression_Baseline

```json
{
  "metrics": {
    "accuracy": 0.7625570776255708,
    "average_precision": 0.5081287185656037,
    "f1": 0.4779971491826646,
    "false_positive_rate": 0.2430971430396525,
    "fn": 3979.0,
    "fp": 30445.0,
    "precision": 0.34110288707094316,
    "prediction_seconds": 0.0011078329989686608,
    "recall": 0.7984295845997974,
    "roc_auc": 0.8417338504003069,
    "tn": 94793.0,
    "tp": 15761.0,
    "training_seconds": 1.4458318750002945
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__C": "1.0",
    "model__class_weight": "None",
    "model__dual": "False",
    "model__fit_intercept": "True",
    "model__intercept_scaling": "1",
    "model__l1_ratio": "0.0",
    "model__max_iter": "200",
    "model__n_jobs": "None",
    "model__penalty": "deprecated",
    "model__random_state": "42",
    "model__solver": "lbfgs",
    "model__tol": "0.0001",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "1.0"
  },
  "source_run_id": "41c73e06a7c44321af16bd66ad2ac612"
}
```


### RandomForest_Baseline

```json
{
  "metrics": {
    "accuracy": 0.8191035881306129,
    "average_precision": 0.5265226562133862,
    "f1": 0.5404590853338006,
    "false_positive_rate": 0.17493093150641179,
    "fn": 4318.0,
    "fp": 21908.0,
    "precision": 0.413126171979641,
    "prediction_seconds": 0.03853050000179792,
    "recall": 0.7812563323201621,
    "roc_auc": 0.8476992100859619,
    "tn": 103330.0,
    "tp": 15422.0,
    "training_seconds": 2.2962752079984057
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__bootstrap": "True",
    "model__ccp_alpha": "0.0",
    "model__class_weight": "None",
    "model__criterion": "gini",
    "model__max_depth": "12",
    "model__max_features": "sqrt",
    "model__max_leaf_nodes": "None",
    "model__max_samples": "None",
    "model__min_impurity_decrease": "0.0",
    "model__min_samples_leaf": "30",
    "model__min_samples_split": "2",
    "model__min_weight_fraction_leaf": "0.0",
    "model__monotonic_cst": "None",
    "model__n_estimators": "20",
    "model__n_jobs": "4",
    "model__oob_score": "False",
    "model__random_state": "42",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "1.0"
  },
  "source_run_id": "11ad74bc96334e31b93f6b4527e3c544"
}
```


### XGBoost_Baseline

```json
{
  "metrics": {
    "accuracy": 0.8229800383506463,
    "average_precision": 0.523595640788289,
    "f1": 0.5435238874461954,
    "false_positive_rate": 0.16930164965904917,
    "fn": 4461.0,
    "fp": 21203.0,
    "precision": 0.4188092758072474,
    "prediction_seconds": 0.01700712499950896,
    "recall": 0.7740121580547112,
    "roc_auc": 0.8471534360684654,
    "tn": 104035.0,
    "tp": 15279.0,
    "training_seconds": 0.5742426249998971
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__base_score": "None",
    "model__booster": "None",
    "model__callbacks": "None",
    "model__colsample_bylevel": "None",
    "model__colsample_bynode": "None",
    "model__colsample_bytree": "None",
    "model__device": "None",
    "model__early_stopping_rounds": "None",
    "model__enable_categorical": "True",
    "model__eval_metric": "None",
    "model__feature_types": "None",
    "model__feature_weights": "None",
    "model__gamma": "None",
    "model__grow_policy": "None",
    "model__importance_type": "None",
    "model__interaction_constraints": "None",
    "model__learning_rate": "0.1",
    "model__max_bin": "None",
    "model__max_cat_threshold": "None",
    "model__max_cat_to_onehot": "None",
    "model__max_delta_step": "None",
    "model__max_depth": "5",
    "model__max_leaves": "None",
    "model__min_child_weight": "None",
    "model__missing": "nan",
    "model__monotone_constraints": "None",
    "model__multi_strategy": "None",
    "model__n_estimators": "80",
    "model__n_jobs": "4",
    "model__num_parallel_tree": "None",
    "model__objective": "binary:logistic",
    "model__random_state": "42",
    "model__reg_alpha": "None",
    "model__reg_lambda": "None",
    "model__sampling_method": "None",
    "model__scale_pos_weight": "None",
    "model__subsample": "None",
    "model__tree_method": "hist",
    "model__validate_parameters": "None",
    "model__verbosity": "None",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "1.0"
  },
  "source_run_id": "68247f279d9e4ec4b9b6f6c882f255f6"
}
```


### LightGBM_Baseline

```json
{
  "metrics": {
    "accuracy": 0.8210487108388859,
    "average_precision": 0.5221658596615354,
    "f1": 0.5406515580736544,
    "false_positive_rate": 0.17144956003768824,
    "fn": 4472.0,
    "fp": 21472.0,
    "precision": 0.4155688622754491,
    "prediction_seconds": 0.08048641600180417,
    "recall": 0.7734549138804458,
    "roc_auc": 0.8476420465848425,
    "tn": 103766.0,
    "tp": 15268.0,
    "training_seconds": 0.5579004169994732
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__boosting_type": "gbdt",
    "model__class_weight": "None",
    "model__colsample_bytree": "1.0",
    "model__importance_type": "split",
    "model__learning_rate": "0.1",
    "model__max_depth": "-1",
    "model__min_child_samples": "20",
    "model__min_child_weight": "0.001",
    "model__min_split_gain": "0.0",
    "model__n_estimators": "80",
    "model__n_jobs": "4",
    "model__num_leaves": "31",
    "model__objective": "None",
    "model__random_state": "42",
    "model__reg_alpha": "0.0",
    "model__reg_lambda": "0.0",
    "model__subsample": "1.0",
    "model__subsample_for_bin": "200000",
    "model__subsample_freq": "0",
    "model__verbosity": "-1",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "1.0"
  },
  "source_run_id": "6a6564a5d525432f8a39ec5dc6f72f0a"
}
```


### CatBoost_Baseline

```json
{
  "metrics": {
    "accuracy": 0.81764129730028,
    "average_precision": 0.5246811102365786,
    "f1": 0.5384749668319252,
    "false_positive_rate": 0.1766316932560405,
    "fn": 4317.0,
    "fp": 22121.0,
    "precision": 0.41079799701683356,
    "prediction_seconds": 0.005273624999972526,
    "recall": 0.7813069908814589,
    "roc_auc": 0.8498155257071387,
    "tn": 103117.0,
    "tp": 15423.0,
    "training_seconds": 1.438070082997001
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__allow_writing_files": "False",
    "model__depth": "5",
    "model__iterations": "80",
    "model__learning_rate": "0.1",
    "model__random_seed": "42",
    "model__thread_count": "4",
    "model__verbose": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "1.0"
  },
  "source_run_id": "1003dca400344927aca47c9e89f61bd2"
}
```


### HistGradientBoosting_Baseline

```json
{
  "metrics": {
    "accuracy": 0.820641752541765,
    "average_precision": 0.518976963562796,
    "f1": 0.539989739416562,
    "false_positive_rate": 0.17187275427585877,
    "fn": 4478.0,
    "fp": 21525.0,
    "precision": 0.41487481990920705,
    "prediction_seconds": 0.06951187500089873,
    "recall": 0.7731509625126647,
    "roc_auc": 0.8473597997882144,
    "tn": 103713.0,
    "tp": 15262.0,
    "training_seconds": 1.651476625000214
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__categorical_features": "from_dtype",
    "model__class_weight": "None",
    "model__early_stopping": "False",
    "model__interaction_cst": "None",
    "model__l2_regularization": "0.0",
    "model__learning_rate": "0.1",
    "model__loss": "log_loss",
    "model__max_bins": "255",
    "model__max_depth": "None",
    "model__max_features": "1.0",
    "model__max_iter": "80",
    "model__max_leaf_nodes": "15",
    "model__min_samples_leaf": "20",
    "model__monotonic_cst": "None",
    "model__n_iter_no_change": "10",
    "model__random_state": "42",
    "model__scoring": "loss",
    "model__tol": "1e-07",
    "model__validation_fraction": "0.1",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "1.0"
  },
  "source_run_id": "3a4da8756d6f4a6f9fb542949da3b9c7"
}
```


### GradientBoosting_Baseline

```json
{
  "metrics": {
    "accuracy": 0.8131716536302059,
    "average_precision": 0.5180834737574755,
    "f1": 0.5326938338911701,
    "false_positive_rate": 0.18192561363164536,
    "fn": 4302.0,
    "fp": 22784.0,
    "precision": 0.4039035110669248,
    "prediction_seconds": 0.06177387499701581,
    "recall": 0.7820668693009118,
    "roc_auc": 0.8471977806131492,
    "tn": 102454.0,
    "tp": 15438.0,
    "training_seconds": 32.99796183299986
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__ccp_alpha": "0.0",
    "model__criterion": "deprecated",
    "model__init": "None",
    "model__learning_rate": "0.1",
    "model__loss": "log_loss",
    "model__max_depth": "3",
    "model__max_features": "None",
    "model__max_leaf_nodes": "None",
    "model__min_impurity_decrease": "0.0",
    "model__min_samples_leaf": "1",
    "model__min_samples_split": "2",
    "model__min_weight_fraction_leaf": "0.0",
    "model__n_estimators": "80",
    "model__n_iter_no_change": "None",
    "model__random_state": "42",
    "model__subsample": "1.0",
    "model__tol": "0.0001",
    "model__validation_fraction": "0.1",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "1.0"
  },
  "source_run_id": "840ba683b76844bea8808dddbc714608"
}
```


### LogisticRegression_Tuned

```json
{
  "metrics": {
    "accuracy": 0.8578336023396653,
    "average_precision": 0.5070081357777765,
    "f1": 0.5177698228866896,
    "false_positive_rate": 0.0953065363547805,
    "fn": 8675.0,
    "fp": 11936.0,
    "precision": 0.48106604060693015,
    "prediction_seconds": 0.0010766669984150212,
    "recall": 0.5605369807497467,
    "roc_auc": 0.8408446360682452,
    "tn": 113302.0,
    "tp": 11065.0,
    "training_seconds": 1.7135720000005676
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__C": "20",
    "model__class_weight": "None",
    "model__dual": "False",
    "model__fit_intercept": "True",
    "model__intercept_scaling": "1",
    "model__l1_ratio": "0.0",
    "model__max_iter": "200",
    "model__n_jobs": "None",
    "model__penalty": "deprecated",
    "model__random_state": "42",
    "model__solver": "lbfgs",
    "model__tol": "0.0001",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "95ba837d12ad4f868762a44500e65c88"
}
```


### RandomForest_Tuned

```json
{
  "metrics": {
    "accuracy": 0.856936914566348,
    "average_precision": 0.5068985051992819,
    "f1": 0.54344141407471,
    "false_positive_rate": 0.10655711525255913,
    "fn": 7396.0,
    "fp": 13345.0,
    "precision": 0.4805169527813461,
    "prediction_seconds": 0.038458375001937384,
    "recall": 0.6253292806484296,
    "roc_auc": 0.8498544776823954,
    "tn": 111893.0,
    "tp": 12344.0,
    "training_seconds": 2.956274083000608
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__bootstrap": "True",
    "model__ccp_alpha": "0.0",
    "model__class_weight": "None",
    "model__criterion": "gini",
    "model__max_depth": "12",
    "model__max_features": "sqrt",
    "model__max_leaf_nodes": "None",
    "model__max_samples": "None",
    "model__min_impurity_decrease": "0.0",
    "model__min_samples_leaf": "65",
    "model__min_samples_split": "2",
    "model__min_weight_fraction_leaf": "0.0",
    "model__monotonic_cst": "None",
    "model__n_estimators": "25",
    "model__n_jobs": "4",
    "model__oob_score": "False",
    "model__random_state": "42",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "d483cbecfa624b8f9cd31c6bf5fa1625"
}
```


### XGBoost_Tuned

```json
{
  "metrics": {
    "accuracy": 0.8593717667508174,
    "average_precision": 0.524144296194814,
    "f1": 0.5445854181558256,
    "false_positive_rate": 0.10250882320062601,
    "fn": 7550.0,
    "fp": 12838.0,
    "precision": 0.4870544989611635,
    "prediction_seconds": 0.03158233399881283,
    "recall": 0.6175278622087133,
    "roc_auc": 0.8467340600922388,
    "tn": 112400.0,
    "tp": 12190.0,
    "training_seconds": 0.8433469169976888
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__base_score": "None",
    "model__booster": "None",
    "model__callbacks": "None",
    "model__colsample_bylevel": "None",
    "model__colsample_bynode": "None",
    "model__colsample_bytree": "None",
    "model__device": "None",
    "model__early_stopping_rounds": "None",
    "model__enable_categorical": "True",
    "model__eval_metric": "None",
    "model__feature_types": "None",
    "model__feature_weights": "None",
    "model__gamma": "None",
    "model__grow_policy": "None",
    "model__importance_type": "None",
    "model__interaction_constraints": "None",
    "model__learning_rate": "0.08",
    "model__max_bin": "None",
    "model__max_cat_threshold": "None",
    "model__max_cat_to_onehot": "None",
    "model__max_delta_step": "None",
    "model__max_depth": "7",
    "model__max_leaves": "None",
    "model__min_child_weight": "None",
    "model__missing": "nan",
    "model__monotone_constraints": "None",
    "model__multi_strategy": "None",
    "model__n_estimators": "100",
    "model__n_jobs": "4",
    "model__num_parallel_tree": "None",
    "model__objective": "binary:logistic",
    "model__random_state": "42",
    "model__reg_alpha": "None",
    "model__reg_lambda": "None",
    "model__sampling_method": "None",
    "model__scale_pos_weight": "None",
    "model__subsample": "None",
    "model__tree_method": "hist",
    "model__validate_parameters": "None",
    "model__verbosity": "None",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "dd07c62c0e0e43a18d42fc379cb2d85e"
}
```


### LightGBM_Tuned

```json
{
  "metrics": {
    "accuracy": 0.8591993267944101,
    "average_precision": 0.5218651194127812,
    "f1": 0.545418104888097,
    "false_positive_rate": 0.10315559175330172,
    "fn": 7494.0,
    "fp": 12919.0,
    "precision": 0.4866282535267236,
    "prediction_seconds": 0.13056816699827323,
    "recall": 0.6203647416413374,
    "roc_auc": 0.8475623668866799,
    "tn": 112319.0,
    "tp": 12246.0,
    "training_seconds": 0.8032929170003626
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__boosting_type": "gbdt",
    "model__class_weight": "None",
    "model__colsample_bytree": "1.0",
    "model__importance_type": "split",
    "model__learning_rate": "0.07",
    "model__max_depth": "-1",
    "model__min_child_samples": "20",
    "model__min_child_weight": "0.001",
    "model__min_split_gain": "0.0",
    "model__n_estimators": "120",
    "model__n_jobs": "4",
    "model__num_leaves": "40",
    "model__objective": "None",
    "model__random_state": "42",
    "model__reg_alpha": "0.0",
    "model__reg_lambda": "0.0",
    "model__subsample": "1.0",
    "model__subsample_for_bin": "200000",
    "model__subsample_freq": "0",
    "model__verbosity": "-1",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "2a1056b2c0bc4759b5c8213d7a0bd443"
}
```


### CatBoost_Tuned

```json
{
  "metrics": {
    "accuracy": 0.8583095366193492,
    "average_precision": 0.524951047594462,
    "f1": 0.5464340914109075,
    "false_positive_rate": 0.10520768456858143,
    "fn": 7366.0,
    "fp": 13176.0,
    "precision": 0.48430528375733856,
    "prediction_seconds": 0.007761374999972759,
    "recall": 0.6268490374873353,
    "roc_auc": 0.8500756895244301,
    "tn": 112062.0,
    "tp": 12374.0,
    "training_seconds": 3.006833125000412
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__allow_writing_files": "False",
    "model__depth": "7",
    "model__iterations": "155",
    "model__learning_rate": "0.08",
    "model__random_seed": "42",
    "model__thread_count": "4",
    "model__verbose": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "8c3b22e3730746a891f589c26940091c"
}
```


### HistGradientBoosting_Tuned

```json
{
  "metrics": {
    "accuracy": 0.8608340575811503,
    "average_precision": 0.5159040614382422,
    "f1": 0.5371840161490113,
    "false_positive_rate": 0.09697535891662275,
    "fn": 8031.0,
    "fp": 12145.0,
    "precision": 0.49086107151840364,
    "prediction_seconds": 0.08249799999975949,
    "recall": 0.593161094224924,
    "roc_auc": 0.8480677849152317,
    "tn": 113093.0,
    "tp": 11709.0,
    "training_seconds": 1.8901022089994512
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__categorical_features": "from_dtype",
    "model__class_weight": "None",
    "model__early_stopping": "False",
    "model__interaction_cst": "None",
    "model__l2_regularization": "0.0",
    "model__learning_rate": "0.04",
    "model__loss": "log_loss",
    "model__max_bins": "255",
    "model__max_depth": "None",
    "model__max_features": "1.0",
    "model__max_iter": "70",
    "model__max_leaf_nodes": "24",
    "model__min_samples_leaf": "20",
    "model__monotonic_cst": "None",
    "model__n_iter_no_change": "10",
    "model__random_state": "42",
    "model__scoring": "loss",
    "model__tol": "1e-07",
    "model__validation_fraction": "0.1",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "459a36a85b36415d8dd81c2fe5f898e7"
}
```


### GradientBoosting_Tuned

```json
{
  "metrics": {
    "accuracy": 0.8670625888065775,
    "average_precision": 0.516910545450261,
    "f1": 0.5251903131235989,
    "false_positive_rate": 0.0813810504798863,
    "fn": 9081.0,
    "fp": 10192.0,
    "precision": 0.5111985036688887,
    "prediction_seconds": 0.03841508299956331,
    "recall": 0.5399696048632219,
    "roc_auc": 0.8490994184559932,
    "tn": 115046.0,
    "tp": 10659.0,
    "training_seconds": 16.618223334000504
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__ccp_alpha": "0.0",
    "model__criterion": "deprecated",
    "model__init": "None",
    "model__learning_rate": "0.09",
    "model__loss": "log_loss",
    "model__max_depth": "3",
    "model__max_features": "None",
    "model__max_leaf_nodes": "None",
    "model__min_impurity_decrease": "0.0",
    "model__min_samples_leaf": "1",
    "model__min_samples_split": "2",
    "model__min_weight_fraction_leaf": "0.0",
    "model__n_estimators": "45",
    "model__n_iter_no_change": "None",
    "model__random_state": "42",
    "model__subsample": "1.0",
    "model__tol": "0.0001",
    "model__validation_fraction": "0.1",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "c90bd18e4e4447b8b6a4fd51f3f80664"
}
```


### LogisticRegression_Advanced

```json
{
  "metrics": {
    "accuracy": 0.8578336023396653,
    "average_precision": 0.5070081357777765,
    "f1": 0.5177698228866896,
    "false_positive_rate": 0.0953065363547805,
    "fn": 8675.0,
    "fp": 11936.0,
    "precision": 0.48106604060693015,
    "prediction_seconds": 0.0010766669984150212,
    "recall": 0.5605369807497467,
    "roc_auc": 0.8408446360682452,
    "tn": 113302.0,
    "tp": 11065.0,
    "training_seconds": 1.7135720000005676
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__C": "20",
    "model__class_weight": "None",
    "model__dual": "False",
    "model__fit_intercept": "True",
    "model__intercept_scaling": "1",
    "model__l1_ratio": "0.0",
    "model__max_iter": "200",
    "model__n_jobs": "None",
    "model__penalty": "deprecated",
    "model__random_state": "42",
    "model__solver": "lbfgs",
    "model__tol": "0.0001",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "63db305ddd604befac13c12759a1a010"
}
```


### RandomForest_Advanced

```json
{
  "metrics": {
    "accuracy": 0.8591441460083599,
    "average_precision": 0.5232550276784987,
    "f1": 0.5461495721746861,
    "false_positive_rate": 0.10354684680368578,
    "fn": 7453.0,
    "fp": 12968.0,
    "precision": 0.48651752128291426,
    "prediction_seconds": 0.11330883300252026,
    "recall": 0.6224417426545086,
    "roc_auc": 0.8511929790238656,
    "tn": 112270.0,
    "tp": 12287.0,
    "training_seconds": 8.457462291000411
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__bootstrap": "True",
    "model__ccp_alpha": "0.0",
    "model__class_weight": "None",
    "model__criterion": "entropy",
    "model__max_depth": "19",
    "model__max_features": "sqrt",
    "model__max_leaf_nodes": "None",
    "model__max_samples": "0.55",
    "model__min_impurity_decrease": "0.0",
    "model__min_samples_leaf": "15",
    "model__min_samples_split": "160",
    "model__min_weight_fraction_leaf": "0.0",
    "model__monotonic_cst": "None",
    "model__n_estimators": "55",
    "model__n_jobs": "4",
    "model__oob_score": "False",
    "model__random_state": "42",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "231ad02c1df54fe38c2463efe8416cba"
}
```


### XGBoost_Advanced

```json
{
  "metrics": {
    "accuracy": 0.860468484873567,
    "average_precision": 0.5233683010075612,
    "f1": 0.5438885256251268,
    "false_positive_rate": 0.10020920168000128,
    "fn": 7679.0,
    "fp": 12550.0,
    "precision": 0.4900654179025639,
    "prediction_seconds": 0.02748362500278745,
    "recall": 0.6109929078014185,
    "roc_auc": 0.8483802665459513,
    "tn": 112688.0,
    "tp": 12061.0,
    "training_seconds": 0.7862901670014253
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__base_score": "None",
    "model__booster": "None",
    "model__callbacks": "None",
    "model__colsample_bylevel": "None",
    "model__colsample_bynode": "None",
    "model__colsample_bytree": "0.85",
    "model__device": "None",
    "model__early_stopping_rounds": "None",
    "model__enable_categorical": "True",
    "model__eval_metric": "None",
    "model__feature_types": "None",
    "model__feature_weights": "None",
    "model__gamma": "1",
    "model__grow_policy": "None",
    "model__importance_type": "None",
    "model__interaction_constraints": "None",
    "model__learning_rate": "0.07",
    "model__max_bin": "256",
    "model__max_cat_threshold": "None",
    "model__max_cat_to_onehot": "None",
    "model__max_delta_step": "None",
    "model__max_depth": "5",
    "model__max_leaves": "None",
    "model__min_child_weight": "1",
    "model__missing": "nan",
    "model__monotone_constraints": "None",
    "model__multi_strategy": "None",
    "model__n_estimators": "110",
    "model__n_jobs": "4",
    "model__num_parallel_tree": "None",
    "model__objective": "binary:logistic",
    "model__random_state": "42",
    "model__reg_alpha": "3",
    "model__reg_lambda": "4",
    "model__sampling_method": "None",
    "model__scale_pos_weight": "None",
    "model__subsample": "0.65",
    "model__tree_method": "hist",
    "model__validate_parameters": "None",
    "model__verbosity": "None",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "01a3a6d07d3246d09f809867cab8fe8c"
}
```


### LightGBM_Advanced

```json
{
  "metrics": {
    "accuracy": 0.8591993267944101,
    "average_precision": 0.5218651194127812,
    "f1": 0.545418104888097,
    "false_positive_rate": 0.10315559175330172,
    "fn": 7494.0,
    "fp": 12919.0,
    "precision": 0.4866282535267236,
    "prediction_seconds": 0.13056816699827323,
    "recall": 0.6203647416413374,
    "roc_auc": 0.8475623668866799,
    "tn": 112319.0,
    "tp": 12246.0,
    "training_seconds": 0.8032929170003626
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__boosting_type": "gbdt",
    "model__class_weight": "None",
    "model__colsample_bytree": "1.0",
    "model__importance_type": "split",
    "model__learning_rate": "0.07",
    "model__max_depth": "-1",
    "model__min_child_samples": "20",
    "model__min_child_weight": "0.001",
    "model__min_split_gain": "0.0",
    "model__n_estimators": "120",
    "model__n_jobs": "4",
    "model__num_leaves": "40",
    "model__objective": "None",
    "model__random_state": "42",
    "model__reg_alpha": "0.0",
    "model__reg_lambda": "0.0",
    "model__subsample": "1.0",
    "model__subsample_for_bin": "200000",
    "model__subsample_freq": "0",
    "model__verbosity": "-1",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "3831ea7f26f642c0aa31434d956e8663"
}
```


### CatBoost_Advanced

```json
{
  "metrics": {
    "accuracy": 0.8583095366193492,
    "average_precision": 0.524951047594462,
    "f1": 0.5464340914109075,
    "false_positive_rate": 0.10520768456858143,
    "fn": 7366.0,
    "fp": 13176.0,
    "precision": 0.48430528375733856,
    "prediction_seconds": 0.007761374999972759,
    "recall": 0.6268490374873353,
    "roc_auc": 0.8500756895244301,
    "tn": 112062.0,
    "tp": 12374.0,
    "training_seconds": 3.006833125000412
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__allow_writing_files": "False",
    "model__depth": "7",
    "model__iterations": "155",
    "model__learning_rate": "0.08",
    "model__random_seed": "42",
    "model__thread_count": "4",
    "model__verbose": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "33e0bce9dccf44ad8bfe44aa50c958b6"
}
```


### HistGradientBoosting_Advanced

```json
{
  "metrics": {
    "accuracy": 0.8608340575811503,
    "average_precision": 0.5159040614382422,
    "f1": 0.5371840161490113,
    "false_positive_rate": 0.09697535891662275,
    "fn": 8031.0,
    "fp": 12145.0,
    "precision": 0.49086107151840364,
    "prediction_seconds": 0.08249799999975949,
    "recall": 0.593161094224924,
    "roc_auc": 0.8480677849152317,
    "tn": 113093.0,
    "tp": 11709.0,
    "training_seconds": 1.8901022089994512
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__categorical_features": "from_dtype",
    "model__class_weight": "None",
    "model__early_stopping": "False",
    "model__interaction_cst": "None",
    "model__l2_regularization": "0.0",
    "model__learning_rate": "0.04",
    "model__loss": "log_loss",
    "model__max_bins": "255",
    "model__max_depth": "None",
    "model__max_features": "1.0",
    "model__max_iter": "70",
    "model__max_leaf_nodes": "24",
    "model__min_samples_leaf": "20",
    "model__monotonic_cst": "None",
    "model__n_iter_no_change": "10",
    "model__random_state": "42",
    "model__scoring": "loss",
    "model__tol": "1e-07",
    "model__validation_fraction": "0.1",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "dd0d4c6d1e7b41f3b695396819def5af"
}
```


### GradientBoosting_Advanced

```json
{
  "metrics": {
    "accuracy": 0.8580543254838665,
    "average_precision": 0.5237804192791258,
    "f1": 0.537373828204033,
    "false_positive_rate": 0.10213353774413517,
    "fn": 7788.0,
    "fp": 12791.0,
    "precision": 0.48304570989774887,
    "prediction_seconds": 0.08249233300011838,
    "recall": 0.6054711246200608,
    "roc_auc": 0.8476456990024732,
    "tn": 112447.0,
    "tp": 11952.0,
    "training_seconds": 17.31613008400018
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__ccp_alpha": "0.0",
    "model__criterion": "deprecated",
    "model__init": "None",
    "model__learning_rate": "0.075",
    "model__loss": "log_loss",
    "model__max_depth": "3",
    "model__max_features": "0.4",
    "model__max_leaf_nodes": "None",
    "model__min_impurity_decrease": "0.0",
    "model__min_samples_leaf": "45",
    "model__min_samples_split": "90",
    "model__min_weight_fraction_leaf": "0.0",
    "model__n_estimators": "110",
    "model__n_iter_no_change": "None",
    "model__random_state": "42",
    "model__subsample": "1.0",
    "model__tol": "0.0001",
    "model__validation_fraction": "0.1",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "a4598a508f7e4040962f97548e5d7c0b"
}
```


### RandomForest_ThresholdOptimized

```json
{
  "metrics": {
    "accuracy": 0.8609927023410449,
    "average_precision": 0.5232550276784987,
    "f1": 0.5435023897433574,
    "false_positive_rate": 0.09909133010747537,
    "fn": 7743.0,
    "fp": 12410.0,
    "precision": 0.4915393124923178,
    "prediction_seconds": 0.11330883300252026,
    "recall": 0.6077507598784194,
    "roc_auc": 0.8511929790238656,
    "tn": 112828.0,
    "tp": 11997.0,
    "training_seconds": 8.457462291000411
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__bootstrap": "True",
    "model__ccp_alpha": "0.0",
    "model__class_weight": "None",
    "model__criterion": "entropy",
    "model__max_depth": "19",
    "model__max_features": "sqrt",
    "model__max_leaf_nodes": "None",
    "model__max_samples": "0.55",
    "model__min_impurity_decrease": "0.0",
    "model__min_samples_leaf": "15",
    "model__min_samples_split": "160",
    "model__min_weight_fraction_leaf": "0.0",
    "model__monotonic_cst": "None",
    "model__n_estimators": "55",
    "model__n_jobs": "4",
    "model__oob_score": "False",
    "model__random_state": "42",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5109967277392646",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "adc11d5eedb54dc9a3086cad7e6c4270"
}
```


### XGBoost_ThresholdOptimized

```json
{
  "metrics": {
    "accuracy": 0.8698285257073487,
    "average_precision": 0.5233683010075612,
    "f1": 0.5280348121842645,
    "false_positive_rate": 0.07736469761573964,
    "fn": 9183.0,
    "fp": 9689.0,
    "precision": 0.5214363331028351,
    "prediction_seconds": 0.02748362500278745,
    "recall": 0.5348024316109422,
    "roc_auc": 0.8483802665459513,
    "tn": 115549.0,
    "tp": 10557.0,
    "training_seconds": 0.7862901670014253
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__base_score": "None",
    "model__booster": "None",
    "model__callbacks": "None",
    "model__colsample_bylevel": "None",
    "model__colsample_bynode": "None",
    "model__colsample_bytree": "0.85",
    "model__device": "None",
    "model__early_stopping_rounds": "None",
    "model__enable_categorical": "True",
    "model__eval_metric": "None",
    "model__feature_types": "None",
    "model__feature_weights": "None",
    "model__gamma": "1",
    "model__grow_policy": "None",
    "model__importance_type": "None",
    "model__interaction_constraints": "None",
    "model__learning_rate": "0.07",
    "model__max_bin": "256",
    "model__max_cat_threshold": "None",
    "model__max_cat_to_onehot": "None",
    "model__max_delta_step": "None",
    "model__max_depth": "5",
    "model__max_leaves": "None",
    "model__min_child_weight": "1",
    "model__missing": "nan",
    "model__monotone_constraints": "None",
    "model__multi_strategy": "None",
    "model__n_estimators": "110",
    "model__n_jobs": "4",
    "model__num_parallel_tree": "None",
    "model__objective": "binary:logistic",
    "model__random_state": "42",
    "model__reg_alpha": "3",
    "model__reg_lambda": "4",
    "model__sampling_method": "None",
    "model__scale_pos_weight": "None",
    "model__subsample": "0.65",
    "model__tree_method": "hist",
    "model__validate_parameters": "None",
    "model__verbosity": "None",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5479568839073181",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "289945f86f2449ea8fe7ff2421249eaa"
}
```


### GradientBoosting_ThresholdOptimized

```json
{
  "metrics": {
    "accuracy": 0.8672695167542661,
    "average_precision": 0.5237804192791258,
    "f1": 0.5262095284993229,
    "false_positive_rate": 0.08135709608904645,
    "fn": 9054.0,
    "fp": 10189.0,
    "precision": 0.5119041916167665,
    "prediction_seconds": 0.08249233300011838,
    "recall": 0.5413373860182371,
    "roc_auc": 0.8476456990024732,
    "tn": 115049.0,
    "tp": 10686.0,
    "training_seconds": 17.31613008400018
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__ccp_alpha": "0.0",
    "model__criterion": "deprecated",
    "model__init": "None",
    "model__learning_rate": "0.075",
    "model__loss": "log_loss",
    "model__max_depth": "3",
    "model__max_features": "0.4",
    "model__max_leaf_nodes": "None",
    "model__min_impurity_decrease": "0.0",
    "model__min_samples_leaf": "45",
    "model__min_samples_split": "90",
    "model__min_weight_fraction_leaf": "0.0",
    "model__n_estimators": "110",
    "model__n_iter_no_change": "None",
    "model__random_state": "42",
    "model__subsample": "1.0",
    "model__tol": "0.0001",
    "model__validation_fraction": "0.1",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.528222223685986",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "5e4b1bd4257f4e98920647519f59550c"
}
```


### LightGBM_ThresholdOptimized

```json
{
  "metrics": {
    "accuracy": 0.8714080757080385,
    "average_precision": 0.5218651194127812,
    "f1": 0.5221581443034731,
    "false_positive_rate": 0.07257381944777144,
    "fn": 9554.0,
    "fp": 9089.0,
    "precision": 0.5284565499351491,
    "prediction_seconds": 0.13056816699827323,
    "recall": 0.5160081053698075,
    "roc_auc": 0.8475623668866799,
    "tn": 116149.0,
    "tp": 10186.0,
    "training_seconds": 0.8032929170003626
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__boosting_type": "gbdt",
    "model__class_weight": "None",
    "model__colsample_bytree": "1.0",
    "model__importance_type": "split",
    "model__learning_rate": "0.07",
    "model__max_depth": "-1",
    "model__min_child_samples": "20",
    "model__min_child_weight": "0.001",
    "model__min_split_gain": "0.0",
    "model__n_estimators": "120",
    "model__n_jobs": "4",
    "model__num_leaves": "40",
    "model__objective": "None",
    "model__random_state": "42",
    "model__reg_alpha": "0.0",
    "model__reg_lambda": "0.0",
    "model__subsample": "1.0",
    "model__subsample_for_bin": "200000",
    "model__subsample_freq": "0",
    "model__verbosity": "-1",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5805976492906365",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "8fd15510396444e29e0308f423fe97df"
}
```


### CatBoost_ThresholdOptimized

```json
{
  "metrics": {
    "accuracy": 0.8724358178482252,
    "average_precision": 0.524951047594462,
    "f1": 0.5208063429548635,
    "false_positive_rate": 0.07029815231798656,
    "fn": 9690.0,
    "fp": 8804.0,
    "precision": 0.5330433860188819,
    "prediction_seconds": 0.007761374999972759,
    "recall": 0.5091185410334347,
    "roc_auc": 0.8500756895244301,
    "tn": 116434.0,
    "tp": 10050.0,
    "training_seconds": 3.006833125000412
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__allow_writing_files": "False",
    "model__depth": "7",
    "model__iterations": "155",
    "model__learning_rate": "0.08",
    "model__random_seed": "42",
    "model__thread_count": "4",
    "model__verbose": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5762911561144178",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "652ffc4da1894319aa859515e14c2fce"
}
```


### LogisticRegression_ThresholdOptimized

```json
{
  "metrics": {
    "accuracy": 0.854605526355723,
    "average_precision": 0.5070081357777765,
    "f1": 0.5201138303927149,
    "false_positive_rate": 0.10190197863268337,
    "fn": 8317.0,
    "fp": 12762.0,
    "precision": 0.47231755220177796,
    "prediction_seconds": 0.0010766669984150212,
    "recall": 0.5786727456940223,
    "roc_auc": 0.8408446360682452,
    "tn": 112476.0,
    "tp": 11423.0,
    "training_seconds": 1.7135720000005676
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__C": "20",
    "model__class_weight": "None",
    "model__dual": "False",
    "model__fit_intercept": "True",
    "model__intercept_scaling": "1",
    "model__l1_ratio": "0.0",
    "model__max_iter": "200",
    "model__n_jobs": "None",
    "model__penalty": "deprecated",
    "model__random_state": "42",
    "model__solver": "lbfgs",
    "model__tol": "0.0001",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.4893435537815094",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "2a91d0ecab36428989a0efecc96c80e6"
}
```


### HistGradientBoosting_ThresholdOptimized

```json
{
  "metrics": {
    "accuracy": 0.8708355750527667,
    "average_precision": 0.5159040614382422,
    "f1": 0.5176197836166925,
    "false_positive_rate": 0.07212667081876108,
    "fn": 9693.0,
    "fp": 9033.0,
    "precision": 0.5265723270440251,
    "prediction_seconds": 0.08249799999975949,
    "recall": 0.5089665653495441,
    "roc_auc": 0.8480677849152317,
    "tn": 116205.0,
    "tp": 10047.0,
    "training_seconds": 1.8901022089994512
  },
  "parameters": {
    "calibration_first": "2023-02-18",
    "calibration_last": "2023-03-14",
    "calibration_rows": "135547",
    "fit_first": "2022-11-12",
    "fit_last": "2023-02-17",
    "fit_rows": "525226",
    "model__categorical_features": "from_dtype",
    "model__class_weight": "None",
    "model__early_stopping": "False",
    "model__interaction_cst": "None",
    "model__l2_regularization": "0.0",
    "model__learning_rate": "0.04",
    "model__loss": "log_loss",
    "model__max_bins": "255",
    "model__max_depth": "None",
    "model__max_features": "1.0",
    "model__max_iter": "70",
    "model__max_leaf_nodes": "24",
    "model__min_samples_leaf": "20",
    "model__monotonic_cst": "None",
    "model__n_iter_no_change": "10",
    "model__random_state": "42",
    "model__scoring": "loss",
    "model__tol": "1e-07",
    "model__validation_fraction": "0.1",
    "model__verbose": "0",
    "model__warm_start": "False",
    "preprocessing": "shared PBI 03, early outer training fit",
    "schema": "order_value,num_items_ordered,order_weekday,order_month,country_code,collect_type,payment_method,days_since_first_order",
    "seed": "42",
    "threshold": "0.5457061396298276",
    "validation_first": "2023-03-15",
    "validation_last": "2023-04-13",
    "validation_rows": "144978",
    "weight_strength": "0.6"
  },
  "source_run_id": "bec883e5805e4614aaabe32dc0db1f15"
}
```
