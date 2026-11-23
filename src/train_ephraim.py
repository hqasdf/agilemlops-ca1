# Fast retraining routine for PBI 10, written by KochuveettilTojiEphraimModels.ipynb.
# Tuned settings below were found in that notebook. Rerun the notebook cell to rewrite this file.
import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.feature_engineering import INPUT_FEATURES, fit_preprocessor

HISTORY = "days_since_first_order"
THRESHOLD = 0.50
SEED = 42
DT_PARAMS = {'max_depth': 10, 'min_samples_leaf': 20, 'class_weight': 'balanced'}
RF_PARAMS = {'max_depth': 14, 'min_samples_leaf': 20, 'class_weight': 'balanced'}
N_TREES = 30


def build_preprocessor(rows_fit):
    shared = fit_preprocessor(rows_fit[INPUT_FEATURES])
    history_step = ColumnTransformer([
        ("customer_history", SimpleImputer(strategy="median", keep_empty_features=True), [HISTORY]),
    ]).fit(rows_fit)
    return FeatureUnion([("shared", shared), ("history", history_step)])


def evaluate(y_true, scores):
    predicted = (np.asarray(scores) >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "flag_rate": float(predicted.mean()),
    }


def train_ephraim_models(x_train, y_train, x_val, y_val, tracking_uri=None, batch_name="base"):
    # returns one dict per model: name, pipeline, metrics, fit_seconds, run_id
    if set(y_train) != {0, 1} or set(y_val) != {0, 1}:
        raise ValueError("train and validation both need fraud and non fraud rows")
    preprocessor = build_preprocessor(x_train)
    train_matrix = preprocessor.transform(x_train)
    val_matrix = preprocessor.transform(x_val)
    models = {
        "dt_tuned": DecisionTreeClassifier(random_state=SEED, **DT_PARAMS),
        "rf_tuned": RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=-1, **RF_PARAMS),
    }
    out = []
    for name, estimator in models.items():
        start = time.perf_counter()
        estimator.fit(train_matrix, y_train)
        seconds = time.perf_counter() - start
        scores = estimator.predict_proba(val_matrix)[:, list(estimator.classes_).index(1)]
        metrics = evaluate(y_val, scores)
        run_id = None
        model_uri = None
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', estimator)])
        if tracking_uri:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("PBI08_Ephraim")
            with mlflow.start_run(run_name=f"pbi08/{batch_name}/{name}",
                                  tags={"member": "Ephraim", "pbi": "08", "batch": batch_name}) as run:
                mlflow.log_params({f"clf_{k}": v for k, v in estimator.get_params().items()})
                mlflow.log_params({"threshold": THRESHOLD, "seed": SEED})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
                mlflow.log_metric("fit_seconds", seconds)
                run_id = run.info.run_id
                model_info = mlflow.sklearn.log_model(pipeline, artifact_path='model')
                model_uri = model_info.model_uri
        out.append({"name": name, "pipeline": pipeline,
                    "metrics": metrics, "fit_seconds": seconds, "run_id": run_id, "model_uri": model_uri})
    return out


