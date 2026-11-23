"""PBI 07 - Nithish's fraud models (three versions, one change at a time).

    v1_logistic_default      logistic regression, cut-off 0.5
    v2_logistic_tuned_cutoff same model, cut-off chosen on recent training data
    v3_boosting_tuned_cutoff gradient boosting, same features, same cut-off rule

All versions use the team's PBI 02 cleaning and PBI 03 chronological split.
The outer validation period is shared with the team. A chronological slice of
the training period is reserved for threshold selection, so the tuned cutoff
and final evaluation use the same fitted model. The extra customer feature is
days since the customer's first recorded order, not verified account age.

PBI 10 entry point: train_nithish_models(customers, orders, labels).
Set MLFLOW_TRACKING_URI to log to the shared service; otherwise runs are
explicitly identified as local-only. The held-out test rows are not returned
to or used by this training routine.
"""

import logging
import os
from pathlib import Path
from time import perf_counter
import warnings
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from scipy import sparse

from src.feature_engineering import (
    CATEGORICAL_FEATURES, DATE_FEATURES, INPUT_FEATURES, NUMERIC_FEATURES,
    build_preprocessor, prepare_training_data,
)
from src.data_access import VALIDATION_START
from src.model_evaluation import best_f1_threshold, recent_training_window as _recent_training_window

MEMBER = "nith-ish22"
MEMBER_NAME = "Nithish"
EXPERIMENT = "PBI07_Nithish"
SEED = 42

AMOUNTS = NUMERIC_FEATURES
CALENDAR = DATE_FEATURES
CATEGORIES = CATEGORICAL_FEATURES
FEATURES = INPUT_FEATURES
TUNING_WINDOW = 0.20                                  # latest whole dates covering about 20% of training rows
REQUIREMENTS = [f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}", f"mlflow=={mlflow.__version__}"]
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)   # hide the routine pickle notice


# --------------------------------------------------------------------------- features

def prepare_data(customers, orders, labels):
    """Return the shared eight features and chronological PBI 03 split."""
    x_train, y_train, x_val, y_val, _, _, report, train_dates = prepare_training_data(
        customers, orders, labels, include_test=False, include_training_dates=True,
        validation_start_date=VALIDATION_START)
    return x_train, y_train, x_val, y_val, train_dates, report


# --------------------------------------------------------------------------- models

def logistic_pipeline():
    """Scorecard-style model: every feature adds or removes risk independently."""
    model = LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED)
    return Pipeline([("prepare", build_preprocessor()), ("model", model)])


def _dense_if_sparse(values):
    """Change matrix storage only for estimators that require dense arrays."""
    return values.toarray() if sparse.issparse(values) else values


def boosting_pipeline():
    """Shared encoded inputs; dense conversion is storage adaptation for HGB."""
    model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=100,
        class_weight="balanced", early_stopping=False, random_state=SEED)
    return Pipeline([
        ("prepare", build_preprocessor()),
        ("dense", FunctionTransformer(_dense_if_sparse, feature_names_out="one-to-one")),
        ("model", model),
    ])


VERSIONS = {
    "v1_logistic_default": {"build": logistic_pipeline, "tune_cutoff": False,
                            "threshold_rule": "fixed 0.5",
                            "change": "baseline: explainable logistic regression, cut-off 0.5"},
    "v2_logistic_tuned_cutoff": {"build": logistic_pipeline, "tune_cutoff": True,
                                 "threshold_rule": "maximum F1 on latest 20% of outer training dates",
                                 "change": "same model as v1; cut-off chosen on recent training data"},
    "v3_boosting_tuned_cutoff": {"build": boosting_pipeline, "tune_cutoff": True,
                                  "threshold_rule": "maximum F1 on latest 20% of outer training dates",
                                 "change": "same features and cut-off rule as v2; gradient boosting model"},
}


# --------------------------------------------------------------------------- cut-off

def recent_training_window(train_dates, share=TUNING_WINDOW):
    """Boolean mask of the latest whole dates in training (about `share` of rows)."""
    return _recent_training_window(train_dates, share)


def best_f1_cutoff(y, scores):
    """Cut-off with the highest fraud F1 (the team's ranking metric)."""
    return best_f1_threshold(y, scores)


def choose_cutoff(model, x_train, y_train, train_dates):
    """Choose a threshold using the same already-fitted model later evaluated."""
    recent = recent_training_window(train_dates)
    return best_f1_cutoff(y_train.loc[recent], model.predict_proba(x_train.loc[recent])[:, 1])


# --------------------------------------------------------------------------- MLflow

def _set_tracking(tracking_uri, experiment_name):
    local = Path("artifacts/mlflow").resolve()
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    else:                                              # same local store layout as PBI 06
        local.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///" + str(local / "mlflow.db"))
    parsed_uri = urlparse(mlflow.get_tracking_uri())
    if parsed_uri.scheme not in {"http", "https"} or parsed_uri.hostname in {"localhost", "127.0.0.1", "::1"}:
        logging.getLogger(__name__).warning(
            "MLflow is using local tracking; this run does not provide shared-MLflow evidence."
        )
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name, artifact_location=(local / "artifacts").as_uri())
    mlflow.set_experiment(experiment_name)


def _figures(y_val, scores, cutoff, metrics, title):
    precision, recall, _ = precision_recall_curve(y_val, scores)
    curve, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.scatter([metrics["recall"]], [metrics["precision"]], color="red", zorder=3,
               label=f"cut-off {cutoff:.2f}")
    ax.set(xlabel="Recall (fraud caught)", ylabel="Precision (flags that are fraud)", title=title)
    ax.legend()
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    confusion, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["legit", "fraud"], yticklabels=["legit", "fraud"],
           xlabel="Predicted", ylabel="Actual", title=title)
    return curve, confusion


def _log_run(version, spec, model, cutoff, metrics, x_val, y_val, scores, report, seconds, batch_name,
             model_fit_rows, threshold_calibration_rows, train_dates):
    with mlflow.start_run(run_name=f"{MEMBER}-{batch_name}-{version}") as run:
        mlflow.set_tags({"member": MEMBER, "member_name": MEMBER_NAME, "pbi": "07", "model_version": version,
                         "model_name": type(model[-1]).__name__, "dataset": "validation",
                         "split": "outer chronological 70/15/15; inner fit/calibration 80/20; test untouched",
                         "change_from_previous": spec["change"]})
        mlflow.log_params({
            "threshold": cutoff, "batch_name": batch_name, "seed": SEED,
            "threshold_rule": spec["threshold_rule"],
            "features": ", ".join(FEATURES), "class_weighting": "balanced",
            "excluded": "ids, refund_value, 50-day customer totals (timing unverified)",
            "train_rows": report["training_rows"], "validation_rows": report["validation_rows"],
            "model_fit_rows": model_fit_rows,
            "threshold_calibration_rows": threshold_calibration_rows,
            "threshold_calibration_start": str(train_dates.loc[recent_training_window(train_dates)].min().date()),
            "threshold_calibration_end": str(train_dates.loc[recent_training_window(train_dates)].max().date()),
            "train_first_date": report["train_first_date"], "train_last_date": report["train_last_date"],
            "validation_first_date": report["validation_first_date"],
            "validation_last_date": report["validation_last_date"]})
        mlflow.log_params({"model__" + key: str(value) for key, value in model[-1].get_params().items()})
        mlflow.log_params({"preprocess__" + key: str(value)[:250]
                           for key, value in model[0].get_params(deep=True).items()})
        mlflow.log_param(
            "preprocessing_summary",
            "shared PBI 03 median imputation, log1p, robust scaling, one-hot encoding",
        )
        mlflow.log_param("preprocessing_schema", ", ".join(FEATURES))
        mlflow.log_param(
            "histgradientboosting_adaptation",
            "shared one-hot inputs; removed native categorical and monotonic constraints",
        )
        mlflow.log_metrics({**metrics, "training_seconds": seconds})
        curve, confusion = _figures(y_val, scores, cutoff, metrics, version)
        mlflow.log_figure(curve, "precision_recall.png")
        mlflow.log_figure(confusion, "confusion_matrix.png")
        plt.close(curve)
        plt.close(confusion)
        mlflow.log_dict({"threshold": cutoff, "threshold_rule": spec["threshold_rule"],
                         "model_version": version, "features_in_order": FEATURES,
                         "first_order_feature_semantics": (
                             "order date minus first recorded order date; missing when unknown or future; "
                             "source must represent point-in-time history"
                         )},
                        "inference.json")
        # One artifact = preprocessing + model. Load it and call predict_proba on FEATURES.
        # Our own trusted model, so plain pickling is fine (same trust as the team's joblib files).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # MLflow's integer-output schema hint
            mlflow.sklearn.log_model(model, name="model", input_example=x_val.head(5),
                                     serialization_format="cloudpickle", pip_requirements=REQUIREMENTS)
        return run.info.run_id


# --------------------------------------------------------------------------- entry point

def validation_metrics(y_true, scores, cutoff):
    """Compute the team's standard binary fraud metrics from fixed scores."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores) or len(y_true) == 0:
        raise ValueError("Validation labels and scores must be non-empty one-dimensional arrays of equal length")
    if not np.isin(y_true, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Validation labels must be binary and scores must be finite")
    if ((scores < 0) | (scores > 1)).any() or not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Validation probabilities and cutoff must be between 0 and 1")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Validation labels must contain both fraud classes")
    y_true = y_true.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, scores >= cutoff, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "average_precision": float(average_precision_score(y_true, scores)),
            "roc_auc": float(roc_auc_score(y_true, scores))}


def evaluate(model, x_val, y_val, cutoff):
    """Validation scores and team metrics, plus the share of orders sent to review."""
    scores = model.predict_proba(x_val)[:, 1]
    metrics = validation_metrics(y_val, scores, cutoff)
    metrics["review_rate"] = float((scores >= cutoff).mean())
    return scores, metrics


def train_nithish_models(customers, orders, labels, tracking_uri=None, experiment_name=EXPERIMENT,
                         batch_name="base", versions=None, log=True, prepared_data=None,
                         current_batch_indices=None):
    """Clean tables in, three trained versions out. Call this from the PBI 10 pipeline.

    customers, orders, labels: retained outputs of the PBI 02 cleaners.
    Returns one dict per version: metrics, cut-off, fitted model, scores and,
    when log=True, the MLflow run ID and model URI ("runs:/<id>/model").
    """
    requested_versions = list(VERSIONS) if versions is None else list(versions)
    if not requested_versions or len(set(requested_versions)) != len(requested_versions):
        raise ValueError("versions must contain one or more distinct known model-version names")
    unknown = sorted(set(requested_versions) - set(VERSIONS))
    if unknown:
        raise ValueError(f"Unknown model version(s): {unknown}. Choose from {list(VERSIONS)}")

    x_train, y_train, x_val, y_val, train_dates, report = (prepare_data(customers, orders, labels)
                                                         if prepared_data is None else prepared_data)
    if log:
        _set_tracking(tracking_uri, experiment_name)
    recent = recent_training_window(train_dates)
    fit_rows = ~recent
    if current_batch_indices is not None and not x_train.loc[fit_rows].index.isin(current_batch_indices).any():
        raise ValueError('No current-batch rows are available for Nithish model fitting')
    fitted, cutoffs, results = {}, {}, []
    for version in requested_versions:
        spec = VERSIONS[version]
        build = spec["build"]
        start = perf_counter()
        if build not in fitted:                          # versions of a model family share one fitted model
            fitted[build] = build().fit(x_train.loc[fit_rows], y_train.loc[fit_rows])
        if spec["tune_cutoff"] and build not in cutoffs:
            cutoffs[build] = choose_cutoff(fitted[build], x_train, y_train, train_dates)
        seconds = perf_counter() - start
        model = fitted[build]
        cutoff = cutoffs[build] if spec["tune_cutoff"] else 0.5
        scores, metrics = evaluate(model, x_val, y_val, cutoff)
        result = {"member": MEMBER, "version": version, "change": spec["change"], "threshold": cutoff,
                  "metrics": metrics, "training_seconds": seconds, "model": model, "scores": scores,
                  "model_fit_rows": int(fit_rows.sum()), "threshold_calibration_rows": int(recent.sum()),
                  "threshold_rule": spec["threshold_rule"],
                  "tracking_uri": mlflow.get_tracking_uri() if log else None}
        if log:
            run_id = _log_run(version, spec, model, cutoff, metrics, x_val, y_val, scores,
                              report, seconds, batch_name, int(fit_rows.sum()), int(recent.sum()), train_dates)
            result.update({"run_id": run_id, "model_uri": f"runs:/{run_id}/model"})
        results.append(result)
    return results


if __name__ == "__main__":
    from src.data_cleaning import clean_customers, clean_labels, clean_orders

    tables = [clean(pd.read_csv(name, low_memory=False))[0] for clean, name in [
        (clean_customers, "customers_base.csv"), (clean_orders, "orders_base.csv"),
        (clean_labels, "labels_base.csv")]]
    for row in train_nithish_models(*tables):
        m = row["metrics"]
        print(f"{row['version']:28s} F1 {m['f1']:.4f}  precision {m['precision']:.4f}  "
              f"recall {m['recall']:.4f}  AP {m['average_precision']:.4f}  run {row.get('run_id')}")


