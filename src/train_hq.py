"""Reproducible Han Qian validation benchmark; no test evaluation.

Call train_benchmark(customers, orders, labels) with raw DataFrames for a batch.
MLFLOW_TRACKING_URI can point to a shared service; without it tracking is local.
The returned comparison contains only validation results, not test labels.
"""

import os
from pathlib import Path
import tempfile
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from src.data_cleaning import clean_customers, clean_orders, clean_labels
from src.feature_engineering import fit_preprocessor, prepare_training_data, training_sample_weights


def model_candidates(seed=42):
    """Small, extensible model dictionary; add more sklearn-compatible models here."""
    return {
        "logistic_regression": LogisticRegression(max_iter=200, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=20, max_depth=12, min_samples_leaf=30,
                                                 class_weight="balanced", n_jobs=4, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15,
                                                                  learning_rate=0.1, random_state=seed),
        "xgboost": XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.1,
                                 tree_method="hist", n_jobs=4, random_state=seed),
        "lightgbm": LGBMClassifier(n_estimators=80, num_leaves=31, learning_rate=0.1,
                                   n_jobs=4, random_state=seed, verbosity=-1),
        "catboost": CatBoostClassifier(iterations=80, depth=5, learning_rate=0.1,
                                       thread_count=4, random_seed=seed, verbose=False,
                                       allow_writing_files=False),
    }


def validation_metrics(y, probabilities, threshold):
    """Calculate fraud-class metrics from validation probabilities only."""
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
    }


def rank_results(results):
    """Rank by validation F1, then AP as a transparent tie-breaker."""
    return sorted(results, key=lambda row: (-row["f1"], -row["average_precision"], row["model"], row["version"]))


def _tracking(experiment_name, tracking_uri):
    root = Path("artifacts/mlflow").resolve()
    local_uri = "sqlite:///" + str(root / "mlflow.db")
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or local_uri
    if uri == local_uri:
        root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name, artifact_location=(root / "artifacts").as_uri())
    else:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


def _log_run(name, version, model, preprocessor, y_val, probabilities, threshold,
             metrics, report, train_seconds, predict_seconds, member, weighting, extra_params=None, run_label=None, split_description="chronological 70/15/15; test untouched"):
    with mlflow.start_run(run_name=run_label or f"{member}-{name}-{version}") as run:
        mlflow.set_tags({"member": member, "model_name": name, "model_version": version,
                         "split": split_description, "dataset": "validation"})
        mlflow.log_params({"threshold": threshold, "class_weighting": weighting,
                           "preprocessing": "train-only median/log1p/robust/onehot",
                           "train_first_date": report["train_first_date"], "train_last_date": report["train_last_date"],
                           "validation_first_date": report["validation_first_date"],
                           "validation_last_date": report["validation_last_date"],
                           "train_rows": report["training_rows"], "validation_rows": report["validation_rows"],
                           "seed": 42})
        if extra_params:
            mlflow.log_params(extra_params)
        mlflow.log_params({"model__" + key: str(value) for key, value in model.get_params().items()})
        mlflow.log_dict(model.get_params(), "model_parameters.json")
        mlflow.log_metrics({**metrics, "training_seconds": train_seconds, "prediction_seconds": predict_seconds})
        precision, recall, _ = precision_recall_curve(y_val, probabilities)
        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set(xlabel="Recall", ylabel="Precision", title="Validation precision-recall curve")
        mlflow.log_figure(fig, "precision_recall.png")
        plt.close(fig)
        matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        fig, ax = plt.subplots()
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual", title="Validation confusion matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            prep_path = Path(directory) / "preprocessor.joblib"
            joblib.dump(model, model_path)
            joblib.dump(preprocessor, prep_path)
            mlflow.log_artifact(str(model_path), "model")
            mlflow.log_artifact(str(prep_path), "model")
            mlflow.log_dict({"threshold": threshold, "features": preprocessor.get_feature_names_out().tolist()}, "model/inference.json")
        return run.info.run_id


def train_benchmark(customers, orders, labels, tracking_uri=None, experiment_name="PBI06_HQ", member="Han Qian"):
    """Clean a new batch, fit on training, compare on validation, and log runs.

    Returns (ranked validation results, split report, tracking URI). The PBI 03
    include_test=False path does not construct test features or return test labels.
    No locally saved preprocessor is needed; each invocation fits a new one.
    """
    customers = clean_customers(customers)[0]
    orders = clean_orders(orders)[0]
    labels = clean_labels(labels)[0]
    x_train, y_train, x_val, y_val, _, _, report = prepare_training_data(customers, orders, labels, include_test=False)
    results, uri = train_prepared(x_train, y_train, x_val, y_val, report,
                                  tracking_uri=tracking_uri, experiment_name=experiment_name, member=member)
    return results, report, uri


def train_prepared(x_train, y_train, x_val, y_val, report, preprocessor=None,
                   tracking_uri=None, experiment_name="PBI06_HQ", member="Han Qian"):
    """Train the six baselines on PBI 03 training and validation data.

    If supplied, the preprocessor must have been fitted on x_train only. The
    notebook can reuse its PBI 03 preprocessor; other callers can omit it.
    """
    if set(y_train.unique()) != {0, 1} or set(y_val.unique()) != {0, 1}:
        raise ValueError("Training and validation must each contain both classes")
    uri = _tracking(experiment_name, tracking_uri)
    if preprocessor is None:
        preprocessor = fit_preprocessor(x_train)
    train = preprocessor.transform(x_train).astype(np.float32)
    val = preprocessor.transform(x_val).astype(np.float32)
    weights = training_sample_weights(y_train)
    results = []
    with threadpool_limits(limits=4):
        for name, model in model_candidates().items():
            print(f"Training {name}...", flush=True)
            # Histogram boosting and CatBoost use the same features in dense form.
            needs_dense = name in ("hist_gradient_boosting", "catboost")
            train_input = train.toarray() if needs_dense and hasattr(train, "toarray") else train
            val_input = val.toarray() if needs_dense and hasattr(val, "toarray") else val
            start = perf_counter()
            if name in ("hist_gradient_boosting", "xgboost", "lightgbm", "catboost"):
                model.fit(train_input, y_train, sample_weight=weights)
                weighting = "balanced training sample weights"
            else:
                model.fit(train_input, y_train)
                weighting = "balanced class weights"
            training_seconds = perf_counter() - start
            start = perf_counter()
            probabilities = model.predict_proba(val_input)[:, 1]
            prediction_seconds = perf_counter() - start
            metrics = validation_metrics(y_val, probabilities, 0.5)
            run_id = _log_run(name, "baseline", model, preprocessor, y_val, probabilities, 0.5, metrics,
                              report, training_seconds, prediction_seconds, member, weighting)
            row = {"model": name, "version": "baseline", "threshold": 0.5, **metrics,
                   "training_seconds": training_seconds, "prediction_seconds": prediction_seconds,
                   "parameters": model.get_params(), "run_id": run_id}
            results.append(row)
            print(f"Finished {name}: validation fraud F1 = {metrics['f1']:.4f}", flush=True)
    return rank_results(results), uri


def tuning_candidates():
    """Small fixed search; strength 0.5 means halfway to balanced weights."""
    return {
        "catboost": [
            ({}, 1.0),
            ({"iterations": 120, "depth": 6, "l2_leaf_reg": 4}, 1.0),
            ({"iterations": 120, "depth": 5, "learning_rate": 0.08, "l2_leaf_reg": 5}, 0.5),
        ],
        "lightgbm": [
            ({}, 1.0),
            ({"n_estimators": 120, "num_leaves": 15, "min_child_samples": 80, "reg_lambda": 3}, 1.0),
            ({"n_estimators": 120, "num_leaves": 15, "learning_rate": 0.08, "reg_lambda": 5}, 0.5),
        ],
        "random_forest": [
            ({}, 1.0),
            ({"n_estimators": 40, "max_depth": 16, "min_samples_leaf": 20}, 1.0),
            ({"n_estimators": 40, "max_depth": 12, "min_samples_leaf": 40, "class_weight": None}, 0.5),
        ],
    }


def chronological_folds(x_train, dates):
    """Two expanding training folds with whole-date validation boundaries."""
    import pandas as pd

    if not x_train.index.is_unique or not dates.index.equals(x_train.index):
        raise ValueError("Training dates must have the same unique index as training features")
    dates = pd.to_datetime(dates, errors="raise")
    if dates.isna().any():
        raise ValueError("Training dates cannot be missing")
    if len(dates) < 3:
        raise ValueError("At least three training dates are needed")
    ordered = dates.sort_values().reset_index(drop=True)
    cutoff1 = ordered.iloc[int(len(ordered) * 0.60)]
    cutoff2 = ordered.iloc[int(len(ordered) * 0.80)]
    folds = [((dates < cutoff1), (dates >= cutoff1) & (dates < cutoff2)),
             ((dates < cutoff2), (dates >= cutoff2))]
    if any(not train.any() or not validation.any() for train, validation in folds):
        raise ValueError("Not enough distinct dates for two chronological folds")
    return folds


def _fit_weighted(model, name, x, y, strength):
    """Apply one training-fold-only class-imbalance treatment."""
    if name == "random_forest" and strength == 1.0:
        model.fit(x, y)  # class_weight='balanced' is set on the model
    else:
        balanced = training_sample_weights(y)
        weights = 1 + strength * (balanced - 1)
        model.fit(x, y, sample_weight=weights)


def _log_inner_trial(name, candidate, fold, model, preprocessor, metrics, seconds, dates, strength, member, search_version="tuning_trial"):
    """Save an inner-fold trial, keeping it distinct from outer validation runs."""
    with mlflow.start_run(run_name=f"{member}-{name}-inner-{candidate}-fold-{fold}") as run:
        mlflow.set_tags({"member": member, "model_name": name, "model_version": search_version,
                         "dataset": "training_internal_validation"})
        mlflow.log_params({"candidate": candidate, "fold": fold, "threshold": 0.5,
                           "weight_strength": strength, "preprocessing": "fold-train-only median/log1p/robust/onehot",
                           **dates})
        mlflow.log_params({"model__" + key: str(value) for key, value in model.get_params().items()})
        mlflow.log_metrics({**metrics, "training_seconds": seconds})
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "model.joblib"
            preprocessor_path = Path(folder) / "preprocessor.joblib"
            joblib.dump(model, model_path)
            joblib.dump(preprocessor, preprocessor_path)
            mlflow.log_artifact(str(model_path), "model")
            mlflow.log_artifact(str(preprocessor_path), "model")
        return run.info.run_id


def tune_top_models(x_train, y_train, train_dates, x_val, y_val, report,
                    tracking_uri=None, member="Han Qian", candidates_override=None,
                    result_version="tuned"):
    """Search only within training, refit selected candidates, then score validation.

    Outer validation is never passed to candidate search. The test set is not
    accepted. This explicit exploratory function is separate from train_benchmark.
    """
    import pandas as pd

    if not x_train.index.equals(y_train.index) or not x_train.index.equals(train_dates.index):
        raise ValueError("Training features, labels and dates must have matching indices")
    if list(x_train.columns) != list(x_val.columns):
        raise ValueError("Training and validation feature schemas differ")
    if not x_train.index.intersection(x_val.index).empty:
        raise ValueError("Training and validation rows overlap")
    if train_dates.max() > pd.Timestamp(report["train_last_date"]) or train_dates.max() >= pd.Timestamp(report["validation_first_date"]):
        raise ValueError("Training dates must precede outer validation")
    folds = chronological_folds(x_train, train_dates)
    local_uri = "sqlite:///" + str(Path("artifacts/mlflow/mlflow.db").resolve())
    _tracking("PBI06_HQ", tracking_uri or local_uri)
    candidates = tuning_candidates() if candidates_override is None else candidates_override
    trials = []
    search_start = perf_counter()
    with threadpool_limits(limits=4):
        for fold_number, (train_mask, valid_mask) in enumerate(folds, start=1):
            inner_x = x_train.loc[train_mask]
            inner_y = y_train.loc[train_mask]
            valid_x = x_train.loc[valid_mask]
            valid_y = y_train.loc[valid_mask]
            if set(inner_y.unique()) != {0, 1} or set(valid_y.unique()) != {0, 1}:
                raise ValueError("Each inner training and validation fold needs both classes")
            fold_preprocessor = fit_preprocessor(inner_x)
            inner_matrix = fold_preprocessor.transform(inner_x).astype(np.float32)
            valid_matrix = fold_preprocessor.transform(valid_x).astype(np.float32)
            fold_dates = {
                "inner_train_first": str(train_dates.loc[train_mask].min().date()),
                "inner_train_last": str(train_dates.loc[train_mask].max().date()),
                "inner_val_first": str(train_dates.loc[valid_mask].min().date()),
                "inner_val_last": str(train_dates.loc[valid_mask].max().date()),
            }
            for name, options in candidates.items():
                for candidate, (params, strength) in enumerate(options, start=1):
                    print(f"Inner fold {fold_number}: {name} candidate {candidate}...", flush=True)
                    model = model_candidates()[name].set_params(**params)
                    train_input = inner_matrix.toarray() if name == "catboost" and hasattr(inner_matrix, "toarray") else inner_matrix
                    valid_input = valid_matrix.toarray() if name == "catboost" and hasattr(valid_matrix, "toarray") else valid_matrix
                    start = perf_counter()
                    _fit_weighted(model, name, train_input, inner_y, strength)
                    seconds = perf_counter() - start
                    probabilities = model.predict_proba(valid_input)[:, 1]
                    metrics = validation_metrics(valid_y, probabilities, 0.5)
                    run_id = _log_inner_trial(name, candidate, fold_number, model, fold_preprocessor,
                                              metrics, seconds, fold_dates, strength, member,
                                              search_version="advanced_tuning_trial" if result_version == "advanced_tuned" else "tuning_trial")
                    trials.append({"model": name, "candidate": candidate, "fold": fold_number,
                                   "inner_f1": metrics["f1"], "parameters": params,
                                   "weight_strength": strength, "run_id": run_id})
    search_seconds = perf_counter() - search_start
    trial_table = pd.DataFrame(trials)
    # The mean of the two inner F1 scores alone determines each model's candidate.
    scores = trial_table.groupby(["model", "candidate"])["inner_f1"].mean().reset_index()
    best = scores.sort_values(["model", "inner_f1", "candidate"], ascending=[True, False, True]).drop_duplicates("model")
    full_preprocessor = fit_preprocessor(x_train)
    train_matrix = full_preprocessor.transform(x_train).astype(np.float32)
    val_matrix = full_preprocessor.transform(x_val).astype(np.float32)
    results = []
    with threadpool_limits(limits=4):
        for _, choice in best.iterrows():
            name = choice["model"]
            candidate = int(choice["candidate"])
            params, strength = candidates[name][candidate - 1]
            model = model_candidates()[name].set_params(**params)
            train_input = train_matrix.toarray() if name == "catboost" and hasattr(train_matrix, "toarray") else train_matrix
            val_input = val_matrix.toarray() if name == "catboost" and hasattr(val_matrix, "toarray") else val_matrix
            print(f"Refitting selected {name} candidate {candidate} on all training data...", flush=True)
            start = perf_counter()
            _fit_weighted(model, name, train_input, y_train, strength)
            training_seconds = perf_counter() - start
            start = perf_counter()
            probabilities = model.predict_proba(val_input)[:, 1]
            prediction_seconds = perf_counter() - start
            metrics = validation_metrics(y_val, probabilities, 0.5)
            run_id = _log_run(name, result_version, model, full_preprocessor, y_val, probabilities, 0.5,
                              metrics, report, training_seconds, prediction_seconds, member,
                              f"training-only balanced weight strength {strength}",
                              extra_params={"selected_candidate": candidate,
                                            "inner_mean_f1": float(choice["inner_f1"]),
                                            "selection": "two chronological training-only folds"})
            results.append({"model": name, "version": result_version, "threshold": 0.5, **metrics,
                            "training_seconds": training_seconds, "prediction_seconds": prediction_seconds,
                            "parameters": model.get_params(), "selected_changes": params,
                            "weight_strength": strength, "inner_mean_f1": float(choice["inner_f1"]),
                            "run_id": run_id})
    return rank_results(results), trial_table, search_seconds



def advanced_candidates(seed=42):
    """Twenty reproducible bounded configurations per model, including the earlier winner."""
    grids = {
        "catboost": {
            "iterations": [80, 120, 160], "depth": [4, 5, 6],
            "learning_rate": [0.04, 0.08, 0.12], "l2_leaf_reg": [3, 5, 10],
            "subsample": [0.7, 0.85, 1.0], "weight_strength": [0.25, 0.5, 0.75, 1.0],
        },
        "lightgbm": {
            "n_estimators": [80, 120, 160], "num_leaves": [15, 23, 31],
            "learning_rate": [0.04, 0.08, 0.12], "reg_lambda": [2, 5, 10],
            "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_samples": [20, 50, 100], "weight_strength": [0.25, 0.5, 0.75, 1.0],
        },
        "random_forest": {
            "n_estimators": [30, 45, 60], "max_depth": [10, 12, 16],
            "min_samples_leaf": [15, 30, 50], "max_features": [0.65, 0.85, 1.0],
            "max_samples": [0.7, 0.85, 1.0], "weight_strength": [0.25, 0.5, 0.75, 1.0],
        },
    }
    previous = tuning_candidates()
    candidates = {}
    for name, grid in grids.items():
        candidates[name] = [previous[name][2]]
        for sampled in ParameterSampler(grid, n_iter=19, random_state=seed):
            params = dict(sampled)
            strength = params.pop("weight_strength")
            if name == "catboost":
                params["bootstrap_type"] = "Bernoulli"
            elif name == "lightgbm":
                params["subsample_freq"] = 1
            else:
                params["class_weight"] = "balanced" if strength == 1.0 else None
            candidates[name].append((params, strength))
    return candidates


def advanced_tune_top_models(x_train, y_train, train_dates, x_val, y_val, report,
                             tracking_uri=None, member="Han Qian"):
    """Run the bounded search separately from normal PBI 10 retraining."""
    return tune_top_models(x_train, y_train, train_dates, x_val, y_val, report,
                           tracking_uri=tracking_uri, member=member,
                           candidates_override=advanced_candidates(), result_version="advanced_tuned")


# The saved runs were selected by the advanced search on the original base files.
SAVED_ADVANCED_RUNS = {
    "lightgbm": "d6e45399a8934446aadeb01caddd7849",
    "catboost": "12c49c06191b47e2898a126c9f722e11",
    "random_forest": "a8998c12d8004c8e9f47db017db1f2f5",
}
BASE_DATA_HASHES = {
    "customers_base.csv": "501c06997d496e127d3e1b2c28d34b47dfe0dcf124f90abf83de60867f557c7e",
    "orders_base.csv": "934d9143861feac2c00aaa9bf782418e52959d13cfa7fd47b67ebf79f10e67f8",
    "labels_base.csv": "cde4cc62977ba48f4402a8bac756518fceb5e899cdbeb420a773936d376d5ce4",
}
ADVANCED_REFERENCE = Path("artifacts/ca1_advanced_reference.json")


def data_hashes():
    """Fingerprint source files without displaying rows or label values."""
    import hashlib

    hashes = {}
    for filename in BASE_DATA_HASHES:
        with open(filename, "rb") as source:
            hashes[filename] = hashlib.file_digest(source, "sha256").hexdigest()
    return hashes


def save_advanced_reference(results):
    """Remember newly rerun advanced models for later notebook executions."""
    import json

    reference = {"runs": {row["model"]: row["run_id"] for row in results}, "data_hashes": data_hashes()}
    ADVANCED_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    ADVANCED_REFERENCE.write_text(json.dumps(reference, indent=2), encoding="utf-8")


def load_saved_advanced_results(report, tracking_uri=None):
    """Load the three saved selected runs and the associated 120 trial summaries.

    The saved results are only valid for the same raw data and split. If either
    changes, explicitly rerun advanced search rather than reusing stale results.
    """
    import json
    import pandas as pd

    local_uri = "sqlite:///" + str(Path("artifacts/mlflow/mlflow.db").resolve())
    _tracking("PBI06_HQ", tracking_uri or local_uri)
    if ADVANCED_REFERENCE.exists():
        reference = json.loads(ADVANCED_REFERENCE.read_text(encoding="utf-8"))
    else:
        reference = {"runs": SAVED_ADVANCED_RUNS, "data_hashes": BASE_DATA_HASHES}
    if data_hashes() != reference["data_hashes"]:
        raise ValueError("Source files differ from the saved advanced search; set RUN_ADVANCED_SEARCH=True and rerun it.")
    client = mlflow.MlflowClient()
    candidates = advanced_candidates()
    results = []
    starts = []
    for name, run_id in reference["runs"].items():
        run = client.get_run(run_id)
        if (run.info.status != "FINISHED" or run.data.tags.get("model_version") != "advanced_tuned"
                or run.data.tags.get("model_name") != name or float(run.data.params.get("threshold", -1)) != 0.5):
            raise ValueError(f"Saved advanced run is unavailable or invalid: {run_id}")
        for key in ("train_first_date", "train_last_date", "validation_first_date", "validation_last_date", "train_rows", "validation_rows"):
            report_key = {"train_rows": "training_rows", "validation_rows": "validation_rows"}.get(key, key)
            if str(run.data.params[key]) != str(report[report_key]):
                raise ValueError("Saved advanced models do not match the current split; rerun advanced search.")
        candidate = int(run.data.params["selected_candidate"])
        changes, strength = candidates[name][candidate - 1]
        for key, value in changes.items():
            if run.data.params.get("model__" + key) != str(value):
                raise ValueError("Saved hyperparameters differ from the current configuration; rerun advanced search.")
        if run.data.params.get("class_weighting") != f"training-only balanced weight strength {strength}":
            raise ValueError("Saved class weighting differs from the current configuration; rerun advanced search.")
        parameter_file = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model_parameters.json")
        parameters = json.loads(Path(parameter_file).read_text(encoding="utf-8"))
        metrics = run.data.metrics
        row = {"model": name, "version": "advanced_tuned", "threshold": 0.5,
               **{key: metrics[key] for key in ("f1", "precision", "recall", "average_precision", "roc_auc", "accuracy", "tn", "fp", "fn", "tp", "false_positive_rate", "training_seconds", "prediction_seconds")},
               "parameters": parameters, "selected_changes": changes, "weight_strength": strength,
               "inner_mean_f1": float(run.data.params["inner_mean_f1"]), "run_id": run_id}
        for count in ("tn", "fp", "fn", "tp"):
            row[count] = int(row[count])
        results.append(row)
        starts.append(run.info.start_time)
    if set(reference["runs"]) != {"lightgbm", "catboost", "random_forest"}:
        raise ValueError("Saved advanced reference must contain all three models")
    experiment = client.get_experiment_by_name("PBI06_HQ")
    runs = client.search_runs([experiment.experiment_id], filter_string="tags.model_version = 'advanced_tuning_trial'",
                              order_by=["start_time DESC"], max_results=10000)
    runs = [run for run in runs if run.info.start_time < min(starts)][:120]
    keys = {(run.data.tags["model_name"], int(run.data.params["candidate"]), int(run.data.params["fold"])) for run in runs}
    expected = {(name, candidate, fold) for name in reference["runs"] for candidate in range(1, 21) for fold in (1, 2)}
    if len(runs) != 120 or keys != expected or any(run.info.status != "FINISHED" for run in runs):
        raise ValueError("The 120 saved advanced trial records cannot be identified; rerun advanced search.")
    trials = []
    for run in runs:
        name = run.data.tags["model_name"]
        candidate = int(run.data.params["candidate"])
        changes, strength = candidates[name][candidate - 1]
        trials.append({"model": name, "candidate": candidate, "fold": int(run.data.params["fold"]),
                       "inner_f1": run.data.metrics["f1"], "parameters": changes,
                       "weight_strength": strength, "run_id": run.info.run_id})
    return rank_results(results), pd.DataFrame(trials).sort_values(["fold", "model", "candidate"])


def select_f1_threshold(y, probabilities):
    """Maximize F1 on training-only predictions; break exact ties upwards."""
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    if len(thresholds) == 0:
        raise ValueError("Threshold selection needs nonempty predictions")
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator != 0)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best]), float(f1[best])


def optimize_advanced_thresholds(x_train, y_train, train_dates, x_val, y_val, report,
                                 advanced_results, tracking_uri=None, member="Han Qian"):
    """Select thresholds on training-only chronological predictions, then score validation.

    Hyperparameters are fixed from the saved advanced search. Those same inner
    folds helped select them, so out-of-fold threshold estimates are optimistic.
    No test data is accepted or constructed by this function.
    """
    import pandas as pd

    if not x_train.index.equals(y_train.index) or not x_train.index.equals(train_dates.index):
        raise ValueError("Training features, labels and dates must have matching indices")
    if list(x_train.columns) != list(x_val.columns) or not x_train.index.intersection(x_val.index).empty:
        raise ValueError("Training and validation features must have the same schema and disjoint rows")
    if train_dates.max() > pd.Timestamp(report["train_last_date"]) or train_dates.max() >= pd.Timestamp(report["validation_first_date"]):
        raise ValueError("Training dates must precede outer validation")
    by_name = {row["model"]: row for row in advanced_results}
    if len(advanced_results) != 3 or set(by_name) != {"lightgbm", "catboost", "random_forest"}:
        raise ValueError("Exactly one advanced result is required for each selected model")
    folds = chronological_folds(x_train, train_dates)
    local_uri = "sqlite:///" + str(Path("artifacts/mlflow/mlflow.db").resolve())
    _tracking("PBI06_HQ", tracking_uri or local_uri)
    for name, original in by_name.items():
        current = model_candidates()[name].set_params(**original["selected_changes"]).get_params()
        saved = original["parameters"]
        if set(current) != set(saved) or any(str(current[key]) != str(saved[key]) for key in current):
            raise ValueError(f"{name} hyperparameters differ from the saved full-training model")
    selections = {}
    start_all = perf_counter()
    with threadpool_limits(limits=4):
        for name, original in by_name.items():
            print(f"Selecting training-only threshold for {name}...", flush=True)
            fold_scores = []
            fold_labels = []
            start = perf_counter()
            for train_mask, valid_mask in folds:
                inner_x, inner_y = x_train.loc[train_mask], y_train.loc[train_mask]
                valid_x, valid_y = x_train.loc[valid_mask], y_train.loc[valid_mask]
                if set(inner_y.unique()) != {0, 1} or set(valid_y.unique()) != {0, 1}:
                    raise ValueError("Each inner training and validation fold needs both classes")
                preprocessor = fit_preprocessor(inner_x)
                training = preprocessor.transform(inner_x).astype(np.float32)
                validation = preprocessor.transform(valid_x).astype(np.float32)
                if name == "catboost" and hasattr(training, "toarray"):
                    training, validation = training.toarray(), validation.toarray()
                model = model_candidates()[name].set_params(**original["selected_changes"])
                _fit_weighted(model, name, training, inner_y, original["weight_strength"])
                fold_scores.append(model.predict_proba(validation)[:, 1])
                fold_labels.append(valid_y.to_numpy())
            probabilities = np.concatenate(fold_scores)
            labels = np.concatenate(fold_labels)
            threshold, oof_f1 = select_f1_threshold(labels, probabilities)
            selections[name] = {"threshold": threshold, "oof_f1": oof_f1,
                                "oof_rows": len(labels), "selection_seconds": perf_counter() - start}
    # All thresholds are selected before any outer-validation label is scored.
    results = []
    with threadpool_limits(limits=4):
        for name, original in by_name.items():
            model_file = mlflow.artifacts.download_artifacts(run_id=original["run_id"], artifact_path="model/model.joblib")
            prep_file = mlflow.artifacts.download_artifacts(run_id=original["run_id"], artifact_path="model/preprocessor.joblib")
            model = joblib.load(model_file)
            preprocessor = joblib.load(prep_file)
            val = preprocessor.transform(x_val).astype(np.float32)
            if name == "catboost" and hasattr(val, "toarray"):
                val = val.toarray()
            start = perf_counter()
            probabilities = model.predict_proba(val)[:, 1]
            prediction_seconds = perf_counter() - start
            selection = selections[name]
            metrics = validation_metrics(y_val, probabilities, selection["threshold"])
            run_id = _log_run(name, "threshold_optimized", model, preprocessor, y_val, probabilities,
                              selection["threshold"], metrics, report, original["training_seconds"],
                              prediction_seconds, member, f"training-only balanced weight strength {original['weight_strength']}",
                              extra_params={"parent_advanced_run": original["run_id"],
                                            "threshold_selection": "pooled chronological training folds",
                                            "training_oof_f1": selection["oof_f1"],
                                            "training_oof_rows": selection["oof_rows"],
                                            "fold1_train_last": str(train_dates.loc[folds[0][0]].max().date()),
                                            "fold1_validation_first": str(train_dates.loc[folds[0][1]].min().date()),
                                            "fold1_validation_last": str(train_dates.loc[folds[0][1]].max().date()),
                                            "fold2_train_last": str(train_dates.loc[folds[1][0]].max().date()),
                                            "fold2_validation_first": str(train_dates.loc[folds[1][1]].min().date()),
                                            "fold2_validation_last": str(train_dates.loc[folds[1][1]].max().date())})
            mlflow.MlflowClient().log_metric(run_id, "threshold_selection_seconds", selection["selection_seconds"])
            results.append({"model": name, "version": "threshold_optimized", **metrics,
                            "threshold": selection["threshold"], "oof_f1": selection["oof_f1"],
                            "oof_rows": selection["oof_rows"], "selection_seconds": selection["selection_seconds"],
                            "training_seconds": original["training_seconds"], "prediction_seconds": prediction_seconds,
                            "parameters": original["parameters"], "selected_changes": original["selected_changes"],
                            "weight_strength": original["weight_strength"], "run_id": run_id})
    return rank_results(results), perf_counter() - start_all


# Frozen PBI 06 comparison: these runs retain the original validation evidence.
FINAL_COMPARISON_RUNS = {
    "baseline": {
        "catboost": "59a72ead8d34483ea1f82644b4511d55",
        "random_forest": "25bb1427c3b647668c8037fc761e88a9",
        "xgboost": "0aeac7b9725849fead64c77b5705b0bb",
        "lightgbm": "91db7fa7fc694ce3952c515cc386405a",
        "hist_gradient_boosting": "f6e01df787d14eb3b7271f1dc047711a",
        "logistic_regression": "d7b938d37c054cea9f9a8df611bdecfd",
    },
    "tuned": {
        "lightgbm": "47e6724a7679483dab87550bbc5f5f9c",
        "catboost": "b69a66a13c2c470aaf2cda872af3a52f",
        "random_forest": "e85dff34a5424913ac43dba5f14dcbc1",
    },
    "advanced_tuned": SAVED_ADVANCED_RUNS,
    "threshold_optimized": {
        "catboost": "ad724329187b4f99a3845cf8abadb0aa",
        "lightgbm": "fad3c3f16b9a4476aecab034eb002831",
        "random_forest": "740d1cfc49be43808a6ca270d4fffdfe",
    },
}


def load_finalized_results(report, tracking_uri=None):
    """Load the original 15 validation versions and trial evidence without training.

    The comparison is specific to the original data and date split. It requires
    the local historical MLflow store and fails clearly if evidence is missing.
    """
    import json
    import pandas as pd

    local_uri = "sqlite:///" + str(Path("artifacts/mlflow/mlflow.db").resolve())
    _tracking("PBI06_HQ", tracking_uri or local_uri)
    if data_hashes() != BASE_DATA_HASHES:
        raise ValueError("Historical comparison is only valid for the original base files; use the training interface for new data.")
    client = mlflow.MlflowClient()
    groups = {}
    initial_starts = []
    try:
        for version, names in FINAL_COMPARISON_RUNS.items():
            rows = []
            for name, run_id in names.items():
                run = client.get_run(run_id)
                if run.info.status != "FINISHED" or run.data.tags.get("model_name") != name or run.data.tags.get("model_version") != version:
                    raise ValueError(f"Invalid historical run {run_id}")
                for key in ("train_first_date", "train_last_date", "validation_first_date", "validation_last_date", "train_rows", "validation_rows"):
                    report_key = {"train_rows": "training_rows", "validation_rows": "validation_rows"}.get(key, key)
                    if str(run.data.params[key]) != str(report[report_key]):
                        raise ValueError(f"Historical run {run_id} does not match the current split")
                for path in ("model/model.joblib", "model/preprocessor.joblib", "model/inference.json", "precision_recall.png"):
                    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
                parameter_file = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model_parameters.json")
                params = json.loads(Path(parameter_file).read_text(encoding="utf-8"))
                metrics = run.data.metrics
                row = {"model": name, "version": version, "threshold": float(run.data.params["threshold"]),
                       **{key: metrics[key] for key in ("f1", "precision", "recall", "average_precision", "roc_auc", "accuracy", "tn", "fp", "fn", "tp", "false_positive_rate", "training_seconds", "prediction_seconds")},
                       "parameters": params, "run_id": run_id}
                for count in ("tn", "fp", "fn", "tp"):
                    row[count] = int(row[count])
                if version in ("tuned", "advanced_tuned"):
                    candidate = int(run.data.params["selected_candidate"])
                    choices = tuning_candidates() if version == "tuned" else advanced_candidates()
                    row["selected_changes"], row["weight_strength"] = choices[name][candidate - 1]
                    row["inner_mean_f1"] = float(run.data.params["inner_mean_f1"])
                    if version == "tuned":
                        initial_starts.append(run.info.start_time)
                if version == "threshold_optimized":
                    parent = client.get_run(run.data.params["parent_advanced_run"])
                    if parent.info.run_id != FINAL_COMPARISON_RUNS["advanced_tuned"][name]:
                        raise ValueError(f"Threshold run {run_id} has an unexpected parent")
                    candidate = int(parent.data.params["selected_candidate"])
                    row["selected_changes"], row["weight_strength"] = advanced_candidates()[name][candidate - 1]
                    row["oof_f1"] = float(run.data.params["training_oof_f1"])
                    row["oof_rows"] = int(run.data.params["training_oof_rows"])
                    row["selection_seconds"] = metrics["threshold_selection_seconds"]
                rows.append(row)
            groups[version] = rank_results(rows)
        experiment = client.get_experiment_by_name("PBI06_HQ")
        initial = client.search_runs([experiment.experiment_id], filter_string="tags.model_version = 'tuning_trial'",
                                     order_by=["start_time DESC"], max_results=10000)
        initial = [run for run in initial if run.info.start_time < min(initial_starts)][:18]
        expected = {(name, candidate, fold) for name in ("catboost", "lightgbm", "random_forest") for candidate in (1, 2, 3) for fold in (1, 2)}
        keys = {(run.data.tags["model_name"], int(run.data.params["candidate"]), int(run.data.params["fold"])) for run in initial}
        if len(initial) != 18 or keys != expected or any(run.info.status != "FINISHED" for run in initial):
            raise ValueError("The 18 historical inner-fold trial records are incomplete")
        trials = []
        for run in initial:
            name = run.data.tags["model_name"]
            candidate = int(run.data.params["candidate"])
            changes, strength = tuning_candidates()[name][candidate - 1]
            trials.append({"model": name, "candidate": candidate, "fold": int(run.data.params["fold"]),
                           "inner_f1": run.data.metrics["f1"], "parameters": changes,
                           "weight_strength": strength, "run_id": run.info.run_id})
        advanced, advanced_trials = load_saved_advanced_results(report, tracking_uri=local_uri)
        if {r["run_id"] for r in advanced} != set(FINAL_COMPARISON_RUNS["advanced_tuned"].values()):
            raise ValueError("Current advanced reference differs from the frozen historical comparison")
        return groups, pd.DataFrame(trials).sort_values(["fold", "model", "candidate"]), advanced_trials
    except Exception as exc:
        raise RuntimeError("Required local PBI 06 historical MLflow runs or artifacts are missing or inconsistent; restore the original artifacts/mlflow directory. " + str(exc)) from exc


# Fixed configurations and decision thresholds selected during the documented experiments.
SELECTED_MODELS = {
    "catboost": {"params": {"subsample": 0.7, "learning_rate": 0.12, "l2_leaf_reg": 3,
                            "iterations": 160, "depth": 6, "bootstrap_type": "Bernoulli"},
                 "weight_strength": 0.5, "threshold": 0.45684071728727366},
    "lightgbm": {"params": {"subsample": 1.0, "reg_lambda": 10, "num_leaves": 15,
                            "n_estimators": 120, "min_child_samples": 50, "learning_rate": 0.04,
                            "colsample_bytree": 0.7, "subsample_freq": 1},
                 "weight_strength": 0.75, "threshold": 0.5},
}


def train_selected_models(x_train, y_train, x_val, y_val, split_report, tracking_uri=None,
                          experiment_name="PBI06_HQ", member="Han Qian", batch_name="retrain"):
    """Refit fixed CatBoost and LightGBM on prepared training data; never search.

    Inputs are unencoded PBI 03 feature frames and binary label Series. The
    report must describe their chronological split. Returns per-model metrics,
    MLflow run IDs and model/preprocessor/inference artifact references.
    No test data or historical model artifact is required.
    """
    from src.feature_engineering import INPUT_FEATURES

    if list(x_train.columns) != INPUT_FEATURES or list(x_val.columns) != INPUT_FEATURES:
        raise ValueError("Training and validation must have the PBI 03 feature schema")
    if not x_train.index.equals(y_train.index) or not x_val.index.equals(y_val.index):
        raise ValueError("Each feature frame and its labels must have matching indices")
    if not x_train.index.is_unique or not x_val.index.is_unique or not x_train.index.intersection(x_val.index).empty:
        raise ValueError("Training and validation row indices must be unique and disjoint")
    if set(y_train.unique()) != {0, 1} or set(y_val.unique()) != {0, 1}:
        raise ValueError("Training and validation must each contain both binary classes")
    if split_report["train_last_date"] >= split_report["validation_first_date"]:
        raise ValueError("Training dates must precede validation dates")
    if split_report["training_rows"] != len(x_train) or split_report["validation_rows"] != len(x_val):
        raise ValueError("Split report row counts must match the supplied frames")
    uri = _tracking(experiment_name, tracking_uri)
    preprocessor = fit_preprocessor(x_train)
    train = preprocessor.transform(x_train).astype(np.float32)
    val = preprocessor.transform(x_val).astype(np.float32)
    results = []
    with threadpool_limits(limits=4):
        for name, config in SELECTED_MODELS.items():
            model = model_candidates()[name].set_params(**config["params"])
            train_input = train.toarray() if name == "catboost" and hasattr(train, "toarray") else train
            val_input = val.toarray() if name == "catboost" and hasattr(val, "toarray") else val
            start = perf_counter()
            _fit_weighted(model, name, train_input, y_train, config["weight_strength"])
            training_seconds = perf_counter() - start
            start = perf_counter()
            probabilities = model.predict_proba(val_input)[:, 1]
            prediction_seconds = perf_counter() - start
            metrics = validation_metrics(y_val, probabilities, config["threshold"])
            run_id = _log_run(name, "selected_retrain", model, preprocessor, y_val, probabilities,
                              config["threshold"], metrics, split_report, training_seconds,
                              prediction_seconds, member, f"training-only balanced weight strength {config['weight_strength']}",
                              extra_params={"batch_name": batch_name, "selection": "fixed PBI 06 configuration; no search"},
                              run_label=f"{member}-{batch_name}-{name}-selected_retrain",
                              split_description="chronological training/validation; test not supplied")
            base_uri = mlflow.MlflowClient().get_run(run_id).info.artifact_uri.rstrip("/")
            results.append({"model": name, "run_id": run_id, "threshold": config["threshold"],
                            "metrics": metrics, "training_seconds": training_seconds,
                            "prediction_seconds": prediction_seconds,
                            "model_uri": base_uri + "/model/model.joblib",
                            "preprocessor_uri": base_uri + "/model/preprocessor.joblib",
                            "inference_uri": base_uri + "/model/inference.json"})
    return results
