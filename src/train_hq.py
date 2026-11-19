"""Reproducible Han Qian validation benchmark; no test evaluation.

Call train_benchmark(customers, orders, labels) with raw DataFrames for a batch.
MLFLOW_TRACKING_URI can point to a shared service; without it tracking is local.
The returned comparison contains only validation results, not test labels.
"""

import os
from pathlib import Path
import tempfile
from time import perf_counter
from uuid import uuid4

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from src.feature_engineering import fit_preprocessor, training_sample_weights
from src.model_evaluation import best_f1_threshold, recent_training_window


def model_candidates(seed=42):
    """Small, extensible model dictionary; add more sklearn-compatible models here."""
    return {
        "logistic_regression": LogisticRegression(max_iter=200, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=20, max_depth=12, min_samples_leaf=30,
                                                 class_weight="balanced", n_jobs=4, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15,
                                                                  learning_rate=0.1, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=80, max_depth=3,
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
             metrics, report, train_seconds, predict_seconds, member, weighting,
             extra_params=None, run_label=None, split_description="chronological 70/15/15; test untouched",
             run_tags=None):
    with mlflow.start_run(run_name=run_label or f"{member}-{name}-{version}") as run:
        tags = {"member": member, "model_name": name, "model_version": version,
                "split": split_description, "dataset": "validation"}
        if run_tags:
            tags.update(run_tags)
        mlflow.set_tags(tags)
        mlflow.log_params({"threshold": threshold, "class_weighting": weighting,
                           "preprocessing": "train-only median/log1p/robust/onehot",
                           "train_first_date": report["train_first_date"], "train_last_date": report["train_last_date"],
                           "validation_first_date": report["validation_first_date"],
                           "validation_last_date": report["validation_last_date"],
                           "train_rows": report["training_rows"], "validation_rows": report["validation_rows"],
                           "seed": 42})
        if extra_params:
            mlflow.log_params({key: str(value) for key, value in extra_params.items()})
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


def train_benchmark(x_train, y_train, x_val, y_val, report, preprocessor=None,
                    tracking_uri=None, experiment_name="PBI06_HQ", member="Han Qian"):
    """Train baseline models from training and validation inputs only."""
    return train_prepared(x_train, y_train, x_val, y_val, report, preprocessor,
                          tracking_uri, experiment_name, member)


def train_prepared(x_train, y_train, x_val, y_val, report, preprocessor=None,
                   tracking_uri=None, experiment_name="PBI06_HQ", member="Han Qian"):
    """Train the seven baselines on PBI 03 training and validation data.

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
            needs_dense = name in ("hist_gradient_boosting", "gradient_boosting", "catboost")
            train_input = train.toarray() if needs_dense and hasattr(train, "toarray") else train
            val_input = val.toarray() if needs_dense and hasattr(val, "toarray") else val
            start = perf_counter()
            if name in ("hist_gradient_boosting", "gradient_boosting", "xgboost", "lightgbm", "catboost"):
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


def train_shared_feature_comparison(x_train, y_train, train_dates, x_val, y_val,
                                    report, selected_models, tracking_uri=None,
                                    member="Han Qian"):
    """Fit fixed selected configurations with shared 8-feature prep and cutoff rule.

    The newest 20% of whole outer-training dates is reserved for threshold
    calibration. Each model is fit once on older training rows; that same fitted
    model produces calibration and validation probabilities. No search or test
    data are used. ``selected_models`` maps model names to saved params and
    training-only weight strength from prior PBI 06 evidence.
    """
    import pandas as pd

    from src.feature_engineering import INPUT_FEATURES

    if list(x_train.columns) != INPUT_FEATURES or list(x_val.columns) != INPUT_FEATURES:
        raise ValueError("Shared comparison requires the exact PBI 03 eight-feature schema")
    if not x_train.index.equals(y_train.index) or not x_train.index.equals(train_dates.index):
        raise ValueError("Training features, labels and dates must have matching indices")
    if not x_val.index.equals(y_val.index) or not x_train.index.intersection(x_val.index).empty:
        raise ValueError("Training and validation rows must have matching, disjoint indices")
    if list(x_train.columns) != list(x_val.columns):
        raise ValueError("Training and validation features must have the same order")
    if not set(selected_models).issubset(model_candidates()):
        raise ValueError("A saved configuration names an unsupported model")

    calibration_mask = recent_training_window(train_dates, share=0.20)
    fit_mask = ~calibration_mask
    fit_y = y_train.loc[fit_mask]
    calibration_y = y_train.loc[calibration_mask]
    if set(fit_y.unique()) != {0, 1} or set(calibration_y.unique()) != {0, 1}:
        raise ValueError("Training and calibration periods must each contain both classes")

    preprocessor = fit_preprocessor(x_train.loc[fit_mask])
    fit_matrix = preprocessor.transform(x_train.loc[fit_mask]).astype(np.float32)
    calibration_matrix = preprocessor.transform(x_train.loc[calibration_mask]).astype(np.float32)
    validation_matrix = preprocessor.transform(x_val).astype(np.float32)
    uri = _tracking("PBI06_SharedEightFeature", tracking_uri)
    rows = []

    with threadpool_limits(limits=4):
        for name, config in selected_models.items():
            print(f"Training shared eight-feature {name}...", flush=True)
            model = _seven_model(name, config["params"], seed=42)
            fit_input = _model_input(name, fit_matrix)
            calibration_input = _model_input(name, calibration_matrix)
            validation_input = _model_input(name, validation_matrix)
            start = perf_counter()
            _fit_weighted(model, name, fit_input, fit_y, config["weight_strength"])
            training_seconds = perf_counter() - start
            calibration_probabilities = model.predict_proba(calibration_input)[:, 1]
            threshold = best_f1_threshold(calibration_y, calibration_probabilities)
            calibration_f1 = f1_score(
                calibration_y, calibration_probabilities >= threshold, zero_division=0
            )
            start = perf_counter()
            probabilities = model.predict_proba(validation_input)[:, 1]
            prediction_seconds = perf_counter() - start
            metrics_at_half = validation_metrics(y_val, probabilities, 0.5)
            metrics = validation_metrics(y_val, probabilities, threshold)
            run_id = _log_run(
                name, "shared_eight_feature", model, preprocessor, y_val, probabilities,
                threshold, metrics, report, training_seconds, prediction_seconds,
                member, f"training-only balanced weight strength {config['weight_strength']}",
                extra_params={
                    "selected_parameters": config["params"],
                    "selected_config_source": config.get("source_run_id", "saved PBI06 configuration"),
                    "feature_schema_version": "shared_8_feature_v1",
                    "threshold_rule": "max fraud F1 on newest 20% of whole training dates",
                    "calibration_f1": float(calibration_f1),
                    "calibration_rows": int(calibration_mask.sum()),
                    "fit_rows": int(fit_mask.sum()),
                },
                run_label=f"{member}-{name}-shared-eight-feature",
                split_description="chronological training/calibration/validation; test untouched",
            )
            row = {
                "model": name, "version": "shared_eight_feature",
                "threshold": threshold, "calibration_f1": float(calibration_f1),
                "f1_at_0.5": metrics_at_half["f1"], **metrics,
                "training_seconds": training_seconds,
                "prediction_seconds": prediction_seconds,
                "fit_rows": int(fit_mask.sum()),
                "calibration_rows": int(calibration_mask.sum()),
                "selected_parameters": config["params"],
                "source_run_id": config.get("source_run_id"),
                "run_id": run_id,
            }
            rows.append(row)
            print(f"Finished {name}: validation fraud F1 = {metrics['f1']:.4f}", flush=True)

    leaderboard = pd.DataFrame(rows).sort_values(
        ["f1", "average_precision"], ascending=[False, False]
    ).reset_index(drop=True)
    return {"results": rows, "leaderboard": leaderboard, "tracking_uri": uri,
            "calibration_rule": "latest 20% of whole outer-training dates"}


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
ADVANCED_REFERENCE = Path("artifacts/ca1_advanced_reference.json")


def save_advanced_reference(results):
    """Remember selected run IDs without reading or hashing raw source files."""
    import json

    reference = {"runs": {row["model"]: row["run_id"] for row in results}}
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
        reference = {"runs": SAVED_ADVANCED_RUNS}
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
    threshold = best_f1_threshold(y, probabilities)
    return threshold, float(f1_score(y, np.asarray(probabilities) >= threshold, zero_division=0))


def seven_model_candidates(seed=42):
    """Return three reproducible tuning configurations for every model family."""
    return {
        "logistic_regression": [
            ({"C": 0.1, "max_iter": 500}, 0.5),
            ({"C": 1.0, "max_iter": 500}, 0.75),
            ({"C": 10.0, "max_iter": 500}, 1.0),
        ],
        "random_forest": [
            ({"n_estimators": 60, "max_depth": 12, "min_samples_leaf": 30, "max_features": 0.8}, 0.5),
            ({"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 50, "max_features": 1.0}, 0.75),
            ({"n_estimators": 80, "max_depth": None, "min_samples_leaf": 40, "max_features": 0.8}, 1.0),
        ],
        "xgboost": [
            ({"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05, "reg_lambda": 2,
              "subsample": 0.8, "colsample_bytree": 0.8}, 0.5),
            ({"n_estimators": 120, "max_depth": 5, "learning_rate": 0.08, "reg_lambda": 5,
              "subsample": 0.9, "colsample_bytree": 0.8}, 0.75),
            ({"n_estimators": 160, "max_depth": 3, "learning_rate": 0.04, "reg_lambda": 10,
              "subsample": 0.8, "colsample_bytree": 1.0}, 1.0),
        ],
        "lightgbm": [
            ({"n_estimators": 80, "num_leaves": 15, "learning_rate": 0.05, "reg_lambda": 2,
              "min_child_samples": 30, "subsample": 0.8, "subsample_freq": 1,
              "colsample_bytree": 0.8}, 0.5),
            ({"n_estimators": 120, "num_leaves": 23, "learning_rate": 0.08, "reg_lambda": 5,
              "min_child_samples": 50, "subsample": 0.9, "subsample_freq": 1,
              "colsample_bytree": 0.8}, 0.75),
            ({"n_estimators": 160, "num_leaves": 31, "learning_rate": 0.04, "reg_lambda": 10,
              "min_child_samples": 80, "subsample": 0.8, "subsample_freq": 1,
              "colsample_bytree": 1.0}, 1.0),
        ],
        "catboost": [
            ({"iterations": 80, "depth": 4, "learning_rate": 0.08, "l2_leaf_reg": 3,
              "subsample": 0.8, "bootstrap_type": "Bernoulli"}, 0.5),
            ({"iterations": 120, "depth": 5, "learning_rate": 0.06, "l2_leaf_reg": 5,
              "subsample": 0.9, "bootstrap_type": "Bernoulli"}, 0.75),
            ({"iterations": 160, "depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 10,
              "subsample": 0.8, "bootstrap_type": "Bernoulli"}, 1.0),
        ],
        "hist_gradient_boosting": [
            ({"max_iter": 80, "max_leaf_nodes": 15, "learning_rate": 0.05,
              "l2_regularization": 1.0}, 0.5),
            ({"max_iter": 120, "max_leaf_nodes": 31, "learning_rate": 0.05,
              "l2_regularization": 5.0}, 0.75),
            ({"max_iter": 160, "max_leaf_nodes": 31, "learning_rate": 0.08,
              "l2_regularization": 10.0}, 1.0),
        ],
        "gradient_boosting": [
            ({"n_estimators": 80, "max_depth": 2, "learning_rate": 0.05,
              "subsample": 0.8, "min_samples_leaf": 20}, 0.5),
            ({"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
              "subsample": 0.8, "min_samples_leaf": 40}, 0.75),
            ({"n_estimators": 120, "max_depth": 2, "learning_rate": 0.1,
              "subsample": 1.0, "min_samples_leaf": 20}, 1.0),
        ],
    }


def _seven_model(name, params, seed):
    """Build one tuned candidate with class weighting handled by sample weights."""
    model = model_candidates(seed)[name]
    if name in ("logistic_regression", "random_forest"):
        model.set_params(class_weight=None)
    current_params = model.get_params()
    clean_params = {}
    for key, value in params.items():
        current_value = current_params.get(key)
        if isinstance(value, str) and not isinstance(current_value, str):
            if value == "None":
                value = None
            elif isinstance(current_value, bool):
                value = value.lower() == "true"
            elif isinstance(current_value, int):
                value = int(value)
            elif isinstance(current_value, float):
                value = float(value)
        clean_params[key] = value
    return model.set_params(**clean_params)


def _model_input(name, matrix):
    if name in ("catboost", "hist_gradient_boosting", "gradient_boosting") and hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def _validate_training_inputs(x_train, y_train, train_dates, x_val, y_val, report):
    import pandas as pd

    if not x_train.index.is_unique or not x_val.index.is_unique:
        raise ValueError("Training and validation row indices must be unique")
    if not x_train.index.equals(y_train.index) or not x_train.index.equals(train_dates.index):
        raise ValueError("Training features, labels and dates must have matching indices")
    if not x_val.index.equals(y_val.index):
        raise ValueError("Validation features and labels must have matching indices")
    if not x_train.index.intersection(x_val.index).empty:
        raise ValueError("Training and validation rows overlap")
    if list(x_train.columns) != list(x_val.columns):
        raise ValueError("Training and validation feature schemas differ")
    if set(y_train.unique()) != {0, 1} or set(y_val.unique()) != {0, 1}:
        raise ValueError("Training and validation labels must each contain both classes")
    dates = pd.to_datetime(train_dates, errors="raise")
    if dates.isna().any() or dates.max() >= pd.Timestamp(report["validation_first_date"]):
        raise ValueError("Training dates must be valid and precede validation")
    if len(x_train) != report["training_rows"] or len(x_val) != report["validation_rows"]:
        raise ValueError("Split report row counts do not match the supplied data")


def run_seven_model_experiments(x_train, y_train, train_dates, x_val, y_val, report,
                                tracking_uri=None, experiment_name="PBI06_FinalOptimization",
                                member="Han Qian", seed=42):
    """Run baseline, train-only tuning and OOF threshold selection for all seven models.

    The outer validation partition is used only for the final comparison. Search
    uses three configurations per model and two expanding whole-date folds.
    Every fold fits its preprocessor and class weights on that fold's training
    rows. The selected threshold uses the selected candidate's pooled OOF scores;
    because those folds also selected hyperparameters, its training F1 is
    optimistic. No test data is accepted by this interface.
    """
    import pandas as pd

    _validate_training_inputs(x_train, y_train, train_dates, x_val, y_val, report)
    uri = _tracking(experiment_name, tracking_uri)
    search_id = uuid4().hex
    candidates = seven_model_candidates(seed)
    model_names = list(candidates)
    folds = chronological_folds(x_train, train_dates)
    trial_rows = []
    candidate_predictions = {}
    model_search_seconds = {name: 0.0 for name in model_names}
    all_start = perf_counter()

    print("Training seven baseline models at threshold 0.5...", flush=True)
    baseline_preprocessor = fit_preprocessor(x_train)
    baseline_train_matrix = baseline_preprocessor.transform(x_train).astype(np.float32)
    baseline_val_matrix = baseline_preprocessor.transform(x_val).astype(np.float32)
    baseline_results = []
    with threadpool_limits(limits=4):
        for name in model_names:
            model = model_candidates(seed)[name]
            train_matrix = _model_input(name, baseline_train_matrix)
            val_matrix = _model_input(name, baseline_val_matrix)
            start = perf_counter()
            if name in ("logistic_regression", "random_forest"):
                model.fit(train_matrix, y_train)
                weighting = "balanced class weights"
            else:
                model.fit(train_matrix, y_train, sample_weight=training_sample_weights(y_train))
                weighting = "balanced training sample weights"
            fit_seconds = perf_counter() - start
            start = perf_counter()
            probabilities = model.predict_proba(val_matrix)[:, 1]
            predict_seconds = perf_counter() - start
            metrics = validation_metrics(y_val, probabilities, 0.5)
            run_id = _log_run(name, "final_baseline", model, baseline_preprocessor, y_val,
                              probabilities, 0.5, metrics, report, fit_seconds, predict_seconds,
                              member, weighting, extra_params={"seed": seed, "search": "baseline"},
                              run_label=f"{member}-{name}-final-baseline",
                              run_tags={"search_id": search_id})
            baseline_results.append({"model": name, "version": "baseline", "threshold": 0.5,
                                    **metrics, "training_seconds": fit_seconds,
                                    "prediction_seconds": predict_seconds,
                                    "parameters": model.get_params(), "run_id": run_id})
            print(f"Baseline {name}: validation F1={metrics['f1']:.4f}", flush=True)

    print("Tuning seven models on two chronological training-only folds...", flush=True)
    search_start = perf_counter()
    for fold_number, (train_mask, valid_mask) in enumerate(folds, start=1):
        fold_x = x_train.loc[train_mask]
        fold_y = y_train.loc[train_mask]
        fold_val_x = x_train.loc[valid_mask]
        fold_val_y = y_train.loc[valid_mask]
        if set(fold_y.unique()) != {0, 1} or set(fold_val_y.unique()) != {0, 1}:
            raise ValueError("Each chronological fold must contain both fraud classes")
        fold_preprocessor = fit_preprocessor(fold_x)
        train_matrix = fold_preprocessor.transform(fold_x).astype(np.float32)
        val_matrix = fold_preprocessor.transform(fold_val_x).astype(np.float32)
        weights = training_sample_weights(fold_y)
        fold_dates = {
            "inner_train_first": str(train_dates.loc[train_mask].min().date()),
            "inner_train_last": str(train_dates.loc[train_mask].max().date()),
            "inner_val_first": str(train_dates.loc[valid_mask].min().date()),
            "inner_val_last": str(train_dates.loc[valid_mask].max().date()),
        }
        for name in model_names:
            for candidate_number, (params, weight_strength) in enumerate(candidates[name], start=1):
                print(f"Fold {fold_number}: {name}, configuration {candidate_number}/3", flush=True)
                model = _seven_model(name, params, seed)
                fit_start = perf_counter()
                model.fit(_model_input(name, train_matrix), fold_y,
                          sample_weight=1 + weight_strength * (weights - 1))
                fit_seconds = perf_counter() - fit_start
                probabilities = model.predict_proba(_model_input(name, val_matrix))[:, 1]
                metrics = validation_metrics(fold_val_y, probabilities, 0.5)
                with mlflow.start_run(run_name=f"{member}-{name}-final-cv{fold_number}-c{candidate_number}") as run:
                    mlflow.set_tags({"member": member, "model_name": name,
                                     "model_version": "final_tuning_trial",
                                     "dataset": "training_internal_validation",
                                     "search_id": search_id})
                    mlflow.log_params({"candidate": candidate_number, "fold": fold_number,
                                       "threshold": 0.5, "weight_strength": weight_strength,
                                       "seed": seed, "split": "chronological whole-date fold",
                                       "train_rows": len(fold_x), "validation_rows": len(fold_val_x),
                                       **fold_dates, **{"model__" + key: str(value)
                                                       for key, value in model.get_params().items()}})
                    mlflow.log_metrics({**metrics, "training_seconds": fit_seconds})
                    mlflow.log_dict(params, "selected_candidate_parameters.json")
                    trial_id = run.info.run_id
                trial_rows.append({"model": name, "candidate": candidate_number,
                                   "fold": fold_number, "inner_f1": metrics["f1"],
                                   "parameters": params, "weight_strength": weight_strength,
                                   "training_seconds": fit_seconds, "run_id": trial_id})
                model_search_seconds[name] += fit_seconds
                candidate_predictions.setdefault((name, candidate_number), []).append(
                    (fold_val_y.to_numpy(), probabilities)
                )

    search_wall_seconds = perf_counter() - search_start

    trial_table = pd.DataFrame(trial_rows)
    mean_scores = trial_table.groupby(["model", "candidate"], as_index=False)["inner_f1"].mean()
    best_rows = mean_scores.sort_values(["model", "inner_f1", "candidate"],
                                        ascending=[True, False, True]).drop_duplicates("model")
    selected = {row.model: int(row.candidate) for row in best_rows.itertuples()}

    tuned_preprocessor = fit_preprocessor(x_train)
    tuned_train = tuned_preprocessor.transform(x_train).astype(np.float32)
    tuned_val = tuned_preprocessor.transform(x_val).astype(np.float32)
    tuned_results = []
    threshold_results = []
    for name in model_names:
        candidate_number = selected[name]
        params, weight_strength = candidates[name][candidate_number - 1]
        choice = best_rows.loc[best_rows["model"] == name].iloc[0]
        oof = candidate_predictions[(name, candidate_number)]
        oof_y = np.concatenate([part[0] for part in oof])
        oof_scores = np.concatenate([part[1] for part in oof])
        threshold_start = perf_counter()
        threshold, oof_f1 = select_f1_threshold(oof_y, oof_scores)
        threshold_seconds = perf_counter() - threshold_start

        print(f"Refitting selected {name} configuration {candidate_number} on training data...", flush=True)
        model = _seven_model(name, params, seed)
        train_matrix = _model_input(name, tuned_train)
        val_matrix = _model_input(name, tuned_val)
        weights = training_sample_weights(y_train)
        fit_start = perf_counter()
        model.fit(train_matrix, y_train, sample_weight=1 + weight_strength * (weights - 1))
        fit_seconds = perf_counter() - fit_start
        predict_start = perf_counter()
        probabilities = model.predict_proba(val_matrix)[:, 1]
        predict_seconds = perf_counter() - predict_start

        common = {"selected_candidate": candidate_number,
                  "inner_mean_f1": float(choice["inner_f1"]),
                  "inner_oof_f1": float(oof_f1), "inner_oof_rows": len(oof_y),
                  "selected_parameters": params, "weight_strength": weight_strength,
                  "training_seconds": fit_seconds, "prediction_seconds": predict_seconds,
                  "search_seconds": model_search_seconds[name], "threshold_seconds": threshold_seconds}
        for version, cutoff, extra in [
            ("final_tuned_0.5", 0.5, {"threshold_selection": "fixed 0.5"}),
            ("final_threshold_optimized", threshold,
             {"threshold_selection": "selected candidate pooled chronological training OOF"}),
        ]:
            metrics = validation_metrics(y_val, probabilities, cutoff)
            run_id = _log_run(name, version, model, tuned_preprocessor, y_val,
                              probabilities, cutoff, metrics, report, fit_seconds,
                              predict_seconds, member,
                              f"training-only balanced weight strength {weight_strength}",
                              extra_params={**extra, "selected_candidate": candidate_number,
                                            "inner_mean_f1": float(choice["inner_f1"]),
                                            "inner_oof_f1": float(oof_f1),
                                            "inner_oof_rows": len(oof_y), "seed": seed,
                                            "weight_strength": weight_strength,
                                            "search_budget_per_model": len(candidates[name]),
                                            "search_wall_seconds": search_wall_seconds,
                                            "search_folds": len(folds),
                                            **{"selected__" + key: value for key, value in params.items()}},
                              run_label=f"{member}-{name}-{version}",
                              run_tags={"search_id": search_id})
            row = {"model": name, "version": version, "threshold": cutoff,
                   **metrics, **common, "run_id": run_id}
            (tuned_results if version == "final_tuned_0.5" else threshold_results).append(row)

    total_seconds = perf_counter() - all_start
    client = mlflow.MlflowClient()
    for result in tuned_results + threshold_results:
        client.log_metric(result["run_id"], "complete_experiment_seconds", total_seconds)

    baseline_by_name = {row["model"]: row for row in baseline_results}
    tuned_by_name = {row["model"]: row for row in tuned_results}
    optimized_by_name = {row["model"]: row for row in threshold_results}
    summary_rows = []
    for name in model_names:
        baseline = baseline_by_name[name]
        tuned = tuned_by_name[name]
        optimized = optimized_by_name[name]
        summary_rows.append({
            "model": name,
            "baseline_f1": baseline["f1"],
            "tuned_f1_at_0.5": tuned["f1"],
            "optimized_f1": optimized["f1"],
            "selected_parameters": tuned["selected_parameters"],
            "threshold": optimized["threshold"],
            "precision": optimized["precision"], "recall": optimized["recall"],
            "fp": optimized["fp"], "fn": optimized["fn"],
            "average_precision": optimized["average_precision"],
            "roc_auc": optimized["roc_auc"],
            "accuracy": optimized["accuracy"],
            "training_seconds": tuned["training_seconds"],
            "search_seconds": tuned["search_seconds"],
            "prediction_seconds": tuned["prediction_seconds"],
            "baseline_run_id": baseline["run_id"],
            "tuned_run_id": tuned["run_id"],
            "threshold_run_id": optimized["run_id"],
        })
    leaderboard = pd.DataFrame(summary_rows).sort_values(
        ["optimized_f1", "average_precision"], ascending=[False, False]
    ).reset_index(drop=True)
    return {
        "baseline_results": rank_results(baseline_results),
        "tuned_results": rank_results(tuned_results),
        "threshold_results": rank_results(threshold_results),
        "trials": trial_table,
        "leaderboard": leaderboard,
        "search_seconds": float(sum(model_search_seconds.values())),
        "search_wall_seconds": search_wall_seconds,
        "total_seconds": total_seconds,
        "tracking_uri": uri,
        "split": report,
    }


def load_seven_model_experiments(report, tracking_uri=None,
                                experiment_name="PBI06_FinalOptimization"):
    """Load the latest complete seven-model experiment and validate its artifacts."""
    import pandas as pd

    uri = _tracking(experiment_name, tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs([experiment.experiment_id], max_results=50000,
                              order_by=["start_time DESC"])
    versions = {"final_baseline", "final_tuning_trial", "final_tuned_0.5",
                "final_threshold_optimized"}
    grouped = {}
    for run in runs:
        version = run.data.tags.get("model_version")
        search_id = run.data.tags.get("search_id")
        if run.info.status == "FINISHED" and version in versions and search_id:
            grouped.setdefault(search_id, []).append(run)

    model_names = set(seven_model_candidates())
    required_versions = {"final_baseline", "final_tuned_0.5", "final_threshold_optimized"}
    complete = []
    for search_id, group in grouped.items():
        version_models = {}
        for run in group:
            version_models.setdefault(run.data.tags["model_version"], set()).add(
                run.data.tags.get("model_name")
            )
        trials = [run for run in group if run.data.tags["model_version"] == "final_tuning_trial"]
        trial_keys = {(run.data.tags.get("model_name"), int(run.data.params["candidate"]),
                       int(run.data.params["fold"])) for run in trials}
        expected_trials = {(name, candidate, fold) for name in model_names
                           for candidate in (1, 2, 3) for fold in (1, 2)}
        if all(version_models.get(version) == model_names for version in required_versions) \
                and len(trials) == 42 and trial_keys == expected_trials:
            complete.append((max(run.info.start_time for run in group), search_id, group))
    if not complete:
        raise RuntimeError("No complete seven-model optimization evidence is saved in local MLflow")
    _, search_id, group = max(complete)
    grouped_runs = {}
    for run in group:
        key = (run.data.tags["model_version"], run.data.tags.get("model_name"))
        if key[0] != "final_tuning_trial":
            grouped_runs[key] = run

    expected_split = {
        "train_first_date": report["train_first_date"],
        "train_last_date": report["train_last_date"],
        "validation_first_date": report["validation_first_date"],
        "validation_last_date": report["validation_last_date"],
        "train_rows": report["training_rows"],
        "validation_rows": report["validation_rows"],
    }

    result_groups = {"baseline_results": [], "tuned_results": [], "threshold_results": []}
    candidate_fit_seconds = {name: sum(
        run.data.metrics["training_seconds"] for run in group
        if run.data.tags["model_version"] == "final_tuning_trial"
        and run.data.tags.get("model_name") == name
    ) for name in model_names}
    search_wall_seconds = float(next(
        run.data.params["search_wall_seconds"] for run in group
        if run.data.tags["model_version"] == "final_tuned_0.5"
    ))
    for name in sorted(model_names):
        for version, result_key in [
            ("final_baseline", "baseline_results"),
            ("final_tuned_0.5", "tuned_results"),
            ("final_threshold_optimized", "threshold_results"),
        ]:
            run = grouped_runs[(version, name)]
            for key, value in expected_split.items():
                if str(run.data.params.get(key)) != str(value):
                    raise RuntimeError("Saved seven-model results do not match the current development split")
            artifact_paths = {item.path.rsplit("/", 1)[-1]
                              for item in client.list_artifacts(run.info.run_id, "model")}
            required_artifacts = {"model.joblib", "preprocessor.joblib", "inference.json"}
            if not required_artifacts.issubset(artifact_paths):
                raise RuntimeError(f"Required model artifacts are missing for MLflow run {run.info.run_id}: {sorted(artifact_paths)}")
            for path in ("precision_recall.png", "confusion_matrix.png", "model_parameters.json"):
                client.download_artifacts(run.info.run_id, path)
            inference_path = client.download_artifacts(run.info.run_id, "model/inference.json")
            import json
            inference = json.loads(Path(inference_path).read_text(encoding="utf-8"))
            threshold = float(run.data.params["threshold"])
            if float(inference["threshold"]) != threshold:
                raise RuntimeError(f"Saved threshold does not match the inference artifact for run {run.info.run_id}")
            metric_names = ("f1", "precision", "recall", "average_precision", "roc_auc",
                            "accuracy", "false_positive_rate", "tn", "fp", "fn", "tp",
                            "training_seconds", "prediction_seconds")
            row = {"model": name, "version": version, "threshold": threshold,
                   **{key: run.data.metrics[key] for key in metric_names},
                   "run_id": run.info.run_id, "search_id": search_id}
            for key in ("tn", "fp", "fn", "tp"):
                row[key] = int(row[key])
            if version != "final_baseline":
                row["selected_candidate"] = int(run.data.params["selected_candidate"])
                row["inner_mean_f1"] = float(run.data.params["inner_mean_f1"])
                row["inner_oof_f1"] = float(run.data.params["inner_oof_f1"])
                row["inner_oof_rows"] = int(run.data.params["inner_oof_rows"])
                row["weight_strength"] = float(run.data.params["weight_strength"])
                row["selected_parameters"] = {
                    key[len("selected__"):]: value
                    for key, value in run.data.params.items() if key.startswith("selected__")
                }
                row["search_wall_seconds"] = float(run.data.params["search_wall_seconds"])
            result_groups[result_key].append(row)

    trial_rows = []
    for run in group:
        if run.data.tags["model_version"] != "final_tuning_trial":
            continue
        params = {key[len("model__"):]: value for key, value in run.data.params.items()
                  if key.startswith("model__")}
        trial_rows.append({"model": run.data.tags["model_name"],
                           "candidate": int(run.data.params["candidate"]),
                           "fold": int(run.data.params["fold"]),
                           "inner_f1": run.data.metrics["f1"],
                           "training_seconds": run.data.metrics["training_seconds"],
                           "parameters": params,
                           "weight_strength": float(run.data.params["weight_strength"]),
                           "run_id": run.info.run_id})
    trials = pd.DataFrame(trial_rows).sort_values(["model", "candidate", "fold"])
    baseline_by_name = {row["model"]: row for row in result_groups["baseline_results"]}
    tuned_by_name = {row["model"]: row for row in result_groups["tuned_results"]}
    optimized_by_name = {row["model"]: row for row in result_groups["threshold_results"]}
    leaderboard = []
    for name in model_names:
        baseline, tuned, optimized = (baseline_by_name[name], tuned_by_name[name], optimized_by_name[name])
        leaderboard.append({
            "model": name, "baseline_f1": baseline["f1"],
            "tuned_f1_at_0.5": tuned["f1"], "optimized_f1": optimized["f1"],
            "selected_parameters": tuned["selected_parameters"],
            "threshold": optimized["threshold"], "precision": optimized["precision"],
            "recall": optimized["recall"], "fp": optimized["fp"], "fn": optimized["fn"],
            "average_precision": optimized["average_precision"],
            "roc_auc": optimized["roc_auc"], "accuracy": optimized["accuracy"],
            "training_seconds": tuned["training_seconds"],
            "search_seconds": candidate_fit_seconds[name],
            "prediction_seconds": tuned["prediction_seconds"],
            "baseline_run_id": baseline["run_id"], "tuned_run_id": tuned["run_id"],
            "threshold_run_id": optimized["run_id"],
        })
    leaderboard = pd.DataFrame(leaderboard).sort_values(
        ["optimized_f1", "average_precision"], ascending=[False, False]
    ).reset_index(drop=True)
    return {**result_groups, "trials": trials, "leaderboard": leaderboard,
            "search_id": search_id, "tracking_uri": uri,
            "search_seconds": sum(candidate_fit_seconds.values()),
            "search_wall_seconds": search_wall_seconds}


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
                          experiment_name="PBI06_HQ", member="Han Qian", batch_name="retrain",
                          *, configurations=None):
    """Refit fixed CatBoost and LightGBM on prepared training data; never search.

    Inputs are unencoded PBI 03 feature frames and binary label Series. The
    report must describe their chronological split. Returns per-model metrics,
    MLflow run IDs and model/preprocessor/inference artifact references.
    No test data or historical model artifact is required. Pass approved
    rebuilt configurations as {family: {"params": ..., "weight": ...,
    "threshold": ...}}. Their thresholds are applied as fixed policy values.
    Refit scores can drift, so evaluate the fixed policy on supplied validation.
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
    selected = SELECTED_MODELS if configurations is None else configurations
    if not selected or not set(selected).issubset(model_candidates()):
        raise ValueError("Supply known fixed model configurations")
    with threadpool_limits(limits=4):
        for name, config in selected.items():
            if configurations is not None:
                from src.pbi06_research import _fit, _matrix
                if not 0 <= config["threshold"] <= 1 or not 0 <= config["weight"] <= 1:
                    raise ValueError("Fixed threshold and weight must be between zero and one")
                model, training_seconds = _fit(name, config, train, y_train)
                val_input = _matrix(name, val)
                weight_strength = config["weight"]
            else:
                model = model_candidates()[name].set_params(**config["params"])
                train_input = train.toarray() if name == "catboost" and hasattr(train, "toarray") else train
                val_input = val.toarray() if name == "catboost" and hasattr(val, "toarray") else val
                start = perf_counter()
                _fit_weighted(model, name, train_input, y_train, config["weight_strength"])
                training_seconds = perf_counter() - start
                weight_strength = config["weight_strength"]
            start = perf_counter()
            probabilities = model.predict_proba(val_input)[:, 1]
            prediction_seconds = perf_counter() - start
            metrics = validation_metrics(y_val, probabilities, config["threshold"])
            run_id = _log_run(name, "selected_retrain", model, preprocessor, y_val, probabilities,
                              config["threshold"], metrics, split_report, training_seconds,
                              prediction_seconds, member, f"training-only balanced weight strength {weight_strength}",
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


# The notebook and programmatic research callers share the same staged workflow.
from src.pbi06_research import evaluation_dates, research_stages, threshold_curve
