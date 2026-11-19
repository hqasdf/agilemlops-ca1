"""Canonical seven-family PBI 06 research workflow and verified local evidence."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from copy import copy
from tempfile import TemporaryDirectory

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from src.data_access import SEALED_START, VALIDATION_START
from src.feature_engineering import INPUT_FEATURES, fit_preprocessor, training_sample_weights
from src.model_evaluation import best_f1_threshold, recent_training_window
from src.pbi06_tracking import (LOCAL_URI, COMPLETED_RUN, COMPLETED_FINGERPRINT,
                                OLD_SOURCES, TRACKING_PATCH_SOURCES, resolve_tracking_uri, EXPERIMENT)

SEED = 42
BUDGETS = (10, 24)

# Values are sampled jointly. Conditional spaces below enforce compatible settings.
ORDINARY = {
    "logistic_regression": dict(C=[.03, .2, 1, 5, 20], max_iter=[200, 350]),
    "random_forest": dict(n_estimators=[25, 45, 65], max_depth=[8, 12, 16], min_samples_leaf=[25, 65]),
    "xgboost": dict(n_estimators=[65, 100, 145], max_depth=[3, 5, 7], learning_rate=[.04, .08, .13]),
    "lightgbm": dict(n_estimators=[70, 120, 170], num_leaves=[12, 24, 40], learning_rate=[.035, .07, .12]),
    "catboost": dict(iterations=[75, 115, 155], depth=[4, 5, 7], learning_rate=[.04, .08, .13]),
    "hist_gradient_boosting": dict(max_iter=[70, 110, 150], max_leaf_nodes=[12, 24, 40], learning_rate=[.04, .08, .13]),
    "gradient_boosting": dict(n_estimators=[45, 65, 90], max_depth=[2, 3, 4], learning_rate=[.045, .09, .15]),
}
ADVANCED = {
    "logistic_regression": dict(C=[.005, .05, .4, 2, 12, 40], l1_ratio=[0., .25, .65, 1.],
                                max_iter=[180, 350, 500], tol=[.00003, .0001, .0005]),
    "random_forest": dict(n_estimators=[30, 55, 85], max_depth=[8, 13, 19], min_samples_split=[30, 80, 160],
                          min_samples_leaf=[15, 40, 85], max_features=["sqrt", .6, 1.],
                          bootstrap=[True, False], max_samples=[.55, .8, None], criterion=["gini", "entropy"]),
    "xgboost": dict(n_estimators=[60, 110, 170], max_depth=[3, 5, 7], learning_rate=[.03, .07, .14],
                    min_child_weight=[1, 12, 45], gamma=[0, .25, 1], subsample=[.65, .82, 1.],
                    colsample_bytree=[.65, .85, 1.], reg_alpha=[0, .5, 3], reg_lambda=[1, 4, 10],
                    max_bin=[96, 192, 256]),
    "lightgbm": dict(n_estimators=[70, 120, 180], num_leaves=[12, 24, 44], max_depth=[5, 8, -1],
                     learning_rate=[.03, .065, .13], min_child_samples=[25, 65, 130],
                     min_child_weight=[.001, .1, 1.], subsample=[.65, .85, 1.],
                     colsample_bytree=[.6, .8, 1.], reg_alpha=[0, .3, 3], reg_lambda=[0, 3, 10],
                     min_split_gain=[0, .1, .5]),
    "catboost": dict(iterations=[75, 125, 180], depth=[4, 6, 8], learning_rate=[.035, .075, .13],
                     l2_leaf_reg=[2, 6, 12], random_strength=[.2, 1, 3], border_count=[48, 96, 160],
                     bootstrap_type=["Bayesian", "Bernoulli"], bagging_temperature=[.2, 1., 2.],
                     subsample=[.65, .8, .95]),
    "hist_gradient_boosting": dict(max_iter=[70, 120, 180], learning_rate=[.03, .065, .13],
                                  max_leaf_nodes=[12, 24, 48], max_depth=[4, 7, None],
                                  min_samples_leaf=[30, 80, 160], l2_regularization=[0, 2, 10],
                                  max_bins=[64, 128, 255]),
    "gradient_boosting": dict(n_estimators=[45, 75, 110], learning_rate=[.035, .075, .145],
                             max_depth=[2, 3, 4], min_samples_split=[30, 90, 180],
                             min_samples_leaf=[15, 45, 90], subsample=[.6, .8, 1.],
                             max_features=[.4, .7, None], loss=["log_loss", "exponential"]),
}


def _key(config):
    return json.dumps(config, sort_keys=True)


def sample_configs(name, stage, reference, count, seed=SEED):
    """Sample distinct configurations, retaining one exact reference."""
    rng = np.random.default_rng(seed + list(ORDINARY).index(name) + (100 if stage == "advanced" else 0))
    space = (ORDINARY if stage == "ordinary" else ADVANCED)[name]
    configs = [reference]
    seen = {_key(reference)}
    while len(configs) < count:
        params = {key: values[int(rng.integers(len(values)))] for key, values in space.items()}
        if stage == "advanced":
            if name == "logistic_regression":
                params["solver"] = "saga" if params["l1_ratio"] else ["lbfgs", "newton-cg", "saga"][int(rng.integers(3))]
            elif name == "random_forest" and not params["bootstrap"]:
                params["max_samples"] = None
            elif name == "catboost":
                params.pop("subsample" if params["bootstrap_type"] == "Bayesian" else "bagging_temperature")
            elif name == "lightgbm":
                params["subsample_freq"] = int(params["subsample"] < 1)
                if params["max_depth"] > 0:
                    params["num_leaves"] = min(params["num_leaves"], 2 ** params["max_depth"])
        config = {"params": params, "weight": [0., .35, .6, .8, 1.][int(rng.integers(5))]}
        if _key(config) not in seen:
            configs.append(config)
            seen.add(_key(config))
    return configs


def _estimator(name, config):
    from src.train_hq import model_candidates

    model = model_candidates(SEED)[name]
    settings = model.get_params()
    if "class_weight" in settings:
        model.set_params(class_weight=None)
    # Use an explicit fixed iteration count on chronological fitting data.
    if name == "hist_gradient_boosting":
        model.set_params(early_stopping=False)
    return model.set_params(**config["params"])


def _matrix(name, matrix):
    from src.train_hq import _model_input

    # XGBoost interprets implicit sparse zeros as missing. Preserve actual zeros.
    if name == "xgboost" and hasattr(matrix, "toarray"):
        return matrix.toarray()
    return _model_input(name, matrix)


def _period(dates):
    return {"first": str(dates.min().date()), "last": str(dates.max().date()), "rows": len(dates)}


def _guard(x_train, y_train, dates, x_val, y_val, report, val_dates):
    from src.train_hq import _validate_training_inputs

    train_dates = pd.to_datetime(dates, errors="raise").dt.normalize()
    val_dates = pd.to_datetime(val_dates, errors="raise").dt.normalize()
    if (train_dates >= SEALED_START).any() or (val_dates >= SEALED_START).any() or pd.Timestamp(report["validation_last_date"]) >= SEALED_START:
        raise ValueError("Sealed order dates cannot enter PBI 06")
    if list(x_train.columns) != INPUT_FEATURES or list(x_val.columns) != INPUT_FEATURES:
        raise ValueError("Expected the exact shared eight-feature schema")
    if not x_val.index.equals(val_dates.index) or val_dates.isna().any():
        raise ValueError("Evaluation dates must align with evaluation features")
    _validate_training_inputs(x_train, y_train, train_dates, x_val, y_val, report)
    if train_dates.max() >= VALIDATION_START or val_dates.min() != VALIDATION_START:
        raise ValueError("Expected the frozen chronological development split")
    for prefix, period in (("train", train_dates), ("validation", val_dates)):
        if _period(period)["first"] != report[f"{prefix}_first_date"] or _period(period)["last"] != report[f"{prefix}_last_date"]:
            raise ValueError("Date metadata does not match the split report")
    return train_dates, val_dates


def _fingerprint(frames, source_overrides=None):
    packages = {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "mlflow",
                "xgboost", "lightgbm", "catboost", "joblib", "threadpoolctl")}
    source = Path(__file__).parent
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in ("pbi06_research.py", "train_hq.py", "feature_engineering.py",
                           "data_access.py", "data_cleaning.py", "model_evaluation.py")}
    hashes.update(source_overrides or {})
    metadata = {"packages": packages, "sources": hashes, "schema": INPUT_FEATURES,
                "budgets": list(BUDGETS), "seed": SEED}
    digest = hashlib.sha256(_key(metadata).encode())
    for frame in frames:
        digest.update(str(frame.dtypes).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest(), metadata


def _fit(name, config, x, y):
    model = _estimator(name, config)
    start = perf_counter()
    weights = 1 + config["weight"] * (training_sample_weights(y) - 1)
    model.fit(_matrix(name, x), y, sample_weight=weights)
    return model, perf_counter() - start


def _probabilities(name, model, x):
    start = perf_counter()
    values = model.predict_proba(_matrix(name, x))[:, 1]
    return values, perf_counter() - start


def _winner(trials):
    """Break F1/AP ties by model size then configuration distance from baseline."""
    def order(row):
        params = row["config"]["params"]
        model_params = _estimator(row["family"], row["config"]).get_params()
        size = next((model_params[k] for k in ("n_estimators", "iterations", "max_iter") if k in model_params), 0)
        return (-row["mean_f1"], -row["mean_ap"], size, len(params), row["trial"])

    eligible = [row for row in trials if row["status"] == "ok"]
    if not eligible:
        raise RuntimeError(f"No successful configuration for {trials[0]['family']}")
    return min(eligible, key=order)


def _save_model(model, path):
    with TemporaryDirectory() as tmp:
        file = Path(tmp) / "model.joblib"
        if (mlflow.get_tracking_uri().lower().startswith(("http://", "https://"))
                and isinstance(model, RandomForestClassifier)):
            model = copy(model)
            model._sample_weight = None
        joblib.dump(model, file, compress=3)
        if mlflow.get_tracking_uri().lower().startswith(("http://", "https://")):
            from src.pbi06_publish import log_shared_artifact
            client = mlflow.MlflowClient(tracking_uri=mlflow.get_tracking_uri())
            log_shared_artifact(client, mlflow.active_run().info.run_id, f"{path}/model.joblib", file, tmp)
        else:
            mlflow.log_artifact(str(file), path)


def _save_preprocessor(preprocessor, path):
    with TemporaryDirectory() as tmp:
        file = Path(tmp) / "preprocessor.joblib"
        joblib.dump(preprocessor, file)
        mlflow.log_artifact(str(file), path)


def _trial(name, stage, number, config, folds, parent_id, fingerprint):
    from src.train_hq import validation_metrics

    row = {"family": name, "stage": stage, "trial": number, "config": config, "folds": []}
    for i, (x_fit, y_fit, x_eval, y_eval, dates_fit, dates_eval) in enumerate(folds, 1):
        with mlflow.start_run(run_name=f"{stage}-{name}-{number}-fold-{i}", nested=True) as run:
            meta = {"fit": dates_fit, "evaluation": dates_eval, "preprocessor": f"runs:/{parent_id}/folds/{i}/preprocessor.joblib"}
            mlflow.set_tags({"member": "Han Qian", "pbi": "06", "family": name, "stage": stage,
                             "trial": number, "fold": i, "model_version": stage, "dataset": fingerprint,
                             "dataset_role": "training_internal_evaluation"})
            mlflow.log_params({"seed": SEED, "threshold": .5, "weight_strength": config["weight"],
                               "schema": ",".join(INPUT_FEATURES), "preprocessing": "shared PBI 03, fold fit",
                               **{f"{part}_{k}": v for part in ("fit", "evaluation") for k, v in meta[part].items()}})
            mlflow.log_dict({"config": config, **meta}, "configuration.json")
            fold = {"fold": i, "run_id": run.info.run_id}
            started = perf_counter()
            try:
                model, seconds = _fit(name, config, x_fit, y_fit)
                scores, prediction_seconds = _probabilities(name, model, x_eval)
                metrics = validation_metrics(y_eval, scores, .5)
            except (ValueError, RuntimeError, TypeError) as error:
                fold.update(status="failed", reason=f"{type(error).__name__}: {error}",
                            training_seconds=perf_counter() - started)
                mlflow.set_tag("trial_status", "failed")
                mlflow.log_text(fold["reason"], "failure.txt")
            else:
                fold.update(status="ok", **metrics, training_seconds=seconds, prediction_seconds=prediction_seconds)
                mlflow.set_tag("trial_status", "ok")
                mlflow.log_metrics({**metrics, "training_seconds": seconds, "prediction_seconds": prediction_seconds})
                _save_model(model, "model")
            row["folds"].append(fold)
    row["status"] = "ok" if all(f["status"] == "ok" for f in row["folds"]) else "failed"
    if row["status"] == "ok":
        row.update(mean_f1=float(np.mean([f["f1"] for f in row["folds"]])),
                   std_f1=float(np.std([f["f1"] for f in row["folds"]])),
                   mean_ap=float(np.mean([f["average_precision"] for f in row["folds"]])))
    return row


def evaluation_dates(orders, labels, x_val):
    """Recover validation dates, using the same label positions as PBI 03."""
    keys = ["country_code", "order_id"]
    index = pd.MultiIndex.from_frame(labels.reset_index(drop=True).loc[x_val.index, keys])
    dates = orders.set_index(keys, verify_integrity=True)["order_date"].reindex(index)
    return pd.Series(dates.to_numpy(), index=x_val.index, name="order_date")


def _outer_row(name, stage, candidate, model, preprocessor, y_val, scores, threshold,
               fit_seconds, prediction_seconds, periods, parent_id, fingerprint, **extras):
    from src.train_hq import validation_metrics

    metrics = validation_metrics(y_val, scores, threshold)
    row = {"family": name, "stage": stage, "config": candidate["config"],
           "selection_source": candidate["stage"], "mean_f1": candidate["mean_f1"],
           "std_f1": candidate["std_f1"], "mean_ap": candidate["mean_ap"],
           "threshold": threshold, "training_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
           **metrics, **extras}
    with mlflow.start_run(run_name=f"{stage}-{name}", nested=True) as run:
        row["run_id"] = run.info.run_id
        mlflow.set_tags({"member": "Han Qian", "pbi": "06", "family": name, "stage": stage,
                         "model_version": stage, "dataset": fingerprint, "dataset_role": "outer_validation"})
        mlflow.log_params({"threshold": threshold, "seed": SEED, "schema": ",".join(INPUT_FEATURES),
                           "preprocessing": "shared PBI 03, early outer training fit",
                           "weight_strength": candidate["config"]["weight"],
                           **{f"{part}_{k}": v for part, period in periods.items() for k, v in period.items()}})
        mlflow.log_dict(model.get_params(), "parameters.json")
        mlflow.log_dict({"threshold": threshold, "family": name, "input_features": INPUT_FEATURES,
                         "transformed_features": list(preprocessor.get_feature_names_out()),
                         "dtype": "float32", "parent_run": parent_id, "metadata": row}, "model/inference.json")
        _save_model(model, "model")
        _save_preprocessor(preprocessor, "model")
        mlflow.log_metrics({**metrics, "training_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
                            "inner_mean_f1": candidate["mean_f1"], "inner_std_f1": candidate["std_f1"]})
    return row


def _prepared_folds(x, y, dates, start_log):
    from src.train_hq import chronological_folds

    result = []
    for i, (fit, evaluation) in enumerate(chronological_folds(x, dates), 1):
        preprocessor = fit_preprocessor(x.loc[fit])
        if start_log:
            _save_preprocessor(preprocessor, f"folds/{i}")
        result.append((preprocessor.transform(x.loc[fit]).astype("float32"), y.loc[fit],
                       preprocessor.transform(x.loc[evaluation]).astype("float32"), y.loc[evaluation],
                       _period(dates.loc[fit]), _period(dates.loc[evaluation])))
    return result


def _paths(client, run_id, directory):
    names = ("model.joblib", "preprocessor.joblib", "inference.json")
    if client.tracking_uri.lower().startswith(("http://", "https://")):
        from src.pbi06_publish import download_published_artifact
        return {name: download_published_artifact(client, run_id, f"model/{name}", directory)
                for name in names}
    return {name: client.download_artifacts(run_id, f"model/{name}") for name in names}


def _verify_final(client, rows, x_val, y_val, predictions):
    from src.train_hq import validation_metrics

    for row in rows:
        with TemporaryDirectory() as tmp:
            paths = _paths(client, row["run_id"], tmp)
            metadata = json.loads(Path(paths["inference.json"]).read_text())
            model = joblib.load(paths["model.joblib"])
            prep = joblib.load(paths["preprocessor.joblib"])
        name = row["family"]
        if metadata["threshold"] != row["threshold"] or metadata["input_features"] != INPUT_FEATURES:
            raise RuntimeError(f"Mismatched inference metadata: {name}")
        matrix = prep.transform(x_val).astype("float32")
        scores, _ = _probabilities(name, model, matrix)
        expected = predictions[f"final_{name}"]
        np.testing.assert_allclose(scores, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(scores >= row["threshold"], expected >= row["threshold"])
        for key, value in validation_metrics(y_val, scores, row["threshold"]).items():
            np.testing.assert_allclose([value, client.get_run(row["run_id"]).data.metrics[key]], row[key],
                                       rtol=1e-10, atol=1e-12)


def _save_evidence(evidence, predictions, uri):
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "predictions.npz"
        np.savez_compressed(path, **predictions)
        if uri.lower().startswith(("http://", "https://")):
            private = Path("artifacts/pbi06_private") / evidence["run_id"]
            private.mkdir(parents=True, exist_ok=True, mode=0o700)
            (private / path.name).write_bytes(path.read_bytes())
            evidence["private_predictions_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            mlflow.log_artifact(str(path))
    mlflow.log_dict(evidence, "evidence.json")
    mlflow.set_tag("verified_complete", "true")


def _load_evidence(client, run_id, fingerprint):
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if run_id:
        runs = [client.get_run(run_id)]
    elif experiment:
        runs = client.search_runs([experiment.experiment_id],
                                  f"tags.verified_complete = 'true' AND tags.fingerprint = '{fingerprint}'",
                                  order_by=["attributes.start_time DESC"], max_results=1)
    else:
        runs = []
    if not runs or not experiment or runs[0].info.experiment_id != experiment.experiment_id or \
            runs[0].info.status != "FINISHED" or runs[0].data.tags.get("verified_complete") != "true" or \
            runs[0].data.tags.get("fingerprint") != fingerprint:
        raise RuntimeError("No compatible verified PBI 06 evidence. Select full mode to build new evidence.")
    run_id = runs[0].info.run_id
    evidence = json.loads(Path(client.download_artifacts(run_id, "evidence.json")).read_text())
    if evidence["run_id"] != run_id or evidence["fingerprint"] != fingerprint:
        raise RuntimeError("PBI 06 evidence fingerprint does not match")
    private_hash = evidence.get("private_predictions_sha256")
    if private_hash:
        path = Path("artifacts/pbi06_private") / run_id / "predictions.npz"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != private_hash:
            raise RuntimeError("Local private predictions are missing or do not match the shared run")
    else:
        path = client.download_artifacts(run_id, "predictions.npz")
    with np.load(path) as archive:
        predictions = {key: archive[key] for key in archive.files}
    return evidence, predictions


def threshold_curve(y, scores):
    """All unique probability cutoffs and their fraud-class metrics."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return pd.DataFrame({"threshold": thresholds, "precision": precision, "recall": recall, "f1": f1})


def research_stages(x_train, y_train, dates, x_val, y_val, val_dates, report, mode="reuse", run_id=None, tracking_uri=None, evidence_uri=None):
    """Yield baseline, ordinary, advanced, selections, and verified final evidence in order."""
    from src.train_hq import _tracking, validation_metrics

    dates, val_dates = _guard(x_train, y_train, dates, x_val, y_val, report, val_dates)
    frames = (x_train, y_train, dates, x_val, y_val, val_dates)
    fingerprint, provenance = _fingerprint(frames)
    if mode == "reuse":
        client = mlflow.MlflowClient(tracking_uri=evidence_uri or LOCAL_URI)
        try:
            evidence, predictions = _load_evidence(client, run_id, fingerprint)
        except RuntimeError:
            if (run_id not in (None, COMPLETED_RUN)
                    or not TRACKING_PATCH_SOURCES
                    or any(provenance["sources"].get(name) != sha
                           for name, sha in TRACKING_PATCH_SOURCES.items())):
                raise
            old_fingerprint, old_provenance = _fingerprint(frames, OLD_SOURCES)
            if old_fingerprint != COMPLETED_FINGERPRINT:
                raise
            evidence, predictions = _load_evidence(client, COMPLETED_RUN, old_fingerprint)
            if evidence["provenance"] != old_provenance:
                raise RuntimeError("Historical PBI 06 provenance does not match")
        for stage in ("baseline", "ordinary", "advanced", "selected"):
            yield evidence[stage]
        with threadpool_limits(limits=4):
            _verify_final(client, evidence["final"], x_val, y_val, predictions)
        yield evidence, predictions
        return
    if mode != "full":
        raise ValueError("Notebook mode must be 'full' or 'reuse'")
    uri = _tracking(EXPERIMENT, resolve_tracking_uri(tracking_uri))
    client = mlflow.MlflowClient(tracking_uri=uri)
    recent = recent_training_window(dates)
    x_fit, y_fit, fit_dates = x_train.loc[~recent], y_train.loc[~recent], dates.loc[~recent]
    x_cal, y_cal, cal_dates = x_train.loc[recent], y_train.loc[recent], dates.loc[recent]
    periods = {"fit": _period(fit_dates), "calibration": _period(cal_dates), "validation": _period(val_dates)}
    distributions = {name: {"nonfraud": int(y.eq(0).sum()), "fraud": int(y.eq(1).sum())}
                     for name, y in (("training", y_train), ("fit", y_fit), ("calibration", y_cal), ("validation", y_val))}
    with mlflow.start_run(run_name="seven-family-research") as parent:
        parent_id = parent.info.run_id
        mlflow.set_tags({"pbi": "06", "member": "Han Qian", "fingerprint": fingerprint, "model_version": "rebuild"})
        mlflow.log_dict({"ordinary": ORDINARY, "advanced": ADVANCED,
                         "weights": [0., .35, .6, .8, 1.], **provenance}, "search_spaces.json")
        evidence = {"run_id": parent_id, "fingerprint": fingerprint, "provenance": provenance,
                    "periods": periods, "distributions": distributions, "split": report, "trials": [], "seconds": {}}
        predictions = {"calibration_truth": y_cal.to_numpy()}
        started = perf_counter()
        preprocessor = fit_preprocessor(x_fit)
        transformers = tuple(preprocessor.transform(frame).astype("float32") for frame in (x_fit, x_cal, x_val))
        fit_matrix, cal_matrix, val_matrix = transformers
        folds = _prepared_folds(x_fit, y_fit, fit_dates, True)
        evidence["fold_periods"] = [{"fit": f[4], "evaluation": f[5]} for f in folds]
        evidence["seconds"]["setup"] = perf_counter() - started
        histories = {name: {} for name in ORDINARY}
        candidates_by_stage = {}
        fitted = {name: {} for name in ORDINARY}
        for stage, budget in zip(("baseline", "ordinary", "advanced"), (1, *BUDGETS)):
            began = perf_counter()
            stage_rows = []
            winners = {}
            for name in ORDINARY:
                reference = winners.get(name, {"params": {}, "weight": 1.})
                if stage == "advanced":
                    reference = candidates_by_stage["ordinary"][name]["config"]
                configs = [reference] if stage == "baseline" else sample_configs(name, stage, reference, budget)
                print(f"{stage}: {name} ({len(configs)} configurations)", flush=True)
                eligible = []
                for number, config in enumerate(configs, 1):
                    key = _key(config)
                    if key not in histories[name]:
                        with threadpool_limits(limits=4):
                            original = _trial(name, stage, number, config, folds, parent_id, fingerprint)
                        histories[name][key] = original
                        reuse = False
                    else:
                        original = histories[name][key]
                        reuse = True
                    evidence["trials"].append({**original, "requested_stage": stage,
                                               "requested_trial": number, "reused": reuse})
                    eligible.append(original)
                winner = _winner(eligible)
                winners[name] = winner
                key = _key(winner["config"])
                with threadpool_limits(limits=4):
                    if key not in fitted[name]:
                        model, train_seconds = _fit(name, winner["config"], fit_matrix, y_fit)
                        val_scores, prediction_seconds = _probabilities(name, model, val_matrix)
                        fitted[name][key] = (model, train_seconds, val_scores, prediction_seconds)
                model, train_seconds, val_scores, prediction_seconds = fitted[name][key]
                predictions[f"{stage}_{name}"] = val_scores
                stage_rows.append(_outer_row(name, stage, winner, model, preprocessor, y_val, val_scores, .5,
                                             train_seconds, prediction_seconds, periods, parent_id, fingerprint))
            candidates_by_stage[stage] = winners
            evidence[stage] = stage_rows
            evidence["seconds"][stage] = perf_counter() - began
            mlflow.log_dict(stage_rows, f"stages/{stage}.json")
            yield stage_rows

        selected = [_winner([candidates_by_stage[s][name] for s in candidates_by_stage]) for name in ORDINARY]
        evidence["selected"] = [
            {**row, "highest_observed_validation_stage": max(
                (r for stage in candidates_by_stage for r in evidence[stage] if r["family"] == row["family"]),
                key=lambda r: r["f1"])["stage"]} for row in selected]
        yield evidence["selected"]

        began = perf_counter()
        final_rows = []
        for row in selected:
            name = row["family"]
            model, seconds, val_scores, prediction_seconds = fitted[name][_key(row["config"])]
            with threadpool_limits(limits=4):
                cal_scores, cal_seconds = _probabilities(name, model, cal_matrix)
            cutoff = best_f1_threshold(y_cal, cal_scores)
            halves = np.array_split(np.sort(cal_dates.unique()), 2)
            diagnostics = []
            for days in halves:
                mask = cal_dates.isin(days).to_numpy()
                if mask.any() and y_cal.iloc[mask].nunique() == 2:
                    diagnostics.append({"period": _period(cal_dates.iloc[mask]),
                                        "positive_share": float(y_cal.iloc[mask].mean()),
                                        "mean_score": float(cal_scores[mask].mean()),
                                        "f1_at_selected": validation_metrics(y_cal.iloc[mask], cal_scores[mask], cutoff)["f1"]})
            predictions[f"calibration_{name}"] = cal_scores
            predictions[f"final_{name}"] = val_scores
            final_rows.append(_outer_row(name, "optimized", row, model, preprocessor, y_val, val_scores, cutoff,
                             seconds, prediction_seconds, periods, parent_id, fingerprint,
                             fixed=validation_metrics(y_val, val_scores, .5),
                             calibration=validation_metrics(y_cal, cal_scores, cutoff),
                             calibration_fixed_f1=validation_metrics(y_cal, cal_scores, .5)["f1"],
                             calibration_prediction_seconds=cal_seconds, calibration_drift=diagnostics))
        evidence["final"] = sorted(final_rows, key=lambda row: (-row["f1"], -row["average_precision"], row["family"]))
        evidence["seconds"]["threshold_and_logging"] = perf_counter() - began
        began = perf_counter()
        with threadpool_limits(limits=4):
            _verify_final(client, evidence["final"], x_val, y_val, predictions)
        evidence["seconds"]["artifact_verification"] = perf_counter() - began
        _save_evidence(evidence, predictions, uri)
    yield evidence, predictions
