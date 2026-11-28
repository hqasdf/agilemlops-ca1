"""Log the 28 completed PBI 06 versions as ordinary MLflow runs."""
import json
from copy import copy
from pathlib import Path
from shutil import copyfile
from importlib.metadata import version
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

from src.feature_engineering import INPUT_FEATURES
from src.pbi06_tracking import EXPERIMENT, evidence_client

NAMES = dict(logistic_regression="LogisticRegression", random_forest="RandomForest",
             xgboost="XGBoost", lightgbm="LightGBM", catboost="CatBoost",
             hist_gradient_boosting="HistGradientBoosting", gradient_boosting="GradientBoosting")
STAGES = dict(baseline="Baseline", ordinary="Tuned", advanced="Advanced", final="ThresholdOptimized")
METRICS = ("f1", "precision", "recall", "average_precision", "roc_auc", "accuracy",
           "false_positive_rate", "tp", "tn", "fp", "fn", "training_seconds", "prediction_seconds")


class SavedFraudModel(mlflow.pyfunc.PythonModel):
    """Apply the saved preprocessing, estimator and decision threshold."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["estimator"])
        self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        self.config = json.loads(Path(context.artifacts["threshold"]).read_text())

    def predict(self, context, model_input):
        matrix = self.preprocessor.transform(model_input[INPUT_FEATURES]).astype("float32")
        if self.config["family"] in {"xgboost", "catboost", "hist_gradient_boosting", "gradient_boosting"}:
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
        scores = self.model.predict_proba(matrix)[:, 1]
        return pd.DataFrame({"fraud_probability": scores,
                             "is_fraud": (scores >= self.config["threshold"]).astype(int)})


def _charts(root, y_val, scores, row):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])).plot(ax=ax)
    fig.savefig(root / "confusion_matrix.png")
    plt.close(fig)
    precision, recall, _ = precision_recall_curve(y_val, scores)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision")
    fig.savefig(root / "precision_recall.png")
    plt.close(fig)


def _verify(client, experiment_id, rid, params, metrics, probe, expected):
    run = client.get_run(rid)
    if run.info.experiment_id != experiment_id or run.data.tags.get("mlflow.parentRunId"):
        raise RuntimeError(f"Run {rid} is outside the requested top-level experiment")
    if run.data.params != params or any(run.data.metrics.get(k) != v for k, v in metrics.items()):
        raise RuntimeError(f"Shared metrics or parameters differ for {rid}")
    with TemporaryDirectory() as tmp:
        folder = client.download_artifacts(rid, "model", dst_path=tmp)
        restored = mlflow.pyfunc.load_model(folder).predict(probe)
    np.testing.assert_allclose(restored["fraud_probability"], expected["fraud_probability"], rtol=1e-7, atol=1e-12)
    np.testing.assert_array_equal(restored["is_fraud"], expected["is_fraud"])
    return run


def log_completed_versions(evidence, probabilities, x_val, y_val, tracking_uri, *, limit=None):
    """Log saved results sequentially, verifying each AWS artifact before continuing."""
    from src.train_hq import validation_metrics

    destination = urlsplit(tracking_uri or "")
    if destination.scheme not in {"http", "https"} or not destination.hostname:
        raise ValueError("Direct PBI 06 logging requires an HTTP(S) MLflow tracking URI")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(EXPERIMENT)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    source = evidence_client()
    probe = x_val.iloc[::max(1, len(x_val)//128)].copy()
    rows = [(stage, row) for stage in STAGES for row in evidence[stage]]
    results = []
    for stage, row in rows[:limit]:
        family, source_id = row["family"], row["run_id"]
        name = f"{NAMES[family]}_{STAGES[stage]}"
        original = source.get_run(source_id)
        metrics = {k: float(row[k]) for k in METRICS}
        scores = probabilities[f"{stage}_{family}"]
        for key, value in validation_metrics(y_val, scores, row["threshold"]).items():
            np.testing.assert_allclose(metrics[key], value, rtol=1e-10, atol=1e-12)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = source.download_artifacts(source_id, "model/inference.json")
            parameter_path = source.download_artifacts(source_id, "parameters.json")
            config = json.loads(Path(config_path).read_text())
            effective = json.loads(Path(parameter_path).read_text())
            if config["threshold"] != row["threshold"] or config["input_features"] != INPUT_FEATURES:
                raise ValueError(f"Saved inference settings disagree for {name}")
            params = {**original.data.params, **{f"model__{k}": str(v) for k,v in effective.items()}}
            model_path = Path(source.download_artifacts(source_id, "model/model.joblib"))
            prep_path = Path(source.download_artifacts(source_id, "model/preprocessor.joblib"))
            estimator = joblib.load(model_path)
            preprocessor = joblib.load(prep_path)
            predictor = SavedFraudModel()
            predictor.model, predictor.preprocessor, predictor.config = estimator, preprocessor, config
            expected = predictor.predict(None, probe)
            positions = np.arange(0, len(x_val), max(1, len(x_val)//128))
            np.testing.assert_allclose(expected["fraud_probability"], scores[positions], rtol=1e-7, atol=1e-12)

            matches = client.search_runs([experiment.experiment_id],
                f"tags.pbi06_direct_source = '{source_id}'")
            if len(matches) > 1:
                raise RuntimeError(f"Duplicate shared runs found for {source_id}")
            match = matches[0] if matches else None
            upload_error = None
            if match and match.info.status == "FINISHED" and match.data.tags.get("inference_verified") == "true":
                _verify(client, experiment.experiment_id, match.info.run_id, params, metrics, probe, expected)
                rid = match.info.run_id
                size = int(match.data.tags["model_bytes"])
            else:
                upload_error = None
                if family == "random_forest":
                    estimator = copy(estimator)
                    estimator._sample_weight = None
                    predictor.model = estimator
                    np.testing.assert_array_equal(predictor.predict(None, probe)["is_fraud"], expected["is_fraud"])
                    model_path = root / "estimator.joblib"
                    joblib.dump(estimator, model_path, compress=3)
                with mlflow.start_run(run_id=match.info.run_id if match else None, run_name=name,
                       tags={"pbi06_direct_source": source_id, "member": "Han Qian", "pbi": "06",
                             "family": family, "stage": stage, "model_version": STAGES[stage],
                             "reused_fitted_model": "true"}) as run:
                    rid = run.info.run_id
                    mlflow.log_params(params)
                    mlflow.log_metrics({k:v for k,v in metrics.items() if not match or k not in match.data.metrics})
                    _charts(root, y_val, scores, row)
                    for filename in ("confusion_matrix.png", "precision_recall.png"):
                        mlflow.log_artifact(str(root / filename))
                    mlflow.log_dict({"selected": row["config"], "selection_source": row["selection_source"],
                                     "trials": [t for t in evidence["trials"] if t["family"] == family]
                                     if stage != "baseline" else []}, "tuning_summary.json")
                    code = root / "code_bundle" / "src"
                    code.mkdir(parents=True)
                    for module in ("pbi06_mlflow.py", "feature_engineering.py", "pbi06_tracking.py"):
                        copyfile(Path(__file__).with_name(module), code / module)
                    for attempt in range(2):
                        package = root / f"package_{attempt}" / "model"
                        package.parent.mkdir()
                        if attempt:
                            joblib.dump(estimator, model_path, compress=("xz", 9))
                            predictor.model = joblib.load(model_path)
                            compressed = predictor.predict(None, probe)
                            np.testing.assert_allclose(compressed["fraud_probability"], expected["fraud_probability"],
                                                       rtol=1e-7, atol=1e-12)
                            np.testing.assert_array_equal(compressed["is_fraud"], expected["is_fraud"])
                        mlflow.pyfunc.save_model(str(package), python_model=SavedFraudModel(),
                            artifacts={"estimator": str(model_path), "preprocessor": str(prep_path),
                                       "threshold": str(config_path)},
                            code_paths=[str(code)],
                            pip_requirements=[f"{dep}=={version(dep)}" for dep in
                                ("mlflow", "cloudpickle", "joblib", "scikit-learn", "pandas", "numpy",
                                 "matplotlib", "xgboost", "lightgbm", "catboost")])
                        size = model_path.stat().st_size
                        try:
                            mlflow.log_artifacts(str(package), "model")
                        except Exception as error:
                            if family != "random_forest" or "413" not in str(error) or not results:
                                raise RuntimeError(f"{name} ({size} bytes): {error}") from error
                            if attempt:
                                upload_error = f"HTTP 413 for the lossless XZ model ({size} bytes): {error}"
                                mlflow.set_tag("artifact_upload_error", upload_error[:1000])
                                print(f"{name}: {upload_error}", flush=True)
                                break
                            print(f"{name}: HTTP 413 at {size} bytes; trying lossless XZ", flush=True)
                        else:
                            break
                    mlflow.set_tag("model_bytes", size)
                    if not upload_error:
                        _verify(client, experiment.experiment_id, rid, params, metrics, probe, expected)
                        mlflow.set_tag("inference_verified", "true")
                if upload_error:
                    client.set_terminated(rid, status="FAILED")
            url = f"{tracking_uri.rstrip('/')}/#/experiments/{experiment.experiment_id}/runs/{rid}"
            print(f"{name}: {url} ({'artifact rejected' if upload_error else 'verified'})", flush=True)
            results.append({"name": name, "run_id": rid, "source_run_id": source_id,
                            "model_bytes": size, "url": url, "metrics": metrics, "parameters": params,
                            "artifact_error": upload_error})
            if len(results) == 1:
                visible = client.search_runs([experiment.experiment_id], "metrics.f1 >= 0",
                                             order_by=["metrics.f1 DESC"], max_results=1000)
                if not any(run.info.run_id == rid for run in visible):
                    raise RuntimeError("Pilot does not appear in the MLflow results query")
    return results
