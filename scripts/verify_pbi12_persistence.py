"""PBI 12 evidence: prove MLflow data survives a container replacement.

Run in two phases against the shared server (set MLFLOW_TRACKING_URI first):

    python -m scripts.verify_pbi12_persistence log     # before replacing the container
    # ... redeploy / replace the Elastic Beanstalk container ...
    python -m scripts.verify_pbi12_persistence check   # after: reads everything back

`log` trains a tiny scikit-learn model, logs params, metrics and the model artifact, registers a
model version, and saves the IDs to artifacts/pbi12_evidence.json. `check` loads those IDs and
retrieves the run, metrics, model artifact (and predicts with it) and the registered version.
"""
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.mlflow_remote import ensure_server_reachable, is_remote

EVIDENCE = Path("artifacts/pbi12_evidence.json")
EXPERIMENT = "pbi12-persistence-check"
MODEL_NAME = "pbi12-persistence-model"


def _connect():
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not is_remote(uri):
        sys.exit("Set MLFLOW_TRACKING_URI to the shared server URL (http://...) first.")
    ensure_server_reachable(uri)
    mlflow.set_tracking_uri(uri)
    return uri, MlflowClient()


def log_phase():
    uri, client = _connect()
    mlflow.set_experiment(EXPERIMENT)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(x, y)
    with mlflow.start_run(run_name="pbi12-real-model-before-replacement") as run:
        mlflow.log_params({"model": "logistic_regression", "rows": len(x)})
        mlflow.log_metrics({"accuracy": float(model.score(x, y))})
        info = mlflow.sklearn.log_model(model, name="model", input_example=x[:2])
        version = mlflow.register_model(info.model_uri, MODEL_NAME)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "tracking_uri": uri, "run_id": run.info.run_id, "model_uri": info.model_uri,
        "model_name": MODEL_NAME, "model_version": version.version,
        "accuracy": float(model.score(x, y)), "sample_prediction": model.predict(x[:3]).tolist(),
        "x_sample": x[:3].tolist()}, indent=2))
    print("LOGGED run", run.info.run_id, "model version", version.version, "->", EVIDENCE)


def check_phase():
    uri, client = _connect()
    if not EVIDENCE.exists():
        sys.exit(f"{EVIDENCE} missing: run the 'log' phase first.")
    saved = json.loads(EVIDENCE.read_text())
    run = client.get_run(saved["run_id"])
    files = [f.path for f in client.list_artifacts(saved["run_id"], "model")]
    version = client.get_model_version(saved["model_name"], str(saved["model_version"]))
    model = mlflow.sklearn.load_model(saved["model_uri"])
    prediction = model.predict(np.array(saved["x_sample"])).tolist()
    ok = {
        "metadata (params)": run.data.params.get("model") == "logistic_regression",
        "metrics": abs(run.data.metrics.get("accuracy", -1) - saved["accuracy"]) < 1e-9,
        "model artifact files": "model/MLmodel" in files,
        "registered model version": version.run_id == saved["run_id"],
        "model loads and predicts the same": prediction == saved["sample_prediction"],
    }
    print("server:", uri, "| run:", saved["run_id"], "| artifacts:", files)
    for name, passed in ok.items():
        print(("PASS  " if passed else "FAIL  ") + name)
    sys.exit(0 if all(ok.values()) else 1)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("log", "check"):
        sys.exit(__doc__)
    log_phase() if sys.argv[1] == "log" else check_phase()
