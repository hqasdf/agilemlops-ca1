"""Load the three PBI 07 logged models from the shared MLflow server.

Usage (from the repository root):
    $env:MLFLOW_TRACKING_URI="http://mlflow-pbi12.eba-9bnrxizu.us-east-1.elasticbeanstalk.com"
    python scripts/verify_pbi07_models.py

Exit code 0 means every model loaded. The printed output is the retrieval evidence.
"""
import os
import sys

import mlflow

MODELS = {
    "v1 logistic, fixed cut-off": "m-141667ee048f42b7973e5a5d9d5dc005",
    "v2 logistic, calibrated cut-off": "m-fbbd22521e3a4c978c77671c0ec5b2cd",
    "v3 histogram gradient boosting": "m-216bea3fda1e46b99b662e2ab7205d42",
}


def main() -> int:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("Set MLFLOW_TRACKING_URI to the shared server URL first.")
        return 2
    mlflow.set_tracking_uri(uri)
    print("Tracking server:", uri, "| mlflow", mlflow.__version__)
    failed = 0
    for label, model_id in MODELS.items():
        model_uri = f"models:/{model_id}"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            signature = model.metadata.signature
            columns = [c.name for c in signature.inputs.inputs] if signature else "no signature"
            print(f"OK   {label}: {model_uri}")
            print(f"     input columns: {columns}")
        except Exception as exc:  # report every failure, then exit non-zero
            failed += 1
            print(f"FAIL {label}: {model_uri}: {type(exc).__name__}: {exc}")
    print("All models loaded." if not failed else f"{failed} model(s) failed to load.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
