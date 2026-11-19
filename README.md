# ST1519 Agile MLOps CA1

`ca1 hq file.ipynb` is the authoritative local development notebook. Keep `customers_base.csv`, `orders_base.csv` and `labels_base.csv` alongside it; the raw CSVs and local `artifacts/` directory are ignored by Git. Run All uses `src.data_access.load_development_tables()` to load only training- and validation-eligible full rows. It reads order keys and dates to create an ignored, hashed holdout-membership manifest; it does not load held-out feature or label columns. Earlier full-history EDA examined labels before the holdout existed, so those historical findings are not test-blind.

## Start the existing project environment

From the project folder on macOS:

```bash
source .venv/bin/activate
jupyter lab "ca1 hq file.ipynb"
```

In JupyterLab, confirm the kernel at the top right is **Python (Agile MLOps CA1)**. If it is not, choose **Kernel → Change Kernel** and select that name. Then choose **Run → Run All Cells**. In VS Code, select the same kernel or the interpreter `.venv/bin/python` before running the notebook. The notebook runs with Python 3.13; the kernel is installed within `.venv` and is found when JupyterLab is launched from that environment.

## Recreate the environment if needed

These commands use `uv` to obtain Python 3.13 and create a project-local environment. The kernel registration is local to that environment.

```bash
uv venv --python 3.13 --seed .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name ca1-project --display-name "Python (Agile MLOps CA1)"
jupyter lab "ca1 hq file.ipynb"
```

The requirements pin the direct dependencies verified with Python 3.13.14. Avoid loading a saved joblib artifact with different library versions or from an untrusted source. Recreating `.venv` does not require touching the ignored data or MLflow artifact directories.

To run tests and execute the notebook from a fresh kernel:

```bash
python -m unittest discover -s tests -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=14400 "ca1 hq file.ipynb"
```

Ordinary Run All uses the newest complete PBI 06 research evidence that matches the input data, split, feature schema, installed packages and source fingerprint. Set `PBI06_RUN_ID` to require a specific compatible run. If no compatible run exists, run the full research workflow from the project root:

```bash
source .venv/bin/activate
PBI06_MODE=full jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=14400 "ca1 hq file.ipynb"
```

Full mode runs seven baselines, ten ordinary and 24 advanced configurations per family, then calibrates the selected models on reserved training dates. The results go to `PBI06_HanQian` on the configured tracking server, with local SQLite as the default. Run All defaults to `reuse` when a new kernel starts without `PBI06_MODE`. Keep `artifacts/mlflow/` and `artifacts/sealed_holdout_order_keys.sha256` together with the development environment. To view the evidence locally, launch MLflow from the project directory:

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in a browser. Stop the UI with Ctrl+C. The notebook explicitly uses the local store. The reusable training API can use a separately configured tracking URI in future; do not put credentials in the repository.

## PBI 02 — reusable data cleaning

`src/data_cleaning.py` contains `clean_customers`, `clean_orders`, `clean_labels` and `check_relationships`. The cleaners accept pandas DataFrames and return `(retained, rejected, report)`; they do not modify the inputs or write files. The notebook demonstrates the same cleaners on the development-eligible records returned by the sealed-data loader; the historical full-dataset cleaning report remains documented separately. A later pipeline can call them on each incoming batch. Run the representative checks with:

```bash
python -m unittest discover -s tests -v
```

Each cleaner requires the full source schema and raises a clear error if columns are missing, duplicated or the input is not a DataFrame. Country codes are trimmed and uppercased and must have two letters; IDs are trimmed and required. Order and label keys are `(country_code, order_id)`; customer keys are `(country_code, customer_id)`. These are checked before relationships are joined. Two-letter format is checked, but membership in an official country-code list is not verified.

For incoming CSVs, `src.data_access.read_dataset_csv(path)` supports filenames beginning `customers_`, `orders_` or `labels_`. It preserves identifier and country columns as strings (including values such as `0007` and `NA`), treats empty fields as missing, and reports duplicate headers, missing required columns, empty files and malformed rows before cleaning. It does not alter or write the source file. The development notebook continues to use `load_development_tables()` so the new-batch reader is not used to load sealed records.

True/false strings become nullable booleans. The meaning of `yes` is unconfirmed, so it and other unrecognized Boolean values become missing and are counted for review. Collection type is lowercased; missing values stay missing. Invalid, negative, nonfinite or fractional count values become missing; valid amounts and counts are preserved. Dates are parsed without assuming a timezone; invalid dates become missing and timezone-bearing dates raise an error. No statistical imputation, outlier capping or model-specific transformation is performed. Refunds above order value and customer refund/cancellation counts above order counts are counted for investigation, not removed.

The returned rejected DataFrame preserves the original values and includes a reason. Missing/invalid keys, invalid fraud labels, extra exact duplicate rows and **all** rows in a conflicting composite-key group are quarantined. Exact duplicates are assessed on original values so different invalid inputs are not silently collapsed. Other invalid feature values are counted in `report["issues"]` and retained as missing. Keep the original inputs if row-level investigation is needed. Issue counts are measured before rejection and can overlap.

Reports contain rows received, changed, rejected and retained; missing counts by column; invalid-value counts; and source row numbers plus composite identifiers for rows flagged for review. `changed` counts retained rows with any difference in their string representation after cleaning; date formatting and case normalization count, while a type conversion with identical text does not. Counts are therefore not estimates of erroneous rows. Collection and payment values are not rejected against an unverified vocabulary: observed values are reported under `category_values`, so new legitimate categories are retained. Relationships are checked separately on the retained tables; the function returns match counts and unmatched keys without deleting them. `compare_batch_keys()` reports overlaps with a caller-supplied historical key frame but cannot decide whether overlapping payloads are duplicates or conflicts. PBI 10 must perform that reconciliation and make database writes idempotent. A partial batch or unmatched customer is not automatically an error. Do not infer missing labels.

### Base-batch verification

With the supplied base files, the cleaners report: customers **901,312 received, 555 changed, 0 rejected, 901,312 retained**; orders **949,889 received, 949,889 changed, 0 rejected, 949,889 retained**; labels **950,318 received, 301 changed, 0 rejected, 950,318 retained**. All order rows count as changed because the collection type and/or date representation is standardized. Counts of changed rows depend on the input representation and the definition above.

After cleaning, 165 unconfirmed mobile values are missing, along with the original 712 customer first-order dates, 21,501 payment methods and one item count. The 199,687 refunds above order value remain available for review. The composite relationship checks yield 949,874 matched order/label rows, 444 label-only rows, 15 order-only rows, 14 label rows without a customer and 279,054 customers without labels. The base data were cleaned twice and the retained and rejected DataFrames and reports matched exactly. The unit tests include a synthetic incoming batch; a real authorised incoming batch and automated pipeline integration remain to be demonstrated when available.

## PBI 03 — shared feature engineering

All members can use the same interface below. Read the detailed [transformation record](reports/PBI03_feature_transformations.md) for the selected schema, split, exclusions and limitations. Use pandas 2.0+ and scikit-learn 1.2+; the local verification used pandas 3.1.1 and scikit-learn 1.9.1. Use the same dependency versions when saving and loading a fitted artifact.

```python
import joblib
import pandas as pd
from src.data_access import load_development_tables, VALIDATION_START
from src.data_cleaning import clean_customers, clean_orders, clean_labels
from src.feature_engineering import (
    prepare_training_data, prepare_features, fit_preprocessor, training_sample_weights,
)

customers_raw, orders_raw, labels_raw = load_development_tables()
customers = clean_customers(customers_raw)[0]
orders = clean_orders(orders_raw)[0]
labels = clean_labels(labels_raw)[0]
x_train, y_train, x_val, y_val, _, _, split_report, train_dates = prepare_training_data(
    customers, orders, labels, include_test=False, include_training_dates=True,
    validation_start_date=VALIDATION_START,
)
split_report.pop("training_dates")
preprocessor = fit_preprocessor(x_train)  # fit on training only
train_matrix = preprocessor.transform(x_train)
val_matrix = preprocessor.transform(x_val)
# Sealed test features and labels are not loaded during development.
weights = training_sample_weights(y_train)  # optional for a compatible estimator
joblib.dump(preprocessor, "artifacts/pbi03_preprocessor.joblib")  # create artifacts/ first

# For inference, add customer_id from a trusted order/customer mapping, not labels.
# new_orders = clean_orders(read_dataset_csv("orders_new.csv"))[0]
# new_customers = clean_customers(read_dataset_csv("customers_new.csv"))[0]
# future_features = prepare_features(new_orders_with_customer_id, new_customers)
# future_matrix = joblib.load("artifacts/pbi03_preprocessor.joblib").transform(future_features)
```

Choose hyperparameters and thresholds using training-only chronological folds; use validation to compare candidates. The sealed test partition was not accessed in this development phase. A future final evaluation requires a separately approved, controlled handoff after the champion and evaluation protocol are frozen.

The exact shared feature schema is `order_value`, `num_items_ordered`, `order_weekday`, `order_month`, `country_code`, `collect_type`, `payment_method`, `days_since_first_order`. Customer history and payment-method normalization are performed only in `src/feature_engineering.py`; PBI 06 and PBI 07 have no separate feature or imputation pipeline. The fitted artifact is saved locally at `artifacts/pbi03_preprocessor.joblib` and ignored by Git. Keep it alongside each model with matching library versions. Never load a joblib file from an untrusted source. Run all tests with `python -m unittest discover -s tests -v`.

The feature `days_since_first_order` is missing when its timestamp is invalid, absent or after the order date. However, the base order CSV does not contain `customer_id`; serving requires a trusted order-to-customer mapping independent of fraud labels. The source also does not prove when `first_order_datetime` is available. These point-in-time checks remain PBI 10 blockers until the data owner confirms them.

### PBI 08 integration for Ephraim

No PBI 08 notebook is present in this branch. Ephraim should call the shared functions rather than implement a member-specific pipeline:

```python
from src.feature_engineering import INPUT_FEATURES, prepare_training_data, fit_preprocessor

x_train, y_train, x_val, y_val, _, _, split_report, train_dates = prepare_training_data(
    customers, orders, labels, include_test=False, include_training_dates=True,
    validation_start_date=VALIDATION_START,
)
assert list(x_train.columns) == INPUT_FEATURES
preprocessor = fit_preprocessor(x_train)
train_matrix = preprocessor.transform(x_train)
validation_matrix = preprocessor.transform(x_val)
```

Fit Ephraim's estimator on `train_matrix` and `y_train`; use the same fitted `preprocessor` for validation and inference. Keep `train_dates` as split/calibration metadata only. If the model requires dense input, convert the transformed matrix without changing the feature values. Use the same latest-20%-of-whole-training-dates threshold calibration policy as PBI 07 for fair comparisons; never select thresholds from outer validation labels.

## PBI 06 — fixed-model retraining interface

Ephraim's future PBI 10 entry point can call the selected models without Jupyter or a search. Supply **unencoded, cleaned PBI 03 feature frames**, labels and the corresponding chronological split report. The function fits preprocessing on the training frame only, trains the fixed CatBoost and LightGBM configurations, evaluates validation using their recorded thresholds and returns local or shared MLflow run IDs, metrics and model/preprocessor/inference artifact URIs. It does not accept test data. No champion is registered.

```python
import os
from src.train_hq import train_selected_models

# x_train, y_train, x_val, y_val and split_report come from the
# development-only loader and prepare_training_data(..., include_test=False).
results = train_selected_models(
    x_train, y_train, x_val, y_val, split_report,
    tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
    batch_name="example-batch",
)
for result in results:
    print(result["model"], result["run_id"], result["threshold"], result["metrics"])
    print(result["model_uri"], result["preprocessor_uri"], result["inference_uri"])
```

If `tracking_uri` and `MLFLOW_TRACKING_URI` are unset, runs are local under `artifacts/mlflow/`. Configure the shared URI and any credentials outside the repository; connectivity and artifact storage on AWS have **not** been verified. `split_report` is a caller-supplied contract: use PBI 03 to ensure the dates and rows actually correspond. Applying saved thresholds to a new batch does not guarantee the same accuracy; monitor drift and assess on a properly separated validation period. The user's historical EDA examined full-data labels before the holdout existed.

To retrain a specific rebuilt configuration, pass `configurations={family: {"params": parameters, "weight": strength, "threshold": cutoff}}` using the approved policy values from the research report or the final MLflow run's `model/inference.json`. Omit this keyword to keep the original fixed CatBoost and LightGBM behavior. A fresh refit can shift probability scores, so the callable reports validation metrics at the supplied cutoff. The PBI 10 team pipeline and its three-minute runtime remain to be verified.

### Shared MLflow publication for the completed PBI 06 rebuild

From the project root, configure your shared tracking destination before opening Jupyter or calling a training entry point.

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com
python -m src.pbi06_publish
jupyter lab "ca1 hq file.ipynb"
```

The publication command resumes the approved 491-run tree by its source run IDs. Run it from one machine at a time. It verifies copied metrics and files and records a local receipt under `artifacts/pbi06_publish/`. Ordinary Run All reads the saved modelling evidence and queries the shared publication status. The configured remote is used for future writes, and a remote connection error is surfaced to the caller. The local fallback applies when no destination is configured.

The approved historical notebook evidence still uses local SQLite and its original fingerprint. For later runs logged directly to shared MLflow, set `PBI06_EVIDENCE_URI` to that tracking URI and keep the corresponding local private predictions under `artifacts/pbi06_private/`. The shared publication excludes the historical `predictions.npz` and removes per-row sample weights from the exported Random Forest copies.

The AWS proxy can reject large artifacts with HTTP 413. The publisher stores those files as verified parts beneath `<artifact>.parts/`. To retrieve either a standard or split model artifact, use the same downloader used by verification:

```python
from pathlib import Path
import os
from mlflow import MlflowClient
from src.pbi06_publish import download_published_artifact

folder = Path("artifacts/downloaded_models")
folder.mkdir(parents=True, exist_ok=True)
client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
model_path = download_published_artifact(client, shared_run_id, "model/model.joblib", folder)
```

The publisher records the shared experiment and parent IDs under `artifacts/pbi06_publish/destination.json`. If the endpoint comes back with a different or empty tracking store, the command stops before creating replacement runs. Resume after the original experiment and its parent are available at the configured endpoint.

If the original AWS store returns, run `python -m src.pbi06_publish --rename-original` to verify its experiment ID and parent before renaming it in place. Then run the normal publisher to finish verification.

If the original store is gone, confirm the replacement endpoint and its tracking and artifact persistence with the team first. After explicit approval and a successful synthetic tracking and artifact probe, run:

```bash
python -m src.pbi06_publish --recover-replacement \
    --readiness-file artifacts/pbi06_publish/readiness.json \
    --workers 4 --batch-size 24
```

The readiness JSON contains the finished probe's `run_id` and the SHA-256 of its `readiness.txt` artifact. The probe in Default uses parameter `scope=synthetic` and metric `probe=1`. The publisher checks the probe again before creating the experiment, uploads and loads all seven final models first, and checks that previously verified runs remain available between batches. A failed check stops further batches. After a successful first publication, use the normal command to resume on the recorded backend.
