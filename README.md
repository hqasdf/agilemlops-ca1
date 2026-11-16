# ST1519 Agile MLOps CA1

`CA1.ipynb` is the main local development notebook. Keep `customers_base.csv`, `orders_base.csv` and `labels_base.csv` alongside it; these raw files and the local `artifacts/` directory are ignored by Git. The notebook runs historical EDA, full-data cleaning, training-only preprocessing and the six validation baseline models. The original full-history EDA examined labels before the test holdout existed; do not use test outcomes for current model selection.

## Start the existing project environment

From the project folder on macOS:

```bash
source .venv/bin/activate
jupyter lab CA1.ipynb
```

In JupyterLab, confirm the kernel at the top right is **Python (Agile MLOps CA1)**. If it is not, choose **Kernel → Change Kernel** and select that name. Then choose **Run → Run All Cells**. In VS Code, select the same kernel or the interpreter `.venv/bin/python` before running the notebook. The notebook runs with Python 3.13; the kernel is installed within `.venv` and is found when JupyterLab is launched from that environment.

## Recreate the environment if needed

These commands use `uv` to obtain Python 3.13 and create a project-local environment. The kernel registration is local to that environment.

```bash
uv venv --python 3.13 --seed .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name ca1-project --display-name "Python (Agile MLOps CA1)"
jupyter lab CA1.ipynb
```

The requirements pin the direct dependencies verified with Python 3.13.14. Avoid loading a saved joblib artifact with different library versions or from an untrusted source. Recreating `.venv` does not require touching the ignored data or MLflow artifact directories.

To run tests and execute the notebook from a fresh kernel:

```bash
python -m unittest discover -s tests -v
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=ca1-project --ExecutePreprocessor.timeout=600 CA1.ipynb
```

Each notebook run trains six baselines, runs the initial training-only search and selects thresholds from training-only folds. The advanced search is disabled by default and saved local MLflow runs are reused; keep `artifacts/mlflow/` intact. To rerun the approximately 9.5-minute advanced search, set `RUN_ADVANCED_SEARCH = True` in its notebook cell. The normal `train_benchmark` callable does not run the searches or threshold optimization. To view local runs, open a second terminal in the project folder:

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in a browser. Stop the UI with Ctrl+C. The notebook explicitly uses the local store. The reusable training API can use a separately configured tracking URI in future; do not put credentials in the repository.

## PBI 02 — reusable data cleaning

`src/data_cleaning.py` contains `clean_customers`, `clean_orders`, `clean_labels` and `check_relationships`. The cleaners accept pandas DataFrames and return `(retained, rejected, report)`; they do not modify the inputs or write files. The notebook demonstrates the same functions on the full base datasets. A later pipeline can call them on each incoming batch. Run the representative checks with:

```bash
python -m unittest discover -s tests -v
```

Each cleaner requires the full source schema and raises a clear error if columns are missing, duplicated or the input is not a DataFrame. Country codes are trimmed and uppercased and must have two letters; IDs are trimmed and required. Order and label keys are `(country_code, order_id)`; customer keys are `(country_code, customer_id)`. These are checked before relationships are joined. Two-letter format is checked, but membership in an official country-code list is not verified.

True/false strings become nullable booleans. The meaning of `yes` is unconfirmed, so it and other unrecognized Boolean values become missing and are counted for review. Collection type is lowercased; missing values stay missing. Invalid, negative, nonfinite or fractional count values become missing; valid amounts and counts are preserved. Dates are parsed without assuming a timezone; invalid dates become missing and timezone-bearing dates raise an error. No statistical imputation, outlier capping or model-specific transformation is performed. Refunds above order value and customer refund/cancellation counts above order counts are counted for investigation, not removed.

The returned rejected DataFrame preserves the original values and includes a reason. Missing/invalid keys, invalid fraud labels, extra exact duplicate rows and **all** rows in a conflicting composite-key group are quarantined. Exact duplicates are assessed on original values so different invalid inputs are not silently collapsed. Other invalid feature values are counted in `report["issues"]` and retained as missing. Keep the original inputs if row-level investigation is needed. Issue counts are measured before rejection and can overlap.

Reports contain rows received, changed, rejected and retained. `changed` counts retained rows with any difference in their string representation after cleaning; date formatting and case normalization count, while a type conversion with identical text does not. Counts are therefore not estimates of erroneous rows. Relationships are checked separately on the retained tables; the function returns match counts and unmatched keys without deleting them. It assumes the supplied tables represent the desired comparison scope, so a partial batch or an unmatched customer is not automatically an error. Do not infer missing labels.

### Base-batch verification

With the supplied base files, the cleaners report: customers **901,312 received, 555 changed, 0 rejected, 901,312 retained**; orders **949,889 received, 949,889 changed, 0 rejected, 949,889 retained**; labels **950,318 received, 301 changed, 0 rejected, 950,318 retained**. All order rows count as changed because the collection type and/or date representation is standardized. Counts of changed rows depend on the input representation and the definition above.

After cleaning, 165 unconfirmed mobile values are missing, along with the original 712 customer first-order dates, 21,501 payment methods and one item count. The 199,687 refunds above order value remain available for review. The composite relationship checks yield 949,874 matched order/label rows, 444 label-only rows, 15 order-only rows, 14 label rows without a customer and 279,054 customers without labels. The base data were cleaned twice and the retained and rejected DataFrames and reports matched exactly. The unit tests include a synthetic incoming batch; a real authorised incoming batch and automated pipeline integration remain to be demonstrated when available.

## PBI 03 — shared feature engineering

All members can use the same interface below. Read the detailed [transformation record](reports/PBI03_feature_transformations.md) for the selected schema, split, exclusions and limitations. Use pandas 2.0+ and scikit-learn 1.2+; the local verification used pandas 3.1.1 and scikit-learn 1.9.1. Use the same dependency versions when saving and loading a fitted artifact.

```python
import joblib
import pandas as pd
from src.data_cleaning import clean_customers, clean_orders, clean_labels
from src.feature_engineering import (
    prepare_training_data, prepare_features, fit_preprocessor, training_sample_weights,
)

customers = clean_customers(pd.read_csv("customers_base.csv", low_memory=False))[0]
orders = clean_orders(pd.read_csv("orders_base.csv", low_memory=False))[0]
labels = clean_labels(pd.read_csv("labels_base.csv", low_memory=False))[0]
x_train, y_train, x_val, y_val, _, _, split_report = prepare_training_data(
    customers, orders, labels, include_test=False
)
preprocessor = fit_preprocessor(x_train)  # fit on training only
train_matrix = preprocessor.transform(x_train)
val_matrix = preprocessor.transform(x_val)
# Test features and labels are not returned during model development.
weights = training_sample_weights(y_train)  # optional for a compatible estimator
joblib.dump(preprocessor, "artifacts/pbi03_preprocessor.joblib")  # create artifacts/ first

# Later, clean new orders and apply this same fitted artifact; never refit on them.
# new_orders = clean_orders(pd.read_csv("orders_new.csv"))[0]
# future_matrix = joblib.load("artifacts/pbi03_preprocessor.joblib").transform(prepare_features(new_orders))
```

Choose hyperparameters and thresholds on training-only folds; use validation to compare the resulting models. Only after selecting the final champion, transform the held-out test features with the saved preprocessor and evaluate that model once; do not inspect test labels or use them for tuning.

The base-data fitted artifact is saved locally at `artifacts/pbi03_preprocessor.joblib` and ignored by Git. Keep a trusted copy alongside each eventual model, with matching library versions. Never load a joblib file from an untrusted source. Run all tests with `python -m unittest discover -s tests -v`.

## PBI 06 — fixed-model retraining interface

Ephraim's future PBI 10 entry point can call the selected models without Jupyter or a search. Supply **unencoded, cleaned PBI 03 feature frames**, labels and the corresponding chronological split report. The function fits preprocessing on the training frame only, trains the fixed CatBoost and LightGBM configurations, evaluates validation using their recorded thresholds and returns local or shared MLflow run IDs, metrics and model/preprocessor/inference artifact URIs. It does not accept test data. No champion is registered.

```python
import os
from src.train_hq import train_selected_models

# x_train, y_train, x_val, y_val and split_report come from
# prepare_training_data(cleaned_customers, cleaned_orders, cleaned_labels,
#                       include_test=False).
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

The final 15-version comparison in `CA1.ipynb` loads the original local runs and does not perform tuning or threshold searches on Run All. If the local MLflow database or its artifacts are missing, restore `artifacts/mlflow/` from backup; the notebook deliberately fails rather than inventing results. This interface is not the full PBI 10 ETL pipeline.
