# PBI 01 — exploratory data analysis

Place `customers_base.csv`, `orders_base.csv` and `labels_base.csv` alongside `lab1.ipynb`. These source files are excluded from Git. The notebook reads them without modifying them and shows the evidence behind the quality register and PBI 02/03 recommendations in `reports/`.

Use Python 3.11 or newer and set up a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy matplotlib seaborn jupyterlab nbconvert ipykernel
jupyter lab lab1.ipynb
```

Run all cells from the notebook directory. For a command-line verification that writes the executed outputs back into the notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 lab1.ipynb
```

The data contain approximately 950,000 rows per file, so execution requires sufficient memory. Outputs are aggregate summaries; avoid sharing raw data or credentials. Run the notebook only on trusted data.

## PBI 02 — reusable data cleaning

`src/data_cleaning.py` contains `clean_customers`, `clean_orders`, `clean_labels` and `check_relationships`. The cleaners accept pandas DataFrames and return `(retained, rejected, report)`; they do not modify the inputs or write files. The notebook demonstrates the same functions on a small sample. A later pipeline can call them on each incoming batch. Run the representative checks with:

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
