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
