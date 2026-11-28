# PBI 10: CSV URL preflight

This first stage downloads and validates the three inputs, then applies the existing PBI 02 cleaners. It reports received, retained and rejected counts. It is an input preflight, not the completed PBI 10 ETL: it does not write to RDS, train models or log to MLflow.

Install the team's pinned dependencies in a project virtual environment as described in `requirements.txt`. From the project root:

```powershell
python pbi10_validate.py --customers-url "https://example.org/customers_001.csv" --orders-url "https://example.org/orders_001.csv" --labels-url "https://example.org/labels_001.csv"
```

Use publicly accessible direct CSV download URLs. Change the three command arguments for each batch; no source edits are required. The files are downloaded to a temporary folder. The CSV reader preserves text identifiers such as leading zeros and rejects missing/duplicate headers, malformed rows and empty files. The download helper rejects failed or empty downloads, HTML pages and files over 250 MB; it uses a 30-second request timeout. Cleaner rejection counts are not database quarantine counts; the pipeline must separately enforce the database schema and reconcile records.

Run the focused test with `python -m unittest discover -s tests -p 'test_pbi10_inputs.py' -v`. The test uses a local HTTP server and synthetic data; it does not contact AWS. The team still needs a verified public practice URL and the integrated pipeline for the assessed demonstration.

