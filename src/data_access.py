"""Read development-eligible CSV rows without loading sealed test fields.

The one-time membership file contains only SHA-256 hashes of held-out order
keys. During ingestion, identifier/date columns are read first; full rows are
loaded only for training and validation records.
"""

import hashlib
import csv
from pathlib import Path

import pandas as pd

from src.data_cleaning import CUSTOMER_COLUMNS, ORDER_COLUMNS, LABEL_COLUMNS


SEALED_START = pd.Timestamp("2023-04-14")
VALIDATION_START = pd.Timestamp("2023-03-15")
MANIFEST_PATH = Path("artifacts/sealed_holdout_order_keys.sha256")
ORDER_KEY = ["country_code", "order_id"]
CUSTOMER_KEY = ["country_code", "customer_id"]

DATASET_COLUMNS = {
    "customers": CUSTOMER_COLUMNS,
    "orders": ORDER_COLUMNS,
    "labels": LABEL_COLUMNS,
}
IDENTIFIER_COLUMNS = {
    "customers": ["country_code", "customer_id"],
    "orders": ["country_code", "order_id"],
    "labels": ["country_code", "order_id", "customer_id"],
}


def read_dataset_csv(path, dataset_type=None):
    """Read one incoming source CSV without inferring its identifiers.

    ``dataset_type`` is customers, orders or labels. If omitted, it is inferred
    from the filename prefix (for example, ``orders_042.csv``). Non-key values
    are left to pandas inference; the cleaners validate and convert them.
    """
    path = Path(path)
    if dataset_type is None:
        dataset_type = path.name.split("_", 1)[0]
    if dataset_type not in DATASET_COLUMNS:
        raise ValueError("dataset_type must be customers, orders or labels")
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file, strict=True)
            header = next(reader, None)
        if not header:
            raise ValueError(f"CSV file is empty or has no header: {path}")
        if len(header) != len(set(header)):
            duplicates = sorted({name for name in header if header.count(name) > 1})
            raise ValueError(f"CSV has duplicate column names: {duplicates}")
        missing = [name for name in DATASET_COLUMNS[dataset_type] if name not in header]
        if missing:
            raise ValueError(f"{dataset_type} CSV is missing required columns: {missing}")
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file, strict=True)
            next(reader, None)
            for row_number, row in enumerate(reader, start=2):
                if row and len(row) != len(header):
                    raise ValueError(
                        f"Malformed CSV row {row_number}: expected {len(header)} fields, found {len(row)}"
                    )

        data = pd.read_csv(
            path,
            dtype={name: "string" for name in IDENTIFIER_COLUMNS[dataset_type]},
            keep_default_na=False,
            na_values=[""],
            encoding="utf-8-sig",
            on_bad_lines="error",
        )
    except (UnicodeDecodeError, csv.Error, pd.errors.ParserError, OSError) as error:
        raise ValueError(f"Could not read {dataset_type} CSV {path}: {error}") from error

    if data.empty:
        raise ValueError(f"CSV contains a header but no data rows: {path}")
    return data


def _key_hash(country, identifier):
    country = str(country).strip().upper()
    identifier = str(identifier).strip()
    value = country + "\0" + identifier
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _key_hashes(frame, identifier):
    return pd.Series(
        [_key_hash(country, value) for country, value in zip(frame["country_code"], frame[identifier])],
        index=frame.index,
    )


def _read_keys(path, identifier, extra_columns=None):
    columns = ["country_code", identifier] + (extra_columns or [])
    data = pd.read_csv(path, usecols=columns, dtype={column: "string" for column in columns},
                       keep_default_na=False, na_values=[""])
    data["country_code"] = data["country_code"].str.strip().str.upper()
    data[identifier] = data[identifier].str.strip()
    return data


def _save_sealed_manifest(order_keys, order_dates, manifest_path):
    if manifest_path.exists():
        return set(manifest_path.read_text().splitlines())

    sealed = order_keys.loc[order_dates >= SEALED_START]
    hashes = sorted(_key_hashes(sealed, "order_id").tolist())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(hashes) + ("\n" if hashes else ""))
    return set(hashes)


def _read_rows_for_positions(path, positions, dtype):
    """Read selected CSV records; skipped lines are not parsed into a DataFrame."""
    keep = set(positions)
    skip_row = lambda line_number: line_number > 0 and line_number - 1 not in keep
    return pd.read_csv(path, dtype=dtype, keep_default_na=False, na_values=[""],
                       skiprows=skip_row, low_memory=False)


def load_development_tables(customers_path="customers_base.csv",
                            orders_path="orders_base.csv",
                            labels_path="labels_base.csv",
                            manifest_path=MANIFEST_PATH):
    """Return customers, orders and labels for the original train/validation dates.

    The holdout manifest is initialized from order keys and dates only. Test
    feature and label columns are skipped while reading full CSV records.
    Unmatched label keys are excluded because they have no safe order date.
    """
    orders_path = Path(orders_path)
    customers_path = Path(customers_path)
    labels_path = Path(labels_path)

    order_metadata = _read_keys(orders_path, "order_id", ["order_date"])
    order_dates = pd.to_datetime(order_metadata["order_date"], errors="coerce", format="mixed")
    sealed_hashes = _save_sealed_manifest(order_metadata, order_dates, Path(manifest_path))
    order_hashes = _key_hashes(order_metadata, "order_id")
    development_mask = order_dates.notna() & (order_dates < SEALED_START)
    development_mask &= ~order_hashes.isin(sealed_hashes)
    development_order_positions = order_metadata.index[development_mask].tolist()
    development_order_keys = order_metadata.loc[development_mask, ORDER_KEY].drop_duplicates()

    label_metadata = _read_keys(labels_path, "order_id", ["customer_id"])
    label_metadata["customer_id"] = label_metadata["customer_id"].str.strip()
    label_metadata["_row_position"] = label_metadata.index
    development_label_keys = label_metadata.merge(
        development_order_keys, on=ORDER_KEY, how="inner", validate="many_to_one"
    )
    label_positions = development_label_keys["_row_position"].tolist()
    development_customer_keys = development_label_keys[CUSTOMER_KEY].drop_duplicates()

    order_dtype = {column: "string" for column in ["country_code", "order_id"]}
    orders = _read_rows_for_positions(orders_path, development_order_positions, order_dtype)
    label_dtype = {column: "string" for column in ["country_code", "order_id", "customer_id"]}
    labels = _read_rows_for_positions(labels_path, label_positions, label_dtype)

    customer_metadata = _read_keys(customers_path, "customer_id")
    customer_metadata["_row_position"] = customer_metadata.index
    customer_positions = customer_metadata.merge(
        development_customer_keys, on=CUSTOMER_KEY, how="inner", validate="many_to_one"
    )["_row_position"].tolist()
    customer_dtype = {column: "string" for column in ["country_code", "customer_id"]}
    customers = _read_rows_for_positions(customers_path, customer_positions, customer_dtype)

    return customers, orders, labels
