"""Download and clean a complete incoming batch before any database work."""
from tempfile import TemporaryDirectory
from pathlib import Path

from src.data_access import read_dataset_csv
from src.data_cleaning import clean_customers, clean_orders, clean_labels
from src.pbi10_download import download_batch


def prepare_downloaded_batch(urls, max_bytes=250_000_000, timeout=30):
    cleaners = {"customers": clean_customers, "orders": clean_orders, "labels": clean_labels}
    with TemporaryDirectory(prefix="pbi10-") as directory:
        paths = download_batch(urls, Path(directory), max_bytes=max_bytes, timeout=timeout)
        raw = {name: read_dataset_csv(path, name) for name, path in paths.items()}
        cleaned, rejected, reports = {}, {}, {}
        for name, cleaner in cleaners.items():
            cleaned[name], rejected[name], reports[name] = cleaner(raw[name])
        return {"cleaned": cleaned, "rejected": rejected, "reports": reports}

