"""Preflight three public CSV URLs without writing to the database."""
import argparse
import json
from src.pbi10_inputs import prepare_downloaded_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("customers", "orders", "labels"):
        parser.add_argument(f"--{name}-url", required=True)
    args = parser.parse_args()
    result = prepare_downloaded_batch({name: getattr(args, f"{name}_url") for name in ("customers", "orders", "labels")})
    summary = {name: {key: report[key] for key in ("received", "retained", "rejected")} for name, report in result["reports"].items()}
    print(json.dumps({"status": "validated_only", "tables": summary}, indent=2))


if __name__ == "__main__":
    main()

