"""Preflight three public CSV URLs without writing to the database."""
import argparse
import json
from src.pbi10_config import find_config_file, read_run_settings
from src.pbi10_inputs import prepare_downloaded_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("customers", "orders", "labels"):
        parser.add_argument(f"--{name}-url", help=f"Default: CA1_{name.upper()}_URL")
    parser.add_argument("--config", help="Settings file (default: CA1_CONFIG_FILE, otherwise .env)")
    args = parser.parse_args()
    settings = read_run_settings(config_file=find_config_file(args.config))
    urls = {name: getattr(args, f"{name}_url") or settings[f"CA1_{name.upper()}_URL"]
            for name in ("customers", "orders", "labels")}
    missing = [f"CA1_{name.upper()}_URL" for name, url in urls.items() if not url]
    if missing:
        parser.error("Missing " + ", ".join(missing) + ". Add them to .env or pass the --...-url options.")
    result = prepare_downloaded_batch(urls)
    summary = {name: {key: report[key] for key in ("received", "retained", "rejected")} for name, report in result["reports"].items()}
    print(json.dumps({"status": "validated_only", "tables": summary}, indent=2))


if __name__ == "__main__":
    main()

