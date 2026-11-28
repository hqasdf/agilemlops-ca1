"""PBI 10 pipeline: download, clean, load and retrain all members' models for a new batch.

Put the three CSV links and other settings in .env beside this file, then run it with no
arguments. Command-line options, where given, take priority over the settings.
"""
import argparse
import json
import sys
import time
from src.pbi10_config import read_config, read_run_settings, find_config_file, open_database, ConfigError
from src.pbi10_holdout import read_sealed_manifest
from src.pbi10_command import execute_pipeline, read_new_keys

# command-line option, setting name in .env
REQUIRED_RUN_VALUES = (('customers_url', 'CA1_CUSTOMERS_URL'), ('orders_url', 'CA1_ORDERS_URL'),
                       ('labels_url', 'CA1_LABELS_URL'), ('sealed_manifest', 'CA1_SEALED_MANIFEST'),
                       ('new_order_ledger', 'CA1_NEW_ORDER_LEDGER'))


def resolve_run_values(args, settings):
    missing = []
    for option, name in REQUIRED_RUN_VALUES:
        value = getattr(args, option) or settings[name]
        setattr(args, option, value)
        if not value:
            missing.append(f"{name} (or --{option.replace('_', '-')})")
    if missing:
        raise ConfigError('Missing run settings: ' + ', '.join(missing) +
                          '. Add them to .env or pass the options on the command line.')
    args.batch_id = args.batch_id or settings['CA1_BATCH_ID'] or time.strftime('batch-%Y%m%d-%H%M%S')
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('customers', 'orders', 'labels'):
        parser.add_argument(f'--{name}-url', help=f'Public CSV URL (default: CA1_{name.upper()}_URL)')
    parser.add_argument('--batch-id', help='Default: CA1_BATCH_ID, otherwise a timestamp')
    parser.add_argument('--sealed-manifest')
    parser.add_argument('--new-order-ledger')
    parser.add_argument('--config', help='Settings file (default: CA1_CONFIG_FILE, otherwise .env)')
    args = parser.parse_args()
    try:
        config_file = find_config_file(args.config)
        resolve_run_values(args, read_run_settings(config_file=config_file))
        config = read_config(config_file=config_file)
        sealed = read_sealed_manifest(args.sealed_manifest)
        new_keys = read_new_keys(args.new_order_ledger)
        if sealed.intersection(new_keys):
            raise ValueError('New-order ledger overlaps sealed historical keys')
        urls = {name: getattr(args, name + '_url') for name in ('customers', 'orders', 'labels')}
        connection = open_database(config)
        summary = execute_pipeline(connection, urls, args.batch_id, config['MLFLOW_TRACKING_URI'], sealed, new_keys,
                                   args.new_order_ledger,
                                   download_options={'max_bytes': config['CA1_DOWNLOAD_MAX_BYTES'],
                                                     'timeout': config['CA1_DOWNLOAD_TIMEOUT_SECONDS']})
    except ConfigError as exc:
        print(json.dumps({'status': 'failed', 'error_type': 'ConfigError', 'error': str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}))
        return 1
    print(json.dumps(summary, default=str, indent=2))
    return 0 if summary['status'] == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
