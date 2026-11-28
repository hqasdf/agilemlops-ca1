"""PBI 10 pipeline: download, clean, load and retrain all members' models for a new batch.

Edit the values below to run with no command-line arguments, or pass the matching
--options, which always take priority over these values.
"""
import argparse
import json
import sys
import time
from src.pbi10_config import read_config, open_database, ConfigError
from src.pbi10_holdout import read_sealed_manifest
from src.pbi10_command import execute_pipeline, read_new_keys

# ---- Edit these for a run (leave '' to supply the option on the command line) ----
CUSTOMERS_URL = ''
ORDERS_URL = ''
LABELS_URL = ''
BATCH_ID = ''             # blank: a timestamp such as batch-20261125-143000 is used
SEALED_MANIFEST = ''      # path to the sealed historical-order manifest
NEW_ORDER_LEDGER = ''     # path to the new-order ledger
CONFIG_FILE = ''          # optional KEY=VALUE settings file (RDS and MLflow settings)
# ----------------------------------------------------------------------------------


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for name,value in (('customers',CUSTOMERS_URL),('orders',ORDERS_URL),('labels',LABELS_URL)):
        parser.add_argument('--'+name+'-url',default=value or None,help='Public CSV URL (default: value set in this file)')
    parser.add_argument('--batch-id',default=BATCH_ID or time.strftime('batch-%Y%m%d-%H%M%S'))
    parser.add_argument('--sealed-manifest',default=SEALED_MANIFEST or None)
    parser.add_argument('--new-order-ledger',default=NEW_ORDER_LEDGER or None)
    parser.add_argument('--config',default=CONFIG_FILE or None,
                        help='Optional KEY=VALUE settings file; environment variables override it')
    args=parser.parse_args()
    missing=[option for option,value in (('--customers-url',args.customers_url),('--orders-url',args.orders_url),
                                         ('--labels-url',args.labels_url),('--sealed-manifest',args.sealed_manifest),
                                         ('--new-order-ledger',args.new_order_ledger)) if not value]
    if missing:
        parser.error('Missing '+', '.join(missing)+'. Pass them on the command line or set them at the top of pbi10_pipeline.py.')
    try:
        config=read_config(config_file=args.config)
        sealed=read_sealed_manifest(args.sealed_manifest)
        new_keys=read_new_keys(args.new_order_ledger)
        if sealed.intersection(new_keys):
            raise ValueError('New-order ledger overlaps sealed historical keys')
        urls={name:getattr(args,name+'_url') for name in ('customers','orders','labels')}
        connection=open_database(config)
        summary=execute_pipeline(connection,urls,args.batch_id,config['MLFLOW_TRACKING_URI'],sealed,new_keys,args.new_order_ledger,
                                 download_options={'max_bytes':config['CA1_DOWNLOAD_MAX_BYTES'],'timeout':config['CA1_DOWNLOAD_TIMEOUT_SECONDS']})
    except ConfigError as exc:
        print(json.dumps({'status':'failed','error_type':'ConfigError','error':str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__}))
        return 1
    print(json.dumps(summary,default=str,indent=2))
    return 0 if summary['status']=='success' else 1


if __name__=='__main__':
    sys.exit(main())

