"""PBI 10 experimental CLI: do not use for assessed runs until live verified."""
import argparse
import json
import sys
from src.pbi10_config import read_config, open_database, ConfigError
from src.pbi10_holdout import read_sealed_manifest
from src.pbi10_command import execute_pipeline, read_new_keys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('customers','orders','labels'):
        parser.add_argument('--'+name+'-url',required=True)
    parser.add_argument('--batch-id',required=True)
    parser.add_argument('--sealed-manifest',required=True)
    parser.add_argument('--new-order-ledger',required=True)
    parser.add_argument('--config',help='Optional KEY=VALUE settings file; environment variables override it')
    args=parser.parse_args()
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

