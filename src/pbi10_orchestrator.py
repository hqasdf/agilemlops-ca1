"""Coordinate batch stages and report each member's retraining outcome."""
import time
import sys
import traceback
from urllib.parse import urlparse


def run_batch(batch_id, tracking_uri, prepare, load, read_updated, train):
    if not batch_id or urlparse(tracking_uri).scheme not in ('http','https') or not urlparse(tracking_uri).hostname:
        raise ValueError('A batch ID and shared HTTP(S) MLflow URL are required')
    start=time.perf_counter()
    summary={'batch_id':batch_id,'status':'failed','models':{}}
    stage='prepare'
    try:
        print('PBI 10: downloading and validating the batch', file=sys.stderr, flush=True)
        prepared=prepare()
        stage='load'
        print('PBI 10: loading accepted records into the database', file=sys.stderr, flush=True)
        summary['database']=load(prepared)
        stage='read_and_prepare_training'
        print('PBI 10: preparing historical and new data for training', file=sys.stderr, flush=True)
        tables=read_updated()
    except Exception as exc:
        summary['failed_stage']=stage
        summary['stage_error']=type(exc).__name__
        if isinstance(exc,ValueError):
            # Raised by our own download, validation and load checks: say what was wrong.
            summary['stage_error_message']=' '.join(str(exc).split())[:300]
        if type(exc).__name__ in ('OperationalError','IntegrityError','DataError','ProgrammingError'):
            summary['database_error_code']=exc.args[0] if exc.args and isinstance(exc.args[0],int) else None
            summary['database_error_message']=str(exc.args[1]) if len(exc.args)>1 else ''
        summary['elapsed_seconds']=round(time.perf_counter()-start,3)
        return summary
    for member in ('han_qian','nithish','ephraim'):
        try:
            print(f'PBI 10: retraining {member}', file=sys.stderr, flush=True)
            runs=train(member,tables,batch_id,tracking_uri)
            if not runs or any(not item.get('run_id') or not item.get('model_uri') for item in runs):
                raise ValueError('A run ID and model URI are required for each model')
            summary['models'][member]={'status':'success','runs':runs}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            summary['models'][member]={'status':'failed','error_type':type(exc).__name__,
                                       'error':' '.join(str(exc).split())[:300]}
    summary['status']='success' if all(v['status']=='success' for v in summary['models'].values()) else 'failed'
    summary['elapsed_seconds']=round(time.perf_counter()-start,3)
    return summary

