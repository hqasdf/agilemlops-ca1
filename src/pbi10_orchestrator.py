"""Coordinate batch stages and report each member's retraining outcome."""
import time
from urllib.parse import urlparse


def run_batch(batch_id, tracking_uri, prepare, load, read_updated, train):
    if not batch_id or urlparse(tracking_uri).scheme not in ('http','https') or not urlparse(tracking_uri).hostname:
        raise ValueError('A batch ID and shared HTTP(S) MLflow URL are required')
    start=time.perf_counter()
    summary={'batch_id':batch_id,'status':'failed','models':{}}
    try:
        prepared=prepare()
        summary['database']=load(prepared)
        tables=read_updated()
    except Exception as exc:
        summary['stage_error']=type(exc).__name__
        summary['elapsed_seconds']=round(time.perf_counter()-start,3)
        return summary
    for member in ('han_qian','nithish','ephraim'):
        try:
            runs=train(member,tables,batch_id,tracking_uri)
            if not runs or any(not item.get('run_id') or not item.get('model_uri') for item in runs):
                raise ValueError('A run ID and model URI are required for each model')
            summary['models'][member]={'status':'success','runs':runs}
        except Exception as exc:
            summary['models'][member]={'status':'failed','error_type':type(exc).__name__}
    summary['status']='success' if all(v['status']=='success' for v in summary['models'].values()) else 'failed'
    summary['elapsed_seconds']=round(time.perf_counter()-start,3)
    return summary

