"""Adapters for the team's existing training functions."""


def train_member(member, prepared, batch_id, tracking_uri, hq=None, nithish=None, ephraim=None):
    if member not in ('han_qian','nithish','ephraim'):
        raise ValueError('Unknown team member')
    x_train,y_train,x_val,y_val,train_dates,report=prepared
    if member=='han_qian':
        if hq is None:
            from src.train_hq import train_selected_models as hq
        results=hq(x_train,y_train,x_val,y_val,report,tracking_uri=tracking_uri,batch_name=batch_id)
    elif member=='nithish':
        if nithish is None:
            from src.train_nithish import train_nithish_models as nithish
        results=nithish(None,None,None,tracking_uri=tracking_uri,batch_name=batch_id,prepared_data=prepared)
    else:
        if ephraim is None:
            from src.train_ephraim import train_ephraim_models as ephraim
        results=ephraim(x_train,y_train,x_val,y_val,tracking_uri=tracking_uri,batch_name=batch_id)
    return [{'run_id':row.get('run_id'),'model_uri':row.get('model_uri'),
             'model_name':row.get('name',row.get('version',row.get('model')))} for row in results]

