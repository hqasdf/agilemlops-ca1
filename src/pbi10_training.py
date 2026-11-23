"""Adapters for the team's existing training functions."""


def prepare_retraining(tables, sealed_keys, new_keys, current_batch_keys):
    from src.feature_engineering import prepare_training_data
    if not current_batch_keys or not set(current_batch_keys).issubset(new_keys):
        raise ValueError('Current batch order keys must be in the verified new-order allowlist')
    x_train,y_train,x_val,y_val,_,_,report,dates=prepare_training_data(
        tables['customers'],tables['orders'],tables['labels'],
        include_training_dates=True,sealed_order_keys=sealed_keys,new_order_keys=new_keys,
        current_batch_keys=current_batch_keys)
    if report['new_training_rows'] < 1:
        raise ValueError('No new labelled orders are in the training split; review the split before retraining')
    return x_train,y_train,x_val,y_val,dates,report


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
        results=nithish(None,None,None,tracking_uri=tracking_uri,batch_name=batch_id,prepared_data=prepared,
                        current_batch_indices=report['new_training_indices'])
    else:
        if ephraim is None:
            from src.train_ephraim import train_ephraim_models as ephraim
        results=ephraim(x_train,y_train,x_val,y_val,tracking_uri=tracking_uri,batch_name=batch_id)
    return [{'run_id':row.get('run_id'),'model_uri':row.get('model_uri'),
             'model_name':row.get('name',row.get('version',row.get('model')))} for row in results]

