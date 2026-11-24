"""Adapters for the team's existing training functions."""


def add_run_links(runs, tracking_uri, client):
    from urllib.parse import quote
    for row in runs:
        info = client.get_run(row['run_id']).info
        row['run_link'] = (tracking_uri.rstrip('/') + '/#/experiments/' +
                           quote(str(info.experiment_id), safe='') + '/runs/' +
                           quote(str(row['run_id']), safe=''))
    return runs


def choose_batch_validation_start(orders, labels, current_batch_keys):
    import pandas as pd
    from src.data_access import _key_hashes
    keys={'country_code','order_id'}
    if not current_batch_keys or not keys.issubset(orders.columns) or not keys.issubset(labels.columns):
        raise ValueError('Current order keys and labels are required')
    matched=orders.merge(labels[['country_code','order_id']],on=['country_code','order_id'],how='inner',validate='one_to_one')
    mask=_key_hashes(matched,'order_id').isin(current_batch_keys)
    dates=pd.to_datetime(matched.loc[mask,'order_date'],errors='coerce').dropna().sort_values().unique()
    if len(dates)<2:
        raise ValueError('Current labelled batch needs at least two distinct valid order dates for chronological validation')
    return pd.Timestamp(dates[min(int(len(dates)*0.8),len(dates)-1)])


def prepare_retraining(tables, sealed_keys, new_keys, current_batch_keys):
    from src.feature_engineering import prepare_training_data
    if not current_batch_keys or not set(current_batch_keys).issubset(new_keys):
        raise ValueError('Current batch order keys must be in the verified new-order allowlist')
    validation_start=choose_batch_validation_start(tables['orders'],tables['labels'],current_batch_keys)
    x_train,y_train,x_val,y_val,_,_,report,dates=prepare_training_data(
        tables['customers'],tables['orders'],tables['labels'],
        include_training_dates=True,sealed_order_keys=sealed_keys,new_order_keys=new_keys,
        current_batch_keys=current_batch_keys,validation_start_date=validation_start)
    if report['new_training_rows'] < 1:
        raise ValueError('No new labelled orders are in the training split; review the split before retraining')
    return x_train,y_train,x_val,y_val,dates,report


def train_member(member, prepared, batch_id, tracking_uri, hq=None, nithish=None, ephraim=None):
    if member not in ('han_qian','nithish','ephraim'):
        raise ValueError('Unknown team member')
    real_training = hq is None and nithish is None and ephraim is None
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
    runs = [{'run_id':row.get('run_id'),'model_uri':row.get('model_uri'),
             'model_name':row.get('name',row.get('version',row.get('model')))} for row in results]
    if real_training:
        if len(runs) != {'han_qian':2,'nithish':3,'ephraim':2}[member]:
            raise ValueError('Unexpected number of model runs')
        import mlflow
        add_run_links(runs, tracking_uri, mlflow.MlflowClient(tracking_uri=tracking_uri))
    return runs

