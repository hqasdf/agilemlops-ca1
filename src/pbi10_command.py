"""Connect PBI 10's components; caller supplies a verified key manifest."""
import os
import re
import tempfile
from pathlib import Path

from src.data_access import _key_hash
from src.pbi10_inputs import prepare_downloaded_batch
from src.pbi10_load import load_cleaned_batch
from src.pbi10_orchestrator import run_batch
from src.pbi10_read import read_tables
from src.pbi10_training import prepare_retraining, train_member


def batch_order_keys(cleaned):
    orders=cleaned['orders']
    return {_key_hash(row.country_code,row.order_id) for row in orders.itertuples()}


def read_new_keys(path):
    path=Path(path)
    if not path.is_file():
        raise ValueError('New-order ledger must already exist, even if empty')
    lines=path.read_text(encoding='utf-8').splitlines()
    if len(set(lines))!=len(lines) or any(not re.fullmatch(r'[0-9a-f]{64}',s) for s in lines):
        raise ValueError('New-order ledger is invalid')
    return set(lines)


def store_new_keys(path, keys):
    path=Path(path)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as handle:
        temporary=handle.name
        handle.write(''.join(key+'\n' for key in sorted(keys)))
    try:
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def execute_pipeline(connection, urls, batch_id, tracking_uri, sealed_keys, old_new_keys, ledger_path):
    current=set()
    def prepare():
        batch=prepare_downloaded_batch(urls)
        current.update(batch_order_keys(batch['cleaned']))
        if current.intersection(sealed_keys):
            raise ValueError('Incoming batch contains sealed historical order keys')
        return batch
    def load(batch):
        from src.pbi04_load import load_table
        def guarded_load(cursor, directory, table, columns):
            received, unchanged=load_table(cursor,directory,table,columns)
            if table=='orders' and unchanged and not current.issubset(old_new_keys):
                raise ValueError('Incoming orders overlap existing keys not verified in the new-order ledger')
            return received, unchanged
        result=load_cleaned_batch(connection,batch['cleaned'],load_table=guarded_load)
        store_new_keys(ledger_path,old_new_keys | current)
        return result
    def read_updated():
        return prepare_retraining(read_tables(connection),sealed_keys,old_new_keys | current,current)
    try:
        return run_batch(batch_id,tracking_uri,prepare,load,read_updated,train_member)
    finally:
        connection.close()

