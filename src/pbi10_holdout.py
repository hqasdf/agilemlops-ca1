"""Identify sealed historical order keys before accepting any new batch.

The caller must supply the complete unmodified original historical order
snapshot, not a database already containing new batches. Persist and verify the
manifest separately before using it for training; this module does not do so.
"""
import pandas as pd
from src.data_access import _key_hash


def build_sealed_keys(historical_orders):
    required={'country_code','order_id','order_date'}
    if not required.issubset(historical_orders.columns) or historical_orders.empty:
        raise ValueError('Complete historical order keys and dates required')
    dates=pd.to_datetime(historical_orders['order_date'],errors='coerce')
    if dates.isna().any() or historical_orders[['country_code','order_id']].isna().any().any():
        raise ValueError('Historical keys and dates must be valid')
    chosen=historical_orders.loc[dates >= pd.Timestamp('2023-04-14')]
    keys={_key_hash(row.country_code,row.order_id) for row in chosen.itertuples()}
    if not keys:
        raise ValueError('No sealed keys found; verify the historical source')
    return frozenset(keys)


def exclude_sealed_keys(orders, sealed_keys):
    if not sealed_keys:
        raise ValueError('A verified sealed-key manifest is required')
    if not {'country_code','order_id'}.issubset(orders.columns):
        raise ValueError('Order keys required')
    if orders[['country_code','order_id']].isna().any().any():
        raise ValueError('Missing order keys')
    hashes=[_key_hash(row.country_code,row.order_id) for row in orders.itertuples()]
    return orders.loc[[key not in sealed_keys for key in hashes]].copy()

