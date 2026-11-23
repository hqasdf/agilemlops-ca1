import unittest
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from src.pbi10_holdout import build_sealed_keys, exclude_sealed_keys, read_sealed_manifest
from src.data_access import _key_hash
from src.feature_engineering import prepare_training_data


class HoldoutTests(unittest.TestCase):
    def test_newer_orders_require_verified_new_key_allowlist(self):
        customers=pd.DataFrame([{'country_code':'MY','customer_id':'c1','first_order_datetime':'2022-01-01'}])
        orders=pd.DataFrame([{'country_code':'MY','order_id':k,'order_date':d,'order_value':10,'num_items_ordered':1,'collect_type':'delivery','payment_method':'card'} for k,d in [('a','2023-01-01'),('b','2023-02-01'),('held','2023-05-01'),('new','2026-01-01')]])
        labels=pd.DataFrame([{'country_code':'MY','order_id':k,'customer_id':'c1','is_fraud':n%2} for n,k in enumerate(['a','b','held','new'])])
        sealed={_key_hash('MY','held')}
        with patch('src.feature_engineering._dates',side_effect=lambda values, column:pd.to_datetime(values,errors='coerce')):
            with self.assertRaises(ValueError):
                prepare_training_data(customers,orders,labels,sealed_order_keys=sealed,new_order_keys=set())
            result=prepare_training_data(customers,orders,labels,sealed_order_keys=sealed,new_order_keys={_key_hash('MY','new')})
        self.assertEqual(result[6]['training_rows']+result[6]['validation_rows'],3)
        self.assertEqual(result[6]['new_training_rows'],0)
        self.assertEqual(result[6]['new_validation_rows'],1)
    def test_manifest_requires_correct_hashes(self):
        with TemporaryDirectory() as root:
            path=Path(root)/'holdout.txt'
            path.write_text('invalid\n')
            with self.assertRaises(ValueError): read_sealed_manifest(path)
            path.write_text('a'*64+'\n'+'b'*64+'\n')
            self.assertEqual(len(read_sealed_manifest(path)),2)
    def test_sealed_historical_keys_excluded_new_dates_not_excluded(self):
        original=pd.DataFrame([{'country_code':'MY','order_id':'old','order_date':'2023-04-15'},
                               {'country_code':'MY','order_id':'train','order_date':'2023-02-01'}])
        sealed=build_sealed_keys(original)
        updated=pd.concat([original,pd.DataFrame([{'country_code':'MY','order_id':'new','order_date':'2026-11-01'}])],ignore_index=True)
        self.assertEqual(set(exclude_sealed_keys(updated,sealed)['order_id']),{'train','new'})

    def test_same_order_id_different_country_not_removed(self):
        original=pd.DataFrame([{'country_code':'MY','order_id':'x','order_date':'2023-05-01'}])
        updated=pd.DataFrame([{'country_code':'SG','order_id':'x','order_date':'2026-01-01'}])
        self.assertEqual(len(exclude_sealed_keys(updated,build_sealed_keys(original))),1)

    def test_missing_manifest_fails_closed(self):
        with self.assertRaises(ValueError): exclude_sealed_keys(pd.DataFrame(),None)

