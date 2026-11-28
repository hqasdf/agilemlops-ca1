import unittest
import pandas as pd
from unittest.mock import patch
from src.pbi10_training import choose_batch_validation_start, prepare_retraining
from src.data_access import _key_hash


class SplitTests(unittest.TestCase):
    def test_current_batch_appears_in_both_temporal_partitions(self):
        orders=pd.DataFrame([{'country_code':'ZZ','order_id':str(i),'order_date':f'2026-01-0{i}', 'order_value':10,'num_items_ordered':1,'collect_type':'delivery','payment_method':'card'} for i in range(1,6)])
        labels=pd.DataFrame([{'country_code':'ZZ','order_id':str(i),'customer_id':'c','is_fraud':i%2} for i in range(1,6)])
        customers=pd.DataFrame([{'country_code':'ZZ','customer_id':'c','first_order_datetime':'2025-01-01'}])
        keys={_key_hash('ZZ',str(i)) for i in range(1,6)}
        with patch('src.feature_engineering._dates',side_effect=lambda v,c:pd.to_datetime(v,errors='coerce')):
            prepared=prepare_retraining({'orders':orders,'customers':customers,'labels':labels},{_key_hash('ZZ','old')},keys,keys)
        self.assertEqual(prepared[-1]['new_training_rows'],4)
        self.assertEqual(prepared[-1]['new_validation_rows'],1)
    def test_batch_dates_produce_later_validation(self):
        orders=pd.DataFrame([{'country_code':'ZZ','order_id':str(i),'order_date':f'2026-01-0{i}'} for i in range(1,6)])
        labels=orders[['country_code','order_id']].copy()
        keys={_key_hash('ZZ',str(i)) for i in range(1,6)}
        self.assertEqual(choose_batch_validation_start(orders,labels,keys),pd.Timestamp('2026-01-05'))

    def test_single_date_fails_with_clear_error(self):
        orders=pd.DataFrame([{'country_code':'ZZ','order_id':'a','order_date':'2026-01-01'}])
        labels=orders[['country_code','order_id']].copy()
        with self.assertRaisesRegex(ValueError,'two distinct'):
            choose_batch_validation_start(orders,labels,{_key_hash('ZZ','a')})

