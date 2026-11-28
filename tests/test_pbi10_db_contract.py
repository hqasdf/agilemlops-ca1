import unittest
import pandas as pd

from src.pbi10_db_contract import validate_db_contract


class DatabaseContractTests(unittest.TestCase):
    def test_rejects_null_required_order_value(self):
        row = pd.DataFrame([{'country_code':'MY','order_id':'o1','collect_type':'delivery','payment_method':None,'order_value':None,'num_items_ordered':1,'refund_value':0,'order_date':'2023-01-01'}])
        with self.assertRaisesRegex(ValueError, 'order_value'):
            validate_db_contract('orders', row)

    def test_rejects_oversized_key(self):
        row = pd.DataFrame([{'country_code':'MY','order_id':'x'*33,'customer_id':'c1','is_fraud':0}])
        with self.assertRaisesRegex(ValueError, 'order_id'):
            validate_db_contract('labels', row)

    def test_nullable_fields_allowed(self):
        row = pd.DataFrame([{'country_code':'MY','order_id':'o1','collect_type':'delivery','payment_method':None,'order_value':10,'num_items_ordered':None,'refund_value':0,'order_date':'2023-01-01'}])
        self.assertEqual(validate_db_contract('orders', row), 1)

