import unittest
import pandas as pd
from src.pbi10_load import load_cleaned_batch
from src.pbi04_load import load_table


class Connection:
    def __init__(self): self.events=[]; self.rowcount=0
    def cursor(self): return self
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def execute(self, query, params=None):
        self.events.append(query)
        if query.startswith('INSERT INTO ') and 'SELECT' in query: self.rowcount=1
    def fetchone(self): return (0,0)
    def begin(self): self.events.append('BEGIN')
    def commit(self): self.events.append('COMMIT')
    def rollback(self): self.events.append('ROLLBACK')


class LoadTests(unittest.TestCase):
    def test_actual_pbi04_loader_is_importable(self):
        self.assertTrue(callable(load_table))

    def test_actual_staging_loader_integrated(self):
        c=Connection()
        result=load_cleaned_batch(c,self.frames())
        self.assertEqual(result['labels']['inserted'],1)
        self.assertTrue(any('CREATE TEMPORARY TABLE pbi04_stage_orders' in x for x in c.events))
        self.assertEqual(c.events[-1],'COMMIT')
    def frames(self):
        return {'customers':pd.DataFrame([{'country_code':'MY','customer_id':'c1','mobile_verified':None,'num_orders_last_50days':1,'num_cancelled_orders_last_50days':0,'num_refund_orders_last_50days':0,'total_payment_last_50days':10,'num_associated_customers':0,'first_order_datetime':None}]),
                'orders':pd.DataFrame([{'country_code':'MY','order_id':'o1','collect_type':'delivery','payment_method':None,'order_value':10,'num_items_ordered':1,'refund_value':0,'order_date':'2023-01-01'}]),
                'labels':pd.DataFrame([{'country_code':'MY','order_id':'o1','customer_id':'c1','is_fraud':0}])}

    def test_commits_only_after_all_tables(self):
        c=Connection(); calls=[]
        def load(cursor,directory,table,columns):
            self.assertTrue((directory / (table+'_cleaned.csv')).exists())
            calls.append(table); return (1,0)
        self.assertEqual(load_cleaned_batch(c,self.frames(),load)['orders']['inserted'],1)
        self.assertEqual(calls,['customers','orders','labels'])
        self.assertEqual(c.events[-1],'COMMIT')

    def test_conflict_rolls_back(self):
        c=Connection()
        def load(cursor,directory,table,columns):
            if table=='orders': raise ValueError('conflict')
            return (1,0)
        with self.assertRaisesRegex(ValueError,'conflict'): load_cleaned_batch(c,self.frames(),load)
        self.assertEqual(c.events[-1],'ROLLBACK')

    def test_invalid_batch_never_opens_transaction(self):
        c=Connection(); frames=self.frames(); frames['orders'].loc[0,'order_value']=None
        with self.assertRaises(ValueError): load_cleaned_batch(c,frames,lambda *args:(1,0))
        self.assertEqual(c.events,[])

