import unittest
from src.pbi10_read import read_tables


class Cursor:
    def __init__(self): self.description=[]; self.calls=[]; self.rows=[]
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def execute(self,sql):
        self.calls.append(sql)
        self.description=[('country_code',),('order_id',)]
        self.rows=[('MY','0001')]
    def fetchmany(self, size):
        rows=self.rows
        self.rows=[]
        return rows

class Connection:
    def __init__(self): self.c=Cursor()
    def cursor(self): return self.c

class ReadTests(unittest.TestCase):
    def test_reads_all_three_tables_without_losing_ids(self):
        c=Connection(); result=read_tables(c)
        self.assertEqual(set(result),{'customers','orders','labels'})
        self.assertEqual(result['orders'].iloc[0]['order_id'],'0001')
        self.assertEqual(len(c.c.calls),3)
        self.assertTrue(all(sql.startswith('SELECT ') for sql in c.c.calls))

