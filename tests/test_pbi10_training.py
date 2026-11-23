import unittest
from src.pbi10_training import train_member


class TrainingAdapterTests(unittest.TestCase):
    def test_calls_each_member_with_prepared_data(self):
        calls=[]
        def hq(*args,**kw): calls.append(('hq',args,kw)); return [{'run_id':'1','model_uri':'m1'}]
        def nit(*args,**kw): calls.append(('nit',args,kw)); return [{'run_id':'2','model_uri':'m2'}]
        def eph(*args,**kw): calls.append(('eph',args,kw)); return [{'run_id':'3','model_uri':'m3'}]
        prepared=('xtrain','ytrain','xval','yval','dates','report')
        for name in ('han_qian','nithish','ephraim'):
            result=train_member(name,prepared,'batch1','http://server',hq,nit,eph)
            self.assertTrue(result[0]['run_id'])
            self.assertTrue(result[0]['model_uri'])
        self.assertEqual([x[0] for x in calls],['hq','nit','eph'])
        self.assertEqual(calls[1][2]['prepared_data'],prepared)
        self.assertEqual(calls[0][2]['batch_name'],'batch1')

    def test_unknown_member_rejected(self):
        with self.assertRaises(ValueError): train_member('unknown',(), 'b','http://server',None,None,None)

