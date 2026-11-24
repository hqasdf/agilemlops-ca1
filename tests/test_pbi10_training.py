import unittest
from unittest.mock import patch
from src.pbi10_training import train_member, prepare_retraining


class TrainingAdapterTests(unittest.TestCase):
    def test_run_links_use_actual_experiment_metadata(self):
        from src.pbi10_training import add_run_links
        class Client:
            def get_run(self, run_id):
                from types import SimpleNamespace
                return SimpleNamespace(info=SimpleNamespace(experiment_id='42'))
        runs=[{'run_id':'abc','model_uri':'models:/example'}]
        linked=add_run_links(runs,'http://tracking.example/',Client())
        self.assertEqual(linked[0]['run_link'],'http://tracking.example/#/experiments/42/runs/abc')
    def test_refuses_retraining_when_no_new_record_is_used(self):
        split=('xtrain','ytrain','xval','yval',None,None,{'new_training_rows':0},'dates')
        with patch('src.feature_engineering.prepare_training_data',return_value=split), patch('src.pbi10_training.choose_batch_validation_start',return_value='2026-01-01'):
            with self.assertRaisesRegex(ValueError,'new labelled'):
                prepare_retraining({'customers':1,'orders':2,'labels':3},{'sealed'},{'new'},{'new'})

    def test_prepare_retraining_passes_verified_keys(self):
        split=('xt','yt','xv','yv',None,None,{'new_training_rows':1},'dates')
        with patch('src.feature_engineering.prepare_training_data',return_value=split) as prepare, patch('src.pbi10_training.choose_batch_validation_start',return_value='2026-01-01'):
            result=prepare_retraining({'customers':1,'orders':2,'labels':3},{'sealed'},{'new'},{'new'})
        self.assertEqual(result,('xt','yt','xv','yv','dates',{'new_training_rows':1}))
        self.assertEqual(prepare.call_args.kwargs['new_order_keys'],{'new'})
    def test_calls_each_member_with_prepared_data(self):
        calls=[]
        def hq(*args,**kw): calls.append(('hq',args,kw)); return [{'run_id':'1','model_uri':'m1'}]
        def nit(*args,**kw): calls.append(('nit',args,kw)); return [{'run_id':'2','model_uri':'m2'}]
        def eph(*args,**kw): calls.append(('eph',args,kw)); return [{'run_id':'3','model_uri':'m3'}]
        prepared=('xtrain','ytrain','xval','yval','dates',{'new_training_indices':[1]})
        for name in ('han_qian','nithish','ephraim'):
            result=train_member(name,prepared,'batch1','http://server',hq,nit,eph)
            self.assertTrue(result[0]['run_id'])
            self.assertTrue(result[0]['model_uri'])
        self.assertEqual([x[0] for x in calls],['hq','nit','eph'])
        self.assertEqual(calls[1][2]['prepared_data'],prepared)
        self.assertEqual(calls[1][2]['current_batch_indices'],[1])
        self.assertEqual(calls[0][2]['batch_name'],'batch1')

    def test_unknown_member_rejected(self):
        with self.assertRaises(ValueError): train_member('unknown',(), 'b','http://server',None,None,None)

