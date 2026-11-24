import unittest
from src.pbi10_orchestrator import run_batch


class PipelineTests(unittest.TestCase):
    def test_all_three_training_results_reported(self):
        calls=[]
        def train(member, tables, batch, tracking):
            calls.append(member)
            return [{'run_id':member+'-run', 'model_uri':'model'}]
        result=run_batch('batch1', 'https://mlflow.example', lambda:{'cleaned':'batch'},
                         lambda batch:{'orders':{'inserted':1}}, lambda:'updated_tables', train)
        self.assertEqual(calls,['han_qian','nithish','ephraim'])
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['models']['ephraim']['runs'][0]['run_id'],'ephraim-run')

    def test_training_failure_does_not_hide_other_members(self):
        def train(member,*args):
            if member=='nithish': raise RuntimeError('unavailable')
            return [{'run_id':'id','model_uri':'model'}]
        result=run_batch('batch1','https://mlflow.example',lambda:{},lambda x:{},lambda:'tables',train)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['models']['nithish']['status'],'failed')
        self.assertEqual(result['models']['ephraim']['status'],'success')

    def test_unlogged_model_is_not_success(self):
        result=run_batch('batch1','https://mlflow.example',lambda:{},lambda x:{},lambda:'tables',lambda *args:[{'run_id':None}])
        self.assertEqual(result['status'],'failed')

    def test_missing_tracking_fails_before_loading(self):
        events=[]
        with self.assertRaises(ValueError):
            run_batch('batch1','',lambda:events.append('prepare'),lambda x:None,lambda:None,lambda *x:None)
        self.assertEqual(events,[])

