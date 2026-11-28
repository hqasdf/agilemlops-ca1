import importlib
import sys
import types
import unittest
from unittest.mock import patch
import pandas as pd


class NithishFitTests(unittest.TestCase):
    def test_current_rows_in_fit_using_historical_calibration(self):
        ml=types.ModuleType('mlflow'); ml.__version__='test'; ml.__path__=[]
        mlsk=types.ModuleType('mlflow.sklearn'); ml.sklearn=mlsk
        with patch.dict(sys.modules,{'mlflow':ml,'mlflow.sklearn':mlsk}):
            module=importlib.import_module('src.train_nithish')
            n=24
            x=pd.DataFrame({'order_value':list(range(1,n+1)), 'num_items_ordered':[1]*n,
                'order_weekday':[i%7 for i in range(n)],'order_month':[1]*n,
                'country_code':['MY']*n,'collect_type':['delivery']*n,'payment_method':['card']*n,
                'days_since_first_order':list(range(1,n+1))})
            y=pd.Series([i%2 for i in range(n)])
            xv=x.iloc[-4:].copy(); xv.index=range(24,28)
            yv=pd.Series([0,1,0,1],index=xv.index)
            dates=pd.Series(pd.date_range('2023-01-01',periods=n),index=x.index)
            prepared=(x,y,xv,yv,dates,{'training_rows':n,'validation_rows':4})
            out=module.train_nithish_models(None,None,None,log=False,versions=['v1_logistic_default'],
                  prepared_data=prepared,current_batch_indices=[20,21,22,23])
            self.assertGreater(out[0]['model_fit_rows'],0)
            self.assertEqual(out[0]['threshold_calibration_rows'],4)
        sys.modules.pop('src.train_nithish',None)

