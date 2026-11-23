import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.train_ephraim import train_ephraim_models


class EphraimTrainingTests(unittest.TestCase):
    def test_logs_model_artifacts_for_each_version(self):
        data=pd.DataFrame({'order_value':[1,3,5,7,9,11,13,15],'num_items_ordered':[1]*8,
          'order_weekday':[0,1,2,3,4,5,6,0], 'order_month':[1]*8, 'country_code':['MY']*8,
          'collect_type':['delivery']*8,'payment_method':['card']*8,'days_since_first_order':[1,2,3,4,5,6,7,8]})
        labels=pd.Series([0,1,0,1,0,1,0,1])
        mlflow=MagicMock()
        mlflow.__version__='3.18.0'
        mlflow.start_run.return_value.__enter__.return_value.info.run_id='run-1'
        with patch.dict('sys.modules',{'mlflow':mlflow}):
            out=train_ephraim_models(data.iloc[:6],labels.iloc[:6],data.iloc[6:],labels.iloc[6:],tracking_uri='http://mlflow')
        self.assertEqual(mlflow.sklearn.log_model.call_count,2)
        self.assertTrue(all(r['model_uri'] for r in out))
    def test_trains_both_models_from_given_input(self):
        data=pd.DataFrame({'order_value':[1,3,5,7,9,11,13,15],
          'num_items_ordered':[1,2,1,2,1,2,1,2], 'order_weekday':[0,1,2,3,4,5,6,0],
          'order_month':[1]*8, 'country_code':['MY']*8, 'collect_type':['delivery']*8,
          'payment_method':['card']*8, 'days_since_first_order':[1,2,3,4,5,6,7,8]})
        labels=pd.Series([0,1,0,1,0,1,0,1])
        out=train_ephraim_models(data.iloc[:6],labels.iloc[:6],data.iloc[6:],labels.iloc[6:])
        self.assertEqual({r['name'] for r in out},{'dt_tuned','rf_tuned'})
        self.assertTrue(all('f1' in r['metrics'] for r in out))

