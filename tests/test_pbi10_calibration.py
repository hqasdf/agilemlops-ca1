import unittest
import pandas as pd
from src.model_evaluation import historical_calibration_mask


class CalibrationTests(unittest.TestCase):
    def test_current_batch_is_in_fit_not_calibration(self):
        dates=pd.Series(pd.date_range('2023-01-01',periods=10))
        mask=historical_calibration_mask(dates,[8,9])
        self.assertEqual(list(mask[mask].index),[6,7])
        self.assertFalse(mask.loc[8])
        self.assertFalse(mask.loc[9])

    def test_requires_historical_and_current_rows(self):
        dates=pd.Series(pd.date_range('2023-01-01',periods=2))
        with self.assertRaises(ValueError): historical_calibration_mask(dates,[0,1])

