import unittest
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from src.pbi10_holdout import build_sealed_keys, exclude_sealed_keys, read_sealed_manifest


class HoldoutTests(unittest.TestCase):
    def test_manifest_requires_correct_hashes(self):
        with TemporaryDirectory() as root:
            path=Path(root)/'holdout.txt'
            path.write_text('invalid\n')
            with self.assertRaises(ValueError): read_sealed_manifest(path)
            path.write_text('a'*64+'\n'+'b'*64+'\n')
            self.assertEqual(len(read_sealed_manifest(path)),2)
    def test_sealed_historical_keys_excluded_new_dates_not_excluded(self):
        original=pd.DataFrame([{'country_code':'MY','order_id':'old','order_date':'2023-04-15'},
                               {'country_code':'MY','order_id':'train','order_date':'2023-02-01'}])
        sealed=build_sealed_keys(original)
        updated=pd.concat([original,pd.DataFrame([{'country_code':'MY','order_id':'new','order_date':'2026-11-01'}])],ignore_index=True)
        self.assertEqual(set(exclude_sealed_keys(updated,sealed)['order_id']),{'train','new'})

    def test_same_order_id_different_country_not_removed(self):
        original=pd.DataFrame([{'country_code':'MY','order_id':'x','order_date':'2023-05-01'}])
        updated=pd.DataFrame([{'country_code':'SG','order_id':'x','order_date':'2026-01-01'}])
        self.assertEqual(len(exclude_sealed_keys(updated,build_sealed_keys(original))),1)

    def test_missing_manifest_fails_closed(self):
        with self.assertRaises(ValueError): exclude_sealed_keys(pd.DataFrame(),None)

