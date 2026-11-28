import argparse
import unittest

from pbi10_pipeline import resolve_run_values
from src.pbi10_config import RUN_SETTINGS, ConfigError


def arguments(**values):
    empty = dict.fromkeys(('customers_url', 'orders_url', 'labels_url', 'batch_id', 'sealed_manifest',
                           'new_order_ledger'))
    return argparse.Namespace(**{**empty, **values})


class RunValueTests(unittest.TestCase):
    def setUp(self):
        self.settings = {name: name.lower() for name in RUN_SETTINGS}

    def test_values_come_from_settings_without_options(self):
        args = resolve_run_values(arguments(), self.settings)
        self.assertEqual(args.orders_url, 'ca1_orders_url')
        self.assertEqual(args.batch_id, 'ca1_batch_id')

    def test_option_overrides_setting(self):
        args = resolve_run_values(arguments(orders_url='from-option'), self.settings)
        self.assertEqual(args.orders_url, 'from-option')

    def test_blank_batch_id_becomes_timestamp(self):
        args = resolve_run_values(arguments(), {**self.settings, 'CA1_BATCH_ID': ''})
        self.assertRegex(args.batch_id, r'^batch-\d{8}-\d{6}$')

    def test_missing_values_are_named(self):
        with self.assertRaises(ConfigError) as caught:
            resolve_run_values(arguments(), {**self.settings, 'CA1_LABELS_URL': ''})
        self.assertIn('CA1_LABELS_URL', str(caught.exception))
        self.assertNotIn('CA1_ORDERS_URL', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
