import functools
import http.server
import json
import contextlib
import io
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pbi10_inputs import prepare_downloaded_batch


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'customers.csv').write_text('country_code,customer_id,mobile_verified,num_orders_last_50days,num_cancelled_orders_last_50days,num_refund_orders_last_50days,total_payment_last_50days,num_associated_customers,first_order_datetime\nMY,0007,true,1,0,0,20,0,2023-01-01\n')
        (self.root / 'orders.csv').write_text('country_code,order_id,collect_type,payment_method,order_value,num_items_ordered,refund_value,order_date\nMY,0001,delivery,card,20,1,0,2023-03-01\n')
        (self.root / 'labels.csv').write_text('country_code,order_id,customer_id,is_fraud\nMY,0001,0007,0\n')
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(self.root)))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.temp.cleanup()

    def urls(self):
        return {n: f'http://127.0.0.1:{self.server.server_port}/{n}.csv' for n in ('customers', 'orders', 'labels')}

    def test_cleaned_inputs_preserve_identifiers(self):
        result = prepare_downloaded_batch(self.urls())
        self.assertEqual(result['cleaned']['customers'].iloc[0]['customer_id'], '0007')
        self.assertEqual(result['cleaned']['orders'].iloc[0]['order_id'], '0001')
        self.assertEqual(result['reports']['labels']['received'], 1)

    def test_bad_schema_fails(self):
        (self.root / 'labels.csv').write_text('country_code,order_id\nMY,0001\n')
        with self.assertRaisesRegex(ValueError, 'missing required'):
            prepare_downloaded_batch(self.urls())

    def test_missing_file_fails(self):
        urls = self.urls()
        urls['orders'] += '.missing'
        with self.assertRaises(ValueError):
            prepare_downloaded_batch(urls)

    def test_documented_command_runs_without_source_edits(self):
        from pbi10_validate import main
        urls = self.urls()
        args = ['pbi10_validate.py']
        for name, url in urls.items():
            args.extend([f'--{name}-url', url])
        output = io.StringIO()
        with patch.object(sys, 'argv', args), contextlib.redirect_stdout(output):
            main()
        summary = json.loads(output.getvalue())
        self.assertEqual(summary['status'], 'validated_only')
        self.assertEqual(summary['tables']['orders']['received'], 1)

