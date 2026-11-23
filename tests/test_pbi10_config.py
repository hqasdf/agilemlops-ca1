import unittest
from unittest.mock import patch
from src.pbi10_config import read_config, open_database


class ConfigTests(unittest.TestCase):
    def test_missing_config_fails_without_disclosing_values(self):
        with self.assertRaisesRegex(ValueError,'CA1_DB_HOST'):
            read_config({})

    def test_connection_verifies_ca_and_hostname(self):
        env={'CA1_DB_HOST':'db.example','CA1_DB_USER':'user','CA1_DB_PASSWORD':'secret',
             'CA1_DB_NAME':'amlops_ca1','CA1_DB_SSL_CA':'ca.pem','MLFLOW_TRACKING_URI':'https://mlflow.example'}
        cfg=read_config(env)
        with patch('src.pbi10_config._connect') as connect:
            open_database(cfg)
        kw=connect.call_args.kwargs
        self.assertTrue(kw['ssl_verify_cert'])
        self.assertTrue(kw['ssl_verify_identity'])
        self.assertEqual(kw['ssl_ca'],'ca.pem')
        self.assertFalse(kw['autocommit'])

    def test_invalid_port_is_rejected(self):
        env={'CA1_DB_HOST':'db','CA1_DB_USER':'user','CA1_DB_PASSWORD':'secret','CA1_DB_NAME':'x',
             'CA1_DB_SSL_CA':'ca.pem','MLFLOW_TRACKING_URI':'https://mlflow.example','CA1_DB_PORT':'65536'}
        with self.assertRaises(ValueError): read_config(env)

