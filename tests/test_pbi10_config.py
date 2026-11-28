import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pbi10_config import OPTIONAL, REQUIRED, ConfigError, read_config, read_config_file, open_database


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.ca = str(Path(self.directory.name) / 'ca.pem')
        Path(self.ca).write_text('placeholder', encoding='utf-8')
        self.env = {'CA1_DB_HOST': 'db.example', 'CA1_DB_USER': 'user', 'CA1_DB_PASSWORD': 'S3cretValue',
                    'CA1_DB_NAME': 'amlops_ca1', 'CA1_DB_SSL_CA': self.ca,
                    'MLFLOW_TRACKING_URI': 'https://mlflow.example'}

    def test_missing_config_names_settings_and_how_to_fix(self):
        with self.assertRaises(ConfigError) as caught:
            read_config({})
        message = str(caught.exception)
        for name in ('CA1_DB_HOST', 'CA1_DB_PASSWORD', 'MLFLOW_TRACKING_URI', '--config', 'CA1_CONFIG_FILE'):
            self.assertIn(name, message)

    def test_missing_one_setting_names_only_that_setting(self):
        del self.env['CA1_DB_HOST']
        with self.assertRaises(ConfigError) as caught:
            read_config(self.env)
        self.assertIn('CA1_DB_HOST', str(caught.exception))
        self.assertNotIn('S3cretValue', str(caught.exception))

    def test_secret_never_appears_in_printed_config_or_errors(self):
        config = read_config(self.env)
        self.assertNotIn('S3cretValue', repr(config) + str(config))
        self.assertEqual(config['CA1_DB_PASSWORD'], 'S3cretValue')
        self.env['CA1_DB_PORT'] = 'abc'
        with self.assertRaises(ConfigError) as caught:
            read_config(self.env)
        self.assertNotIn('S3cretValue', str(caught.exception))

    def test_defaults_for_optional_settings(self):
        config = read_config(self.env)
        self.assertEqual(config['CA1_DB_PORT'], 3306)
        self.assertEqual(config['CA1_DOWNLOAD_TIMEOUT_SECONDS'], 30)

    def test_settings_file_supplies_values_and_environment_overrides_it(self):
        path = Path(self.directory.name) / 'settings.env'
        lines = [f'{key}={value}' for key, value in self.env.items() if key != 'CA1_DB_HOST']
        path.write_text('# comment\n' + '\n'.join(lines) + '\nexport CA1_DB_HOST="from-file.example"\nCA1_DB_PORT=3307\n', encoding='utf-8')
        config = read_config({}, config_file=str(path))
        self.assertEqual(config['CA1_DB_HOST'], 'from-file.example')
        self.assertEqual(config['CA1_DB_PORT'], 3307)
        config = read_config({'CA1_DB_HOST': 'from-env.example', 'CA1_DB_PORT': '3399'}, config_file=str(path))
        self.assertEqual(config['CA1_DB_HOST'], 'from-env.example')
        self.assertEqual(config['CA1_DB_PORT'], 3399)

    def test_changing_a_setting_changes_the_connection_without_code_edits(self):
        with patch('src.pbi10_config._connect') as connect:
            open_database(read_config({**self.env, 'CA1_DB_HOST': 'first.example', 'CA1_DB_PORT': '3306'}))
            open_database(read_config({**self.env, 'CA1_DB_HOST': 'second.example', 'CA1_DB_PORT': '3307'}))
        first, second = [call.kwargs for call in connect.call_args_list]
        self.assertEqual((first['host'], first['port']), ('first.example', 3306))
        self.assertEqual((second['host'], second['port']), ('second.example', 3307))

    def test_bad_file_line_and_missing_file_are_actionable(self):
        path = Path(self.directory.name) / 'bad.env'
        path.write_text('CA1_DB_PASSWORD S3cretValue\n', encoding='utf-8')
        with self.assertRaises(ConfigError) as caught:
            read_config_file(path)
        self.assertIn('line 1', str(caught.exception))
        self.assertNotIn('S3cretValue', str(caught.exception))
        with self.assertRaises(ConfigError):
            read_config({}, config_file=str(Path(self.directory.name) / 'absent.env'))

    def test_connection_verifies_ca_and_hostname(self):
        with patch('src.pbi10_config._connect') as connect:
            open_database(read_config(self.env))
        kw = connect.call_args.kwargs
        self.assertTrue(kw['ssl_verify_cert'])
        self.assertTrue(kw['ssl_verify_identity'])
        self.assertEqual(kw['ssl_ca'], self.ca)
        self.assertFalse(kw['autocommit'])

    def test_example_file_lists_every_setting_with_placeholders(self):
        example = read_config_file(Path(__file__).resolve().parents[1] / '.env.example')
        self.assertEqual(set(example), set(REQUIRED) | set(OPTIONAL))
        self.assertTrue(all(example.values()))
        for name in ('CA1_DB_HOST', 'CA1_DB_PASSWORD', 'MLFLOW_TRACKING_URI'):
            with self.subTest(name=name):
                self.assertIn('your', example[name])

    def test_invalid_values_are_rejected(self):
        for key, value in (('CA1_DB_PORT', '65536'), ('CA1_DOWNLOAD_TIMEOUT_SECONDS', '0'),
                           ('MLFLOW_TRACKING_URI', 'ftp://x'), ('MLFLOW_TRACKING_URI', 'https://u:p@x.example'),
                           ('CA1_DB_SSL_CA', '/no/such/ca.pem')):
            with self.subTest(key=key, value=value), self.assertRaises(ConfigError):
                read_config({**self.env, key: value})


if __name__ == '__main__':
    unittest.main()
