"""External resource settings for the pipeline and loaders (PBI 10 and PBI 11).

Settings come from environment variables, optionally backed by a KEY=VALUE
configuration file. A value in the environment always wins over the file. No
setting has a default that points at a real server or holds a credential.
"""
import os
from pathlib import Path
from urllib.parse import urlparse

REQUIRED = ('CA1_DB_HOST', 'CA1_DB_USER', 'CA1_DB_PASSWORD', 'CA1_DB_NAME', 'CA1_DB_SSL_CA', 'MLFLOW_TRACKING_URI')
OPTIONAL = {
    'CA1_DB_PORT': '3306',
    'CA1_DB_CONNECT_TIMEOUT_SECONDS': '10',
    'CA1_DOWNLOAD_TIMEOUT_SECONDS': '30',
    'CA1_DOWNLOAD_MAX_BYTES': '250000000',
}
# Values that change from one pipeline run to the next. Command-line options override them.
RUN_SETTINGS = ('CA1_CUSTOMERS_URL', 'CA1_ORDERS_URL', 'CA1_LABELS_URL', 'CA1_BATCH_ID',
                'CA1_SEALED_MANIFEST', 'CA1_NEW_ORDER_LEDGER')
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / '.env'
SECRETS = ('CA1_DB_PASSWORD',)
KNOWN = set(REQUIRED) | set(OPTIONAL) | {'CA1_CONFIG_FILE'}
HELP = ('Put KEY=VALUE lines in a .env file beside pbi10_pipeline.py, set them as environment variables, '
        'or pass another file with --config PATH or CA1_CONFIG_FILE=PATH. Copy .env.example for the full list of names.')


class ConfigError(ValueError):
    """A configuration problem. The message names settings but never shows their values."""


class Config(dict):
    """Settings dictionary whose printed form hides secrets."""
    def __repr__(self):
        return repr({key: ('********' if key in SECRETS else value) for key, value in self.items()})
    __str__ = __repr__


def read_config_file(path):
    path = Path(path)
    try:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
    except OSError:
        raise ConfigError(f'Config file {path} cannot be read. Check the path in --config or CA1_CONFIG_FILE.') from None
    values = {}
    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text or text.startswith('#'):
            continue
        if text.startswith('export '):
            text = text[7:].lstrip()
        name, separator, value = text.partition('=')
        name, value = name.strip(), value.strip()
        if not separator or not name:
            raise ConfigError(f'Config file {path.name} line {number} is not NAME=VALUE.')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        values[name] = value
    return values


def find_config_file(explicit=None, env=None, default=None):
    """The file named on the command line or in CA1_CONFIG_FILE, otherwise .env if it exists."""
    env = os.environ if env is None else env
    default = DEFAULT_ENV_FILE if default is None else Path(default)
    return explicit or env.get('CA1_CONFIG_FILE') or (default if default.is_file() else None)


def read_run_settings(env=None, config_file=None):
    """Per-run values from the settings file, overridden by the environment. Absent values are ''."""
    env = os.environ if env is None else env
    path = config_file or env.get('CA1_CONFIG_FILE')
    merged = read_config_file(path) if path else {}
    merged.update({key: value for key, value in env.items() if key in RUN_SETTINGS})
    return {key: merged.get(key, '').strip() for key in RUN_SETTINGS}


def read_config(env=None, config_file=None):
    env = os.environ if env is None else env
    path = config_file or env.get('CA1_CONFIG_FILE')
    merged = read_config_file(path) if path else {}
    merged.update({key: value for key, value in env.items() if key in KNOWN or key.startswith('MLFLOW_')})
    missing = [key for key in REQUIRED if not merged.get(key)]
    if missing:
        raise ConfigError('Missing required settings: ' + ', '.join(missing) + '. ' + HELP)
    for key, default in OPTIONAL.items():
        merged.setdefault(key, default)
    for key in ('CA1_DB_PORT', 'CA1_DB_CONNECT_TIMEOUT_SECONDS', 'CA1_DOWNLOAD_TIMEOUT_SECONDS', 'CA1_DOWNLOAD_MAX_BYTES'):
        try:
            merged[key] = int(merged[key])
        except ValueError:
            raise ConfigError(f'{key} must be a whole number.') from None
    if not 1 <= merged['CA1_DB_PORT'] <= 65535:
        raise ConfigError('CA1_DB_PORT must be between 1 and 65535.')
    for key in ('CA1_DB_CONNECT_TIMEOUT_SECONDS', 'CA1_DOWNLOAD_TIMEOUT_SECONDS', 'CA1_DOWNLOAD_MAX_BYTES'):
        if merged[key] < 1:
            raise ConfigError(f'{key} must be a positive whole number.')
    url = urlparse(merged['MLFLOW_TRACKING_URI'])
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password:
        raise ConfigError('MLFLOW_TRACKING_URI must be an HTTP(S) URL without credentials, e.g. https://mlflow.example.com.')
    if not Path(merged['CA1_DB_SSL_CA']).is_file():
        raise ConfigError('CA1_DB_SSL_CA must be the path of an existing CA certificate file.')
    return Config({key: merged[key] for key in (*REQUIRED, *OPTIONAL)})


def _connect(**kwargs):
    import pymysql
    return pymysql.connect(**kwargs)


def open_database(config):
    return _connect(host=config['CA1_DB_HOST'], port=config['CA1_DB_PORT'],
                    user=config['CA1_DB_USER'], password=config['CA1_DB_PASSWORD'],
                    database=config['CA1_DB_NAME'], ssl_ca=config['CA1_DB_SSL_CA'],
                    ssl_verify_cert=True, ssl_verify_identity=True, charset='utf8mb4',
                    autocommit=False,
                    connect_timeout=config.get('CA1_DB_CONNECT_TIMEOUT_SECONDS', 10))
