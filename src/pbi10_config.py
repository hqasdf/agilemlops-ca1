"""External connection settings for the PBI 10 pipeline."""
import os
from urllib.parse import urlparse

REQUIRED=('CA1_DB_HOST','CA1_DB_USER','CA1_DB_PASSWORD','CA1_DB_NAME','CA1_DB_SSL_CA','MLFLOW_TRACKING_URI')


def read_config(env=None):
    env=os.environ if env is None else env
    missing=[key for key in REQUIRED if not env.get(key)]
    if missing:
        raise ValueError('Missing required settings: '+', '.join(missing))
    try: port=int(env.get('CA1_DB_PORT','3306'))
    except ValueError: raise ValueError('CA1_DB_PORT must be an integer') from None
    if not 1 <= port <= 65535: raise ValueError('CA1_DB_PORT is out of range')
    url=urlparse(env['MLFLOW_TRACKING_URI'])
    if url.scheme not in ('http','https') or not url.hostname or url.username or url.password:
        raise ValueError('MLFLOW_TRACKING_URI must be an HTTP(S) URL without credentials')
    return {**{key:env[key] for key in REQUIRED},'CA1_DB_PORT':port}


def _connect(**kwargs):
    import pymysql
    return pymysql.connect(**kwargs)


def open_database(config):
    return _connect(host=config['CA1_DB_HOST'],port=config['CA1_DB_PORT'],
                    user=config['CA1_DB_USER'],password=config['CA1_DB_PASSWORD'],
                    database=config['CA1_DB_NAME'],ssl_ca=config['CA1_DB_SSL_CA'],
                    ssl_verify_cert=True,ssl_verify_identity=True,charset='utf8mb4',
                    autocommit=False,connect_timeout=10)

