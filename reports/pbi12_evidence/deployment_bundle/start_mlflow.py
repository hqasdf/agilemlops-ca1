"""Start MLflow with an RDS backend and proxied S3 artifacts."""
import os
import sys
from urllib.parse import quote, urlencode, urlsplit


def make_environment():
    env = os.environ.copy()
    required = ('DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME',
                'MLFLOW_ARTIFACTS_DESTINATION', 'MLFLOW_ALLOWED_HOSTS')
    missing = [name for name in required if not env.get(name)]
    if missing:
        raise ValueError('Missing required environment settings: ' + ', '.join(missing))
    host = env['DB_HOST']
    if any(c in host for c in '/@?#: ') or not host.endswith('.rds.amazonaws.com'):
        raise ValueError('DB_HOST must be an RDS hostname without a port or URL scheme')
    location = urlsplit(env['MLFLOW_ARTIFACTS_DESTINATION'])
    if location.scheme != 's3' or not location.netloc or location.query or location.fragment:
        raise ValueError('MLFLOW_ARTIFACTS_DESTINATION must be an s3:// bucket URI')
    for key, default in (('PORT', '5000'), ('DB_PORT', '3306')):
        value = env.get(key, default)
        if not value.isdigit() or not 1 <= int(value) <= 65535:
            raise ValueError(key + ' must be a valid port number')
        env[key] = value
    tls = urlencode({'charset': 'utf8mb4', 'ssl_ca': '/app/certs/rds-bundle.pem',
                     'ssl_check_hostname': 'true'})
    env['MLFLOW_BACKEND_STORE_URI'] = (
        'mysql+pymysql://' + quote(env['DB_USER'], safe='') + ':'
        + quote(env['DB_PASSWORD'], safe='') + '@' + host + ':'
        + env['DB_PORT'] + '/' + quote(env['DB_NAME'], safe='') + '?' + tls
    )
    env['MLFLOW_SERVER_ALLOWED_HOSTS'] = env['MLFLOW_ALLOWED_HOSTS']
    return env


def main():
    try:
        env = make_environment()
    except ValueError as error:
        print('MLflow configuration error:', error, file=sys.stderr)
        return 1
    os.execvpe('mlflow', ['mlflow', 'server', '--host', '0.0.0.0',
                          '--port', env['PORT'], '--serve-artifacts'], env)


if __name__ == '__main__':
    sys.exit(main())
