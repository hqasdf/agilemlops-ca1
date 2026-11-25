# Deployed bundle for `mlflow-pbi12`

These are the files from the Elastic Beanstalk application version `pbi12-mlflow-rds-s3-v1`
(`1795342950035-pbi12-mlflow-rds-s3.zip`, 4,573 bytes, created 22 Nov 2026 18:22 Singapore time),
read back from the copy that Elastic Beanstalk stored. They are what ran on `mlflow-pbi12`.
The zip holds four files; each is kept here with its SHA-256 so it can be checked.

| Path inside the zip | Path in this folder | Bytes | SHA-256 |
| --- | --- | --- | --- |
| `Dockerfile` | `Dockerfile` | 341 | `4827674061e7be4898743e7dfaad9329193a6e52088792e17a2e152967133694` |
| `start_mlflow.py` | `start_mlflow.py` | 2031 | `2f9235d3b91f4947a58d8a458c42f2bac51f5a4e3684d6e4f65bf4ac49fc6182` |
| `certs/rds-bundle.pem` | `certs/rds-bundle.pem` | 4576 | `b1711d12bae51838581281e23b6cb97b1074016873b4dafc80ed14002462dd77` |
| `.platform/nginx/conf.d/upload-limit.conf` | `upload-limit.conf` | 27 | `c48c9c3c352c78a7a7f7081622e45962ff1c5a958021a6c7c313ffed85f7be2e` |

To rebuild a zip with the same contents, copy `upload-limit.conf` to `.platform/nginx/conf.d/upload-limit.conf`
and zip the four files with those paths at the top level of the zip. The zip's own bytes will differ (file dates),
but the four files inside are identical.

No password or key is in this folder. The database login and the S3 bucket name are set as Elastic Beanstalk
environment properties (see `../../PBI12_MLFLOW_STORAGE.md`). `rds-bundle.pem` holds public AWS certificates only.
