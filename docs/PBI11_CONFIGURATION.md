# PBI 11 - external connection settings

Data links, database and MLflow settings are read at run time from a `.env` file or environment variables. No server name, password or URL is held in the Python source used by the pipeline (`pbi10_pipeline.py`, `src/pbi10_config.py`) or the PBI 04 loader (`src/pbi04_load.py`).

## Settings

| Name | Required | Meaning |
| --- | --- | --- |
| `CA1_DB_HOST` | yes | Database host name |
| `CA1_DB_PORT` | no (3306) | Database port |
| `CA1_DB_NAME` | yes | Schema / database name |
| `CA1_DB_USER` | yes | Database user |
| `CA1_DB_PASSWORD` | yes | Database password (never printed) |
| `CA1_DB_SSL_CA` | yes | Path to the RDS CA certificate; certificate and hostname are verified |
| `MLFLOW_TRACKING_URI` | yes | Shared MLflow `http(s)://` URL, without credentials |
| `CA1_DB_CONNECT_TIMEOUT_SECONDS` | no (10) | Database connect timeout |
| `CA1_DOWNLOAD_TIMEOUT_SECONDS` | no (30) | Per-file download timeout |
| `CA1_DOWNLOAD_MAX_BYTES` | no (250000000) | Maximum size of each downloaded CSV |

## Run settings (change these for each batch)

| Name | Required | Meaning |
| --- | --- | --- |
| `CA1_CUSTOMERS_URL` | yes | Public customers CSV link |
| `CA1_ORDERS_URL` | yes | Public orders CSV link |
| `CA1_LABELS_URL` | yes | Public labels CSV link |
| `CA1_BATCH_ID` | no (timestamp) | Name used for the MLflow runs |
| `CA1_SEALED_MANIFEST` | yes | Path to the sealed historical-order manifest |
| `CA1_NEW_ORDER_LEDGER` | yes | Path to the new-order ledger |

## Demonstration day

Edit the three `_URL` lines in `.env`, save, then run:

```powershell
$env:PYTHONUTF8 = "1"
python pbi10_pipeline.py
```

No Python file is edited. `python pbi10_validate.py` reads the same links and checks them without writing to the database.

The order of priority is: command-line option, environment variable, settings file. The settings file is `--config`, else `CA1_CONFIG_FILE`, else `.env` beside `pbi10_pipeline.py`.

## Set up

1. Copy `.env.example` to `.env` beside `pbi10_pipeline.py` (git ignores it) and fill in your values. A file kept elsewhere also works.
2. `.env` is read automatically. To use another file pass `--config PATH` or `CA1_CONFIG_FILE=PATH`.
3. If a name is in both the file and the environment, the environment wins. Command-line options override both.

```bash
python pbi10_pipeline.py                      # everything from .env
python pbi10_pipeline.py --config ~/ca1.env   # everything from another file
```

The PBI 04 loader reads the same names as defaults for `--host`, `--port`, `--user`, `--database` and `--ssl-ca`; the password comes from `CA1_DB_PASSWORD` or a hidden prompt.

## Windows

Set `PYTHONUTF8=1` before running the pipeline on Windows. Without it the MLflow run messages contain emoji that the default cp1252 console encoding cannot print, and the Han Qian and Ephraim training steps fail with `UnicodeEncodeError` (seen in a full-history test on 23 Nov 2026).

## Errors and secrets

- A missing setting stops the run before any connection or download, and the message lists the missing names and how to set them. Example: `Missing required settings: CA1_DB_HOST, MLFLOW_TRACKING_URI. Set them as environment variables, or put KEY=VALUE lines in a config file ...`
- Messages name settings, never values. The loaded configuration object prints the password as `********`.
- Keep real files out of git. `.env`, `.env.*` (except `.env.example`) and `*.pem` are in `.gitignore`.

## Notebooks

`KochuveettilTojiEphraimModels.ipynb` reads `MLFLOW_TRACKING_URI` from the environment and stops with a message naming the variable if it is missing. Set it in the shell that starts Jupyter:

```powershell
$env:MLFLOW_TRACKING_URI = "http://<mlflow-host>"
jupyter notebook
```

## Known gaps

The teammate notebooks (`ca1 hq file.ipynb`, `CA1 copy.ipynb`, `pbi05_connection_check.ipynb`, `PBI07_Nithish_models.ipynb`) still contain a default MLflow URL in a notebook cell. Some honour `MLFLOW_TRACKING_URI`; those cells should read the variable only. Their owners need to make that change.
