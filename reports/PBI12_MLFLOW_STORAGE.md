# PBI 12: Durable MLflow storage on AWS

| | |
|---|---|
| Issue | #12 (sub-issues #59 limits and GenAI discussion, #60 implementation, #61 verification) |
| Owner | Nithish (`nith-ish22`) |
| Built and tested | 22 Nov 2026 (times below are Singapore time, SGT, unless marked UTC) |
| Server | `http://mlflow-pbi12.eba-9bnrxizu.us-east-1.elasticbeanstalk.com` |
| Evidence | `reports/pbi12_evidence/` (31 screenshots, command output, deployment files) |

## 0. Summary

The shared MLflow server built in PBI 05 kept everything inside its own Docker container. When Elastic Beanstalk replaced the container, every experiment, run, metric, model file and registered model was lost. PBI 12 moves the data out of the container:

- experiment metadata and the model registry go to **Amazon RDS for MySQL** (`mlflow-mysql`),
- model files and other artifacts go to an **Amazon S3** bucket (`mlflow-artifacts-nithish-20261122`),
- the Elastic Beanstalk container (`mlflow-pbi12`) now holds no data and can be replaced at any time.

We proved this by logging a run, a trained model and a registered model version, replacing the container, and reading all of it back. All five checks passed (section 7).

Acceptance criteria and where each one is covered:

| # | Criterion | Where |
|---|---|---|
| 1 | Explain the SQLite and container-local artifact limits | Section 1 |
| 2 | GenAI web conversation (under 10 prompts, shareable link, no local coding agent) | Section 9 |
| 3 | Implement AWS Learner Lab storage | Sections 3 to 5 |
| 4 | Document before/after architecture, services, reasons, and the real configuration and deployment changes | Sections 2 to 6 |
| 5 | Show metadata, metrics, model artifacts and a registered model version retrieved after container replacement, with evidence kept | Sections 7 and 11 |

## 1. Why the PBI 05 design was not good enough (criterion 1)

PBI 05 ran MLflow in one Docker container on Elastic Beanstalk. MLflow needs two stores, and both lived inside that container.

| Store | What it holds | PBI 05 location | Problem |
|---|---|---|---|
| Backend store | experiments, runs, parameters, metrics, tags, model registry | a SQLite file in the container | Deleted when the container is replaced. SQLite lets only one writer in at a time, so several teammates logging together can fail with "database is locked". The file cannot be backed up or inspected without logging in to the instance. |
| Artifact store | model files, plots, input examples | a folder in the container | Deleted with the container, and it fills the instance disk. A registered model would still be listed in the registry but its files would be gone, so it could not be loaded. |

Elastic Beanstalk replaces the container on every redeploy, platform update and instance failure, so the loss is routine and not rare. That makes a shared model registry unreliable for the team.

Screenshot ss01 shows the old PBI 05 environment (`mlflow-aws-eb`).

## 2. Architecture before and after (criterion 4)

Before:

```
Team laptops --HTTP--> Elastic Beanstalk (Docker container)
                         MLflow + SQLite file + artifact folder   <- all data inside
```

After:

```
Team laptops --HTTP--> Elastic Beanstalk environment "mlflow-pbi12"
                       Docker container: MLflow 3.17.2 (--serve-artifacts), no data stored
                          |                                   |
                          | metadata, over TLS (port 3306)    | artifacts (boto3)
                          v                                   v
              Amazon RDS MySQL 8.4.11                Amazon S3 bucket
              instance "mlflow-mysql"                "mlflow-artifacts-nithish-20261122"
              database "mlflow"
```

`--serve-artifacts` makes the MLflow server upload and download artifacts for the clients. Because of this, team laptops need no AWS keys. Only the Elastic Beanstalk instance role (`LabInstanceProfile`) talks to S3.

## 3. Services chosen and why

| Need | Service | Reason |
|---|---|---|
| Metadata | Amazon RDS for MySQL 8.4 | MLflow supports MySQL directly. A real database server accepts many writers at once, survives container loss, has automated backups (7 days kept) and keeps data on its own storage. Available in the Learner Lab. |
| Artifacts | Amazon S3 | MLflow supports S3 directly. Object storage is durable, not tied to any server, and cheap. Available in the Learner Lab. |
| Hosting | Elastic Beanstalk (kept from PBI 05) | Already known to the team, so only the container configuration changed. |

Options considered and not used:

- Amazon EFS or an EBS volume for the SQLite file and artifacts: keeps the data, but still one writer and still tied to one instance.
- Aurora: more cost and setup than the project needs; RDS MySQL is enough for a team of this size.
- Reusing the RDS instance from PBI 04: it belongs to another AWS account (see `PBI12_persistent_mlflow_guide.md`), so a separate RDS instance in this account was created.

## 4. What was built, step by step (criterion 3)

### 4.1 RDS MySQL database

| Setting | Value | Screenshot |
|---|---|---|
| Engine | MySQL Community 8.4.11 | ss02 |
| Instance class | db.t3.micro (2 vCPU, 1 GiB) | ss03 |
| Storage | 20 GiB General Purpose SSD (gp3) | ss09 |
| Network | default VPC, **Publicly accessible: No** | ss04, ss08 |
| Certificate authority | rds-ca-rsa2048-g1 | ss05, ss08 |
| Automated backups | on, 7 days | ss06 |
| Initial database name | `mlflow` | ss09 |
| Status | Available, identifier `mlflow-mysql` | ss07 |
| Endpoint | `mlflow-mysql.cpgusg6make8.us-east-1.rds.amazonaws.com`, port 3306 | ss08 |

The database is private. It is reachable only from inside the VPC, and only from the security groups listed in 4.3.

Two database accounts are used: the RDS master user `admin` (set up and maintenance only) and a separate application user `mlflow_app` that the MLflow container uses. Passwords are never stored in the repository or the image.

### 4.2 S3 bucket

| Setting | Value | Screenshot |
|---|---|---|
| Name and region | `mlflow-artifacts-nithish-20261122`, us-east-1 | ss10 |
| Object ownership | ACLs disabled (bucket owner enforced) | ss10 |
| Block all public access | on | ss11 |
| Default encryption | SSE-S3 (S3 managed keys) | ss12 |
| Versioning | Enabled | ss14 |
| Created | 22 Nov 2026 16:14 | ss13, ss14 |

Public access is blocked, so model files cannot be read by anyone without credentials. Versioning keeps older copies if an object is overwritten or deleted by mistake.

### 4.3 Network rules (security groups)

The RDS instance uses security group `mlflow-rds-sg`. Its inbound rules allow MySQL (TCP 3306) only from:

1. the security group of the Elastic Beanstalk instance, described as "MLflow server access", so the MLflow container can reach the database,
2. a temporary security group used by an AWS CloudShell session for database administration,
3. one individual IP address used during setup (address hidden in the screenshots).

Nothing else can connect. Screenshots: ss15 and ss16 (first edit), ss18 (rules at the end).

Database administration was done from AWS CloudShell attached to the VPC. ss17 shows a successful connection that checks the server certificate (`--ssl-verify-server-cert`) and lists the `mlflow` database next to the system databases.

### 4.4 Elastic Beanstalk environment

A new environment `mlflow-pbi12` was created in the existing application `mlflow-aws-eb`. The old environment was left unchanged.

| Setting | Value | Screenshot |
|---|---|---|
| Platform | Docker on 64bit Amazon Linux 2023 (4.14.4) | ss19 |
| Environment type | Single instance, nginx proxy | ss20 |
| Service role / instance profile | `LabRole` / `LabInstanceProfile` (the roles the Learner Lab allows) | ss20 |
| First state | launched with the AWS sample application, health Ok | ss21, ss22 |

### 4.5 Application bundle and configuration

The application bundle (`pbi12-mlflow-rds-s3.zip`, about 4.6 KB) contains:

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.12 slim image with pinned `mlflow==3.17.2`, `PyMySQL==1.1.1`, `boto3==1.40.61`, `cryptography==45.0.7`. Copies in the RDS CA bundle. Copy in `pbi12_evidence/deployment_bundle/Dockerfile`. |
| `start_mlflow.py` | Start script. Reads `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `MLFLOW_ARTIFACTS_DESTINATION`, `MLFLOW_ALLOWED_HOSTS`; stops with an error if one is missing, if `DB_HOST` does not end in `.rds.amazonaws.com`, or if the artifact destination is not an `s3://` address. Builds the backend URI `mysql+pymysql://...?charset=utf8mb4&ssl_ca=/app/certs/rds-bundle.pem&ssl_check_hostname=true` and starts `mlflow server --host 0.0.0.0 --port $PORT --serve-artifacts`. |
| `certs/rds-bundle.pem` | AWS RDS certificate bundle, so the connection to MySQL is encrypted and the server identity is checked. |
| `.platform/nginx/conf.d/upload-limit.conf` | `client_max_body_size 100m;` so model uploads are not rejected by nginx (default is 1 MB). Copy in `pbi12_evidence/deployment_bundle/`. |

Settings are passed as Elastic Beanstalk environment properties: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` (`mlflow_app`), `DB_PASSWORD`, `MLFLOW_ARTIFACTS_DESTINATION` (`s3://mlflow-artifacts-nithish-20261122`), `MLFLOW_ALLOWED_HOSTS`, `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS`, `PORT`. The password exists only as an Elastic Beanstalk property, not in the zip, the image or Git.

The bundle was uploaded with "Upload and deploy" (ss23). The application version label used for later redeploys is `pbi12-mlflow-rds-s3-v1`.

## 5. Problems met during deployment and how they were fixed

This section is the debugging record. The first deployment did not work.

### 5.1 First result: 502 Bad Gateway

After the deployment finished (ss24: "Environment update successfully completed"), the site returned `502 Bad Gateway` from nginx (ss25), and the environment health went to Degraded (ss26). A 502 from nginx means nginx is running but nothing is answering behind it, so the MLflow container was exiting right after start.

### 5.2 Database connection checks

From CloudShell, a connection test to the RDS endpoint timed out (error 2002), which meant the network path was blocked by a security group, and a second attempt reached the server but the client reported a TLS error (error 2026) (ss27). Inbound rules were corrected so that the Elastic Beanstalk instance security group and the CloudShell group could reach port 3306 (ss18). The container still did not stay up, so the environment logs were checked next.

### 5.3 Root cause: half-finished database migration

On start, MLflow 3.17 builds its tables (an "Alembic migration"). One migration step creates a database trigger. MySQL only allows that when `log_bin_trust_function_creators` is 1. We had put this setting in a custom parameter group, but:

1. a parameter group change on RDS is applied only after the instance is rebooted, and the instance had not been rebooted yet, so the trigger step failed;
2. MySQL cannot undo table creation (DDL is not transactional), so six tables from that migration (`endpoint_tags`, `endpoint_bindings`, `endpoint_model_mappings`, `model_definitions`, `endpoints`, `secrets`) stayed behind;
3. on every restart the migration tried to create the same tables again and stopped with MySQL error 1050 (table already exists). The container exited each time, so nginx kept answering 502.

### 5.4 Fix

1. A parameter group named `mlflow-mysql-trigger-fix` with `log_bin_trust_function_creators = 1` was attached to the instance.
2. The RDS instance was rebooted so the parameter group changed to `in-sync`.
3. The six leftover tables were dropped in the `mlflow` database (ss28: `Query OK, 0 rows affected`). They were empty, created by the failed migration only.
4. The application version `pbi12-mlflow-rds-s3-v1` was redeployed. The migration then ran from the beginning, finished, and the server started.

Result after the fix: `/health` returns 200 and the experiments API responds. This was checked before the persistence tests.

### 5.5 Version note

The client pin in `requirements.txt` is `mlflow==3.18.0` but the server image is 3.17.2. The tests in this report used a 3.17.2 client so client and server match. MLflow 3.18 has a different database schema, and schema migrations cannot be reversed, so do not change the server image version without taking an RDS snapshot first.

## 6. Code changes in this repository

| File | Change |
|---|---|
| `src/mlflow_remote.py` (new) | `ensure_server_reachable(uri)` does a GET on `<uri>/health` and raises a clear error ("MLflow server not reachable at ... Check that MLFLOW_TRACKING_URI is the current Elastic Beanstalk URL, that it starts with http://, and that the environment is Green") before training starts. Local URIs are skipped. |
| `src/train_nithish.py` | Calls `ensure_server_reachable` before connecting, so a wrong or stopped server fails at once and not part-way through training. |
| `src/train_hq.py` | Not changed. PBI 06 pins the file by SHA-256 hash (`TRACKING_PATCH_SOURCES` in `src/pbi06_tracking.py`, checked by `tests/test_pbi06_tracking.py`), so editing it would break that integrity check. It already reads `MLFLOW_TRACKING_URI`, so it logs to the new server with no code change; the early reachability message is only in `train_nithish.py`. |
| `tests/test_mlflow_remote.py` (new) | 3 unit tests: local URI not checked, a running server passes, a closed port gives the error message. |
| `scripts/verify_pbi12_persistence.py` (new) | The repeatable persistence test (`log` then `check`, section 7.2). |
| `.env.example` (new) | Shows the `MLFLOW_TRACKING_URI` to use. No secrets. |

Test run on this branch: 61 tests pass. One older test, `tests/test_train_hq.py::test_threshold_selection_and_outer_validation_isolation`, fails because it patches `mlflow.artifacts.download_artifacts` while `train_hq.py` downloads through the MLflow client; it fails the same way without the PBI 12 changes.

To log to the shared server:

```
set MLFLOW_TRACKING_URI=http://mlflow-pbi12.eba-9bnrxizu.us-east-1.elasticbeanstalk.com     (Windows cmd)
$env:MLFLOW_TRACKING_URI="http://mlflow-pbi12.eba-9bnrxizu.us-east-1.elasticbeanstalk.com"  (PowerShell)
```

## 7. Verification after container replacement (criterion 5)

What "container replacement" means here: the version `pbi12-mlflow-rds-s3-v1` was redeployed from the Elastic Beanstalk console, which stops the old Docker container and starts a new one with an empty filesystem. If data were stored in the container it would be gone afterwards.

### 7.1 Test 1: REST test with a small stand-in model (12:23 to 12:26 UTC)

A run, parameters, metrics, a model folder and a registered model were created through the MLflow REST API, the container was replaced, and everything was read back.

| Item | Before replacement | After replacement |
|---|---|---|
| Run `e1d22567bf4946c783dd13e320dcd35a` | FINISHED | FINISHED |
| Parameters | model=demo-stand-in, threshold=0.5 | identical |
| Metrics | f1=0.8123, average_precision=0.7345 | identical |
| Files in `model/` | MLmodel, model.txt | identical, `model.txt` content identical |
| Registered model | `pbi12-persistence-model` v1, READY | identical |

Limit of this test: the artifact was a small text file, not a trained model.

### 7.2 Test 2: real scikit-learn model (22 Nov 2026, evening)

`scripts/verify_pbi12_persistence.py` was run from a team laptop with MLflow client 3.17.2.

Phase `log`: trains a logistic regression on 200 generated rows, logs parameters (`model`, `rows`), logs a metric, logs the model with `mlflow.sklearn.log_model` and an input example, registers it as `pbi12-persistence-model`, and writes the run ID to a local file. Result: run `1223ed104ad249af965e665cd3072770` (`pbi12-real-model-before-replacement`), registered model version 2 (ss29, ss31).

Container replaced: new container `a97629a175bd` started at 12:38:49 UTC. The first redeploy attempt hit a one-off nginx file-copy error; Elastic Beanstalk retried by itself and the environment finished Ready and Green.

Phase `check`: reads the run ID from the file and queries the new container. It checks five things and exits with an error unless all pass.

```
PASS  metadata (params)
PASS  metrics
PASS  model artifact files
PASS  registered model version
PASS  model loads and predicts the same
```

The last check loads the model from S3 through the server and confirms its predictions equal the ones made before replacement. The artifact list returned after replacement was `model/MLmodel`, `model/conda.yaml`, `model/input_example.json`, `model/model.skops`, `model/python_env.yaml`, `model/requirements.txt`, `model/serving_input_example.json`. Full output: `pbi12_evidence/logs/persistence_check_output.txt` and screenshot ss29. The MLflow web interface on the new container shows the two runs (ss30) and the run with its registered model v2 and a logged model in Ready state (ss31).

### 7.3 Result

| Criterion item | Result |
|---|---|
| Metadata (parameters) retrieved | Pass, both tests |
| Metrics retrieved | Pass, both tests |
| Model artifacts retrieved | Pass, both tests |
| Registered model version retrieved | Pass, v1 and v2 |
| Model loads and predicts identically | Pass, test 2 |

## 8. Changes for teammates

- Set `MLFLOW_TRACKING_URI` to the new URL. The PBI 05 server and its URL are not the shared server any more.
- Runs stored on the old PBI 05 server (about 491) were **not** migrated. Each owner should re-run their experiments, or re-log runs from their local MLflow stores, against the new server.
- The PBI 07 notebook default was changed to the new server (committed on the PBI 7 branch).

## 9. GenAI web conversation (criterion 2)

The CA1 specification asks for the investigation and design to be done through a GenAI web chat with fewer than 10 prompting rounds and no local coding agent. The shareable link to that conversation: https://claude.ai/share/60d64645-d228-48de-9c92-293091fad6dc.

## 10. Known limits

- The server uses plain HTTP and has no login. Anyone with the URL can read and write. Do not log personal or sensitive data.
- One Elastic Beanstalk instance and one RDS instance: the data is durable but the service is not highly available.
- The inbound HTTP rule on the Elastic Beanstalk security group was limited to one IP address. After the lab was restarted on 23 November the server looked healthy in the console but could not be reached from other computers. The rule now allows all addresses so teammates and the interview demo can connect. Together with the missing login, this means the server is open to anyone who knows the URL.
- A Learner Lab session ending can stop the environment or database. Check the URL and that the environment is Green when a session restarts. The RDS instance keeps its data while stopped.
- The temporary CloudShell and personal-IP inbound rules on `mlflow-rds-sg` (4.3) were for setup and should be removed when administration is finished.

## 11. Evidence index

All screenshots are in `pbi12_evidence/screenshots/`. Account ID, sign-in name and personal IP addresses were covered with black boxes. See `pbi12_evidence/README.md` for the mapping of each file to a criterion.

## 12. Lab stop and start check (23 November 2026)

The AWS Learner Lab session was ended and started again on 23 November, which stops the Elastic Beanstalk instance and the RDS database. This was a second proof of persistence, separate from the container replacement in section 7.

| Check after the restart | Result |
| --- | --- |
| `/health` on the server URL | Responds `OK` |
| Experiments `PBI06_HanQian` (28 runs), `PBI07_Nithish` (3 runs), `PBI08_Ephraim` and `pbi12-persistence-check` | All present with their runs and metrics |
| Model files | Present on the registered runs and on the logged models |
| Registered model `fraud-detector` | Versions 1 and 2 present, status READY, aliases `champion` and `runner-up` intact |

**Recovery steps after a lab restart:**

1. Start the lab and wait for the environment to turn Green in Elastic Beanstalk and for the RDS instance `mlflow-mysql` to show Available.
2. Open the server URL followed by `/health`. If it does not respond from another computer, check the inbound HTTP source on the Elastic Beanstalk security group (see section 10).
3. Open the Experiments page and confirm the runs are listed.
4. Teammates keep the same `MLFLOW_TRACKING_URI`; the address does not change.

Stopping and starting the lab does not terminate the environment or the database. Do not terminate `mlflow-pbi12` or `mlflow-mysql`.
