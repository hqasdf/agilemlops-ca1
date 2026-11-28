# PBI 05 — Shared MLflow service

## Goal and boundary

Deploy one Docker-based MLflow tracking and model-registry service to AWS Elastic Beanstalk (EB), then prove all three members can use it. This baseline supports PBIs 06–10. PBI 12 owns the later durability improvement. PBI 05 coordinates resource configuration with PBI 11. RDS PBI 04 is not a listed blocker for PBI 05.

## Configuration contract

The container reads settings from environment variables. Confirm the names and handoff with the PBI 11 owner before deployment.

| Setting | Baseline/example | Purpose |
| --- | --- | --- |
| PORT | 5000 | Container listening port |
| MLFLOW_BACKEND_STORE_URI | sqlite:////mlflow/mlflow.db | Run metadata and model registry database |
| MLFLOW_ARTIFACTS_DESTINATION | /mlflow/artifacts | Server-proxied run/model artifacts |
| MLFLOW_ALLOWED_HOSTS | localhost,127.0.0.1 locally; exact EB hostname in AWS | Browser/API host validation |
| MLFLOW_TRACKING_URI | http://localhost:5000 locally; EB URL for team clients | Client-side connection setting |

Do not commit credentials or a .env file. The baseline server has no user login. Use only sample/course data, and agree on network access with the team and tutor. Do not use unrestricted Host/CORS wildcards as a substitute for access control. No CORS override is needed for normal same-origin UI use.

## Local smoke check

Prerequisite: Docker Desktop is running.

From the repository root:

```sh
docker compose -f docker-compose.pbi05.yml up --build
```

Open http://127.0.0.1:5000 and confirm the MLflow UI loads. Keep the terminal output for troubleshooting. Stop with Ctrl+C. The compose volume stores local test data in mlflow-data; that is local development state and must not be submitted.

## Elastic Beanstalk deployment outline

Follow the course WS3 workflow and the current AWS Academy console:

1. Start the active Learner Lab session and use its provided roles. The workshop specifies Docker on EB, LabRole, LabInstanceProfile and a t3.medium example; confirm those options remain available in your lab.
2. Create/select the agreed EB application and environment in the team-agreed region. Record the exact environment name and generated hostname.
3. Build the source ZIP with Dockerfile at the ZIP root (not nested inside a folder). The course workshop demonstrates a Dockerfile-only ZIP. Keep the Dockerfile/dependency pin in Git as well.
4. Set PORT, MLFLOW_BACKEND_STORE_URI, MLFLOW_ARTIFACTS_DESTINATION, and MLFLOW_ALLOWED_HOSTS in EB environment configuration. Set the allowed host to the actual EB hostname. Check the health-check path and port use the service root/5000 according to the selected EB Docker platform.
5. Deploy the ZIP. Wait for Healthy/Ok, open the recorded EB URL, and confirm the UI responds.
6. Save the exact ZIP that was deployed and its SHA-256. Record EB application/environment, region, deployment time, URL, Dockerfile, MLflow version (3.17.2), and configuration variable names (never secret values).

The EB container uses SQLite and a local artifact directory for the PBI 05 baseline. These may be lost when the EB instance/container is replaced. Do not claim durability; prove persistence only under PBI 12. Coordinate before terminating the EB environment because PBIs 06–10 need the shared URL. Check Learner Lab credit and stop/terminate only when the team agrees the endpoint is no longer needed.

## Client connection

Each member sets the MLflow client tracking URI to the shared EB URL (no trailing path) and confirms the browser UI opens. The team should use the same server URL and compatible MLflow client version; record the exact client and server versions used.

## Acceptance evidence

- [x] EB Docker environment is Healthy/Ok and the service URL responds (screenshots 01 and 02, 14 Nov; the current server is described in the final status below).
- [x] Each of the three members logs and retrieves parameters, metrics and model artifacts through the shared service (final status below).
- [x] A model registry is available for PBI 09 (`fraud-detector`, final status below).
- [x] The exact deployed ZIP, its SHA-256, the Dockerfile and the dependency versions are kept in this repository (`reports/pbi05_evidence/mlflow-eb.zip`, `Dockerfile`); the bundle now running is in `reports/pbi12_evidence/deployment_bundle/`.
- [x] Deployment, connection and external-configuration instructions are documented in this file.
- [ ] Teammate review or pull-request evidence for this document.

## Out of scope

PBI 05 does not train or compare each member's models, select the champion, implement PBI 04 RDS, orchestrate PBI 10, or complete the storage-hardening work of PBI 12.

## Evidence captured — 14 November 2026

Evidence files are in `reports/pbi05_evidence/`:

- `01-elastic-beanstalk-healthy.png` — Elastic Beanstalk environment reported Healthy/Ok.
- `02-mlflow-ui-available.png` — MLflow 3.17.2 UI opened at the deployed environment URL.
- `03-owner-run-completed.png` — Nithish's tracking smoke-test run finished and its metric is visible. The parameter and text artifact were also confirmed in the run UI by the owner; this screenshot is the run overview.
- `04-model-registry-version-1.png` — `pbi05_registry_smoke_test`, Version 1, is visible in the registry.
- `mlflow-eb.zip` — exact Dockerfile-only source ZIP deployed to Elastic Beanstalk.

Deployed ZIP SHA-256: `D330E002C119A378D68300916FC5A60C56973AF8B7E4FF6248EB2D9ECFE00E28`.

Progress boundary: these results verify the owner's deployment and smoke checks. The environment was terminated after those checks to conserve AWS Academy credits. It was restored on 16 November, and the MLflow UI became reachable again. The latest Elastic Beanstalk health state has not been captured.

## Teammate smoke test — 16 November 2026

The MLflow Experiments page showed PBI05-teammate-smoke-test at the restored endpoint. This confirms that the browser reached the service and the experiment was created.

Evidence files:

- reports/pbi05_evidence/05-teammate-experiment-created.png — the experiment is visible in the MLflow UI.
- reports/pbi05_evidence/06-teammate-runs-page-error.png — the Runs page displayed INTERNAL_ERROR. Its screenshot also shows an old example filter (metrics.rmse < 1 and params.model = "tree") and the state filter set to Active.

These screenshots do not confirm that teammate-connection-check finished, that connection_test = 1.0 was logged, or that test.txt was retrieved. This attempt verifies endpoint access and experiment creation only. The acceptance criterion requiring every member to log and retrieve a parameter, metric, and artifact remains pending. Retry after clearing the filter and selecting All or Finished, then record the run ID and verify the metric and artifact.

The local SQLite database and artifact directory are baseline-only and are not claimed to be durable; PBI 12 owns that improvement.

## Final status (23 November 2026)

**Which server is the shared server.** The PBI 05 environment was a baseline with SQLite and container-local artifacts. PBI 12 replaced it with `mlflow-pbi12` (RDS MySQL for metadata, S3 for artifacts). All team work now uses this URL:

`http://mlflow-pbi12.eba-9bnrxizu.us-east-1.elasticbeanstalk.com`

The roughly 491 runs held by the old PBI 05 server were not migrated (see `reports/PBI12_MLFLOW_STORAGE.md`). The PBI 05 evidence above remains the record of the first deployment. The teammate smoke test of 16 November, which ended with an `INTERNAL_ERROR` on the Runs page, is superseded by the results below.

**All three members use the shared service.** On 23 November the shared server holds:

| Member | Experiment | Runs | Retrievable models |
| --- | --- | ---: | --- |
| Han Qian | `PBI06_HanQian` | 28 | model files on the runs (checked on the two registered runs: `MLmodel`, `python_model.pkl`, environment files) |
| Nithish | `PBI07_Nithish` | 3 | one logged model per run |
| Ephraim | `PBI08_Ephraim` | 18 | logged models on the tuned runs; the baseline weights were kept locally |

Parameters, metrics, tags and member identity are logged on every run. Screenshots of the three experiment run lists are in `reports/pbi09_evidence/` on the PBI 09 branch (`final07_hanqian_runs_28.jpg`, `final08_ephraim_runs_8.jpg`, `final09_nithish_runs_3.jpg`; the Ephraim image was taken when that experiment had 8 runs).

**Model registry.** The registry holds `fraud-detector` version 1 (alias `champion`, Han Qian's calibrated Random Forest) and version 2 (alias `runner-up`, calibrated CatBoost), registered on 23 November for PBI 09. Han Qian loaded both back through the registry aliases the same day (`reports/pbi09_evidence/final06_load_test_han_qian.jpg`).

**Using the server.** Set the client tracking URI to the address above (no trailing path) before logging, for example `$env:MLFLOW_TRACKING_URI="http://mlflow-pbi12.eba-9bnrxizu.us-east-1.elasticbeanstalk.com"` in PowerShell. Connection settings come from environment variables and are not committed. The server uses plain HTTP and has no login, so do not log personal data. The server stops when the AWS Learner Lab session ends; start the lab, wait until the environment is Green and open `/health` before use.
