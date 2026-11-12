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

## Acceptance evidence — do not tick until shown

- [ ] EB Docker environment is Healthy/Ok and the service URL responds.
- [ ] Each of the three members logs and retrieves a parameter, metric, and small harmless artifact through the shared service; record member, run ID and retrieval result.
- [ ] Register a small test model/version and verify it appears in the registry for PBI 09.
- [ ] Preserve the exact deployed ZIP, SHA-256, Dockerfile, setup files and dependency versions in the group submission location.
- [ ] Document deployment, connection, recovery and external configuration; retain a teammate review/PR.

## Out of scope

PBI 05 does not train or compare each member's models, select the champion, implement PBI 04 RDS, orchestrate PBI 10, or complete the storage-hardening work of PBI 12.
