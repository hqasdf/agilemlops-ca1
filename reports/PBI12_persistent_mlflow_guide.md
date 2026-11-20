# PBI 12: Persistent MLflow storage setup guide

**Owner:** Nithish (`nith-ish22`)
**Status:** Preparation only; no AWS changes or persistence verification have been made by this branch.
**Purpose:** Make the shared MLflow service retain experiment metadata, registered models and model artifacts when its Elastic Beanstalk container is replaced.

## 1. Requirements and evidence

The CA1 specification section 4 requires a GenAI web-service conversation of fewer than 10 prompting rounds, investigation and implementation of suitable AWS Academy Learner Lab services, and a short summary of the changes. Preserve the actual web conversation. PBI 12 additionally requires an observed container replacement followed by retrieval of previously logged metadata, metrics, artifacts and a registered model version. Record test dates and results; do not claim completion before these tests pass. [PBI 12](https://github.com/hqasdf/agilemlops-ca1/issues/12)

## 2. Current baseline and the intended design

The PBI 05 documentation describes Elastic Beanstalk running MLflow 3.17.2 in Docker with SQLite at `/mlflow/mlflow.db` and artifacts at `/mlflow/artifacts`. These are container-local in the documented AWS deployment. A replacement can lose them. A laptop Docker Compose bind mount is local-only and does not establish AWS persistence. Live configuration, active version, 5 MB upload limit and prior deployment completion still need verification.

Recommended candidate: Elastic Beanstalk continues to host MLflow; RDS MySQL stores run metadata and model registry; S3 stores model artifacts. Use a separate MLflow database/user and least-privilege S3 access. The current MLflow service is HTTP and has no demonstrated authentication. Restrict access appropriately; CORS and allowed-hosts do not authenticate users. Never expose passwords, credentials, secrets, customer data or full sensitive config in Git or screenshots.

**Crucial new information:** Nithish confirmed that Ephraim’s private PBI 04 RDS is in a different AWS account from the MLflow environment. PBI 04 verification showed access only from a CloudShell client inside its VPC; laptop and training access were not verified. An endpoint alone does not provide network access. [PBI 04 verification](https://github.com/hqasdf/agilemlops-ca1/blob/feat/pbi-04-rds/pbi04/verification.md)

## 3. Choose a feasible network before provisioning

1. Together, identify the MLflow account/region/VPC/subnets and the RDS account/region/VPC/subnets. Inspect CIDR blocks and whether they overlap. Note account owners, remaining lab budget, role permissions and current costs privately; do not paste credentials into chat or Git.
2. Confirm the existing RDS is accessible and Ephraim can create a separate database and restricted MLflow user. Confirm the storage capacity, TLS CA and expected billing. Do not change the existing `amlops_ca1` ETL tables or user.
3. Ask the lab tutor or inspect actual permissions for an approved private connection. Cross-account VPC peering is one possible option only if non-overlapping ranges, routes, security groups, DNS, both account permissions and lab rules allow it. It can incur data transfer charges. Do not configure peering or make RDS public just to bypass restrictions. [AWS peering guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/integrate-third-party-services/architecture-2.html)
4. If cross-account private networking is unavailable, evaluate hosting a separate MLflow database in the same account/VPC as Elastic Beanstalk, or co-locating services in an agreed account. A second RDS instance can add substantial cost and account/session coordination. Check the $50 budget and tutor requirements before creating one.
5. Record the chosen approach, reason and the team’s agreement. At this point no route, access, S3 bucket or separate database is assumed to exist.

## 4. Preserve existing results

- Ask the team to pause writes while collecting any backup. Determine which results exist in local stores and which are on the current shared server. User reports that local results and artifacts are preserved; independently check before relying on them.
- Decide whether to migrate existing runs or republish preserved artifacts. Changing the backend URI does not migrate SQLite, and new experiments may be necessary for the new artifact location. Keep old and new run IDs and evidence separate.
- Before changing an active service, preserve the exact deployed application bundle and non-secret environment settings. If current shared runs are needed, capture a consistent SQLite backup and all artifact/model folders using authorized instance access, store outside the instance and verify restoration. Screenshots and exported metrics alone are not a complete backup.
- Previous guidance proposed a 5 MB nginx package; deployment was interrupted before confirmation. Identify the actually running application version and configuration before packaging an update. Keep any verified upload-limit configuration rather than accidentally discarding it.

## 5. Guided setup after network decision

### A. S3 artifact storage

1. In the account selected by the team, open S3 and check whether the learner lab allows bucket creation and the projected storage cost.
2. Create a dedicated private bucket or approved prefix for MLflow. Retain block-public-access settings and apply the lab’s encryption and retention policy where available. Record bucket region and prefix, not credentials.
3. Confirm the Elastic Beanstalk EC2 instance profile can list the intended bucket/prefix and read/write only the required objects. In a lab account, role changes may be restricted; verify rather than assume permissions.
4. The intended model is server-proxied artifact access: the server needs S3 access; teammates use the MLflow URL. Proxied artifact users inherit the scope of the server’s artifact access, so secure the server endpoint. [MLflow tracking server](https://mlflow.org/docs/latest/tracking/server/)

### B. RDS metadata and registry

1. With the instance owner, create the separate database and user through an authorized connection; scope database privileges to that database and grant only the DDL/DML MLflow needs, including schema initialization/migrations. Do not use the ETL schema for MLflow tables.
2. Enable a permitted private network route from the MLflow EC2 instance to the selected RDS instance. Restrict port 3306 to the appropriate source; test DNS and network access from the actual MLflow host/container.
3. Use TLS with RDS CA and hostname verification. Determine the exact driver options using the chosen client version and AWS CA. Check the user can authenticate and can create the needed schema without printing passwords.
4. Store credentials outside source code using approved lab-supported secrets management. Plain-text environment settings may be visible to account users; environment variables are configuration, not inherently a secret vault. [AWS secrets documentation](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/AWSHowTo.secrets.env-vars.html)

### C. Build and deploy the application

1. Start from the exact currently deployed bundle and capture its SHA-256. Add a pinned MySQL driver and S3 support compatible with the chosen MLflow version; configure `--backend-store-uri` and `--artifacts-destination` using external settings and server-side artifact proxying.
2. Retain the existing working port, host allowlist, CORS configuration and any verified nginx upload-limit file. Do not use wildcard access rules as a substitute for authentication. MLflow’s S3 destination and proxy options are documented [here](https://mlflow.org/docs/latest/tracking/server/).
3. Build a source ZIP with Dockerfile and `.platform/` at its root. Record the version and SHA-256. Perform local review and, if possible, a local disposable integration check before touching the live EB environment.
4. During an agreed team window, protect required existing data, apply the new non-secret/secret settings, and deploy. A new backend starts empty unless data is explicitly migrated. Do not expose credentials in command output or screenshots.
5. Wait for AWS events to confirm success and the application to respond; investigate failures from logs without repeatedly rebuilding. Record actual environment version and status.

## 6. Verification and evidence checklist

- [ ] Preserve the GenAI web conversation and describe the original limitation.
- [ ] Record actual account/region/VPC relationship and approved route.
- [ ] Record backup or explicit team decision to republish; test any backup restore.
- [ ] Prove RDS authentication and TLS from the actual server; check MLflow tables in the separate database.
- [ ] In a new test experiment, log a parameter, metric and artifact; retrieve each from another client.
- [ ] Log a test model, retain its returned model URI, register a version and load it. MLflow 3 may use `models:/<model_id>`; do not assume a `runs:/.../model` layout.
- [ ] Show S3 artifact objects and RDS metadata without revealing secrets. Verify the 413 limit with a test file representative of the actual models; 5 MB is insufficient for any request larger than 5 MB.
- [ ] Capture run ID, metrics, registered model/version, alias and checksum of an artifact. Protect any work first, then perform a controlled **container replacement**. Verify the same run, metric, artifact checksum and registered model still load afterward. A restart alone does not demonstrate replacement.
- [ ] Test each teammate’s access and only then republish preserved project results if desired. Record new IDs and whether old IDs were migrated; do not claim the 491 runs moved automatically.
- [ ] Explain actual changes, evidence, remaining access/security/cost risks and cleanup/backup plan in the group template. Keep PBI 12 open until the acceptance criteria are demonstrated.

## 7. Recovery and costs

Save the previous bundle and settings. Roll back the application version only with an explicit data plan: an old container-local SQLite backend will not automatically contain new RDS runs. An RDS snapshot and S3 versioning/backups depend on lab permissions, cost and retention; test restores before claiming they work. Stopping an eligible RDS instance saves instance-hours but not all storage charges, and AWS may automatically restart it after seven days. [AWS RDS stop guidance](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_StopInstance.html)

## Sources

- ST1519 CA1 specification, section 2.3.2, part 4, in the team Brief folder.
- [PBI 12 issue](https://github.com/hqasdf/agilemlops-ca1/issues/12), [PBI 04 verification](https://github.com/hqasdf/agilemlops-ca1/blob/feat/pbi-04-rds/pbi04/verification.md), and PBI 05 deployment guide.
- [MLflow tracking server](https://mlflow.org/docs/latest/tracking/server/), [AWS proxy configuration](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/platforms-linux-extend.proxy.html).
