# PBI 05 evidence

Captured 14 November 2026 for issue #5 in hqasdf/agilemlops-ca1.

| File | What it demonstrates |
|---|---|
| 01-elastic-beanstalk-healthy.png | Elastic Beanstalk health was Ok after deployment. |
| 02-mlflow-ui-available.png | MLflow 3.17.2 UI was reachable at the Elastic Beanstalk URL. |
| 03-owner-run-completed.png | Nithish's smoke-test run finished and test_metric = 1 is visible. The owner also confirmed the parameter and text artifact in the run UI. |
| 04-model-registry-version-1.png | pbi05_registry_smoke_test has Version 1. |
| mlflow-eb.zip | Exact source ZIP deployed to Elastic Beanstalk (Dockerfile at ZIP root). |

SHA-256 of mlflow-eb.zip: D330E002C119A378D68300916FC5A60C56973AF8B7E4FF6248EB2D9ECFE00E28

pbi05_connection_check.ipynb at the repository root contains the reproducible owner-side tracking and registry checks. The three-member log-and-retrieve acceptance criterion remains pending teammate evidence. Do not mark PBI 05 complete on this evidence alone.