# PBI 05 — WS3 redeployment verification

On 17 November 2026 at 09:30 SGT, Elastic Beanstalk reported that the fresh `mlflow-aws-eb` environment launched successfully and displayed **Health: Ok**. Opening the AWS Domain link rendered the MLflow 3.18.0 home page and the Default experiment.

| Setting | Value |
| --- | --- |
| Region | N. Virginia (`us-east-1`) |
| Application and environment | `mlflow-aws-eb` |
| Deployment | Standard web server, single `t3.medium`, x86_64 |
| Platform | Docker on 64-bit Amazon Linux 2023, version 4.14.3 |
| Application version | `app-261117_092632` |
| Source bundle | WS3 `mlflow-eb.zip`, containing the Dockerfile at the ZIP root |
| Lab roles | `LabRole` and `LabInstanceProfile` |

The local Docker run was checked at `http://127.0.0.1:5000` before deploying to AWS. For team verification, copy the live Domain from Elastic Beanstalk and use it as `MLFLOW_TRACKING_URI`. Share the live URI through the team's private channel because the WS3 endpoint has no sign-in.

## Verification status

- [x] Local WS3 MLflow UI opened.
- [x] Fresh AWS environment reported Health: Ok and its MLflow UI loaded.
- [ ] Each teammate logs and retrieves parameters, metrics, and model artifacts against this environment.
- [ ] Verify model registration against this environment.

This record covers the 17 November deployment. Screenshots and run evidence from earlier environments do not verify runs against the replacement environment.
