# PBI 12 evidence

Main document: [`../PBI12_MLFLOW_STORAGE.md`](../PBI12_MLFLOW_STORAGE.md). Captured 22 Nov 2026 (Singapore time in the screenshot names and clocks). AWS account ID, sign-in name, personal IP addresses and the role ARN were covered with black boxes before the images were added to the repository.

## Criterion 1: limits of the PBI 05 design
| File | Shows |
|---|---|
| `screenshots/ss01_before_old_environment.jpg` | Old PBI 05 environment `mlflow-aws-eb` (container-local data) |

## Criterion 3: AWS storage built
| File | Shows |
|---|---|
| `ss02_rds_engine_mysql_8_4_11` | RDS engine MySQL 8.4.11 |
| `ss03_rds_instance_class_t3_micro` | Instance class db.t3.micro |
| `ss04_rds_private_network_no_public_access` | Default VPC, public access: No |
| `ss05_rds_certificate_authority` | Certificate authority rds-ca-rsa2048-g1 |
| `ss06_rds_automated_backups` | Automated backups, 7 days |
| `ss07_rds_instance_available` | `mlflow-mysql` status Available |
| `ss08_rds_endpoint_and_security_group` | Endpoint, port 3306, security group `mlflow-rds-sg`, not publicly accessible |
| `ss09_rds_configuration_details` | Engine 8.4.11, DB name `mlflow`, 20 GiB gp3, not Multi-AZ |
| `ss10_s3_bucket_name` | Bucket name, ACLs disabled |
| `ss11_s3_block_public_access` | Block all public access on |
| `ss12_s3_default_encryption` | SSE-S3 default encryption |
| `ss13_s3_bucket_created` | Bucket created |
| `ss14_s3_bucket_versioning` | Versioning Enabled |
| `ss15_rds_security_group_rules` | RDS security group rules, first edit |
| `ss16_rds_security_group_edit_inbound` | Inbound MySQL rules being edited |
| `ss17_database_reachable_with_tls` | CloudShell connection with certificate check, `SHOW DATABASES` lists `mlflow` |
| `ss18_rds_security_group_final_rules` | Inbound rules at the end (3 rules) |
| `ss19_eb_environment_details` | Beanstalk platform: Docker on Amazon Linux 2023 |
| `ss20_eb_single_instance_service_access` | Single instance, nginx, LabRole / LabInstanceProfile |
| `ss21_eb_environment_launched` | Environment `mlflow-pbi12` launched next to the old one |
| `ss22_eb_sample_app_running` | Environment serving the sample app |
| `ss23_eb_upload_and_deploy` | Uploading the bundle |
| `ss24_eb_deployment_events` | Deployment events |

## Problem and fix (criterion 4)
| File | Shows |
|---|---|
| `ss25_first_result_502_bad_gateway` | First result: 502 Bad Gateway |
| `ss26_eb_environment_degraded` | Environment health Degraded |
| `ss27_database_connection_troubleshooting` | Connection test errors from CloudShell |
| `ss28_partial_tables_dropped` | Six leftover tables dropped, `Query OK` |

## Criterion 5: persistence after container replacement
| File | Shows |
|---|---|
| `ss29_persistence_check_all_pass` | `log` result at top, then `check` with 5 PASS lines |
| `ss30_mlflow_runs_on_new_server` | Both runs on the new server |
| `ss31_mlflow_run_with_registered_model` | Run with registered model `pbi12-persistence-model` v2 |
| `logs/persistence_log_output.txt` | Text of the `log` phase |
| `logs/persistence_check_output.txt` | Text of the `check` phase |

## Deployment files
`deployment_bundle/Dockerfile` and `deployment_bundle/upload-limit.conf` are copies of the files in the deployed bundle. The start script and the certificate bundle are described in section 4.5 of the main document. No passwords or keys are stored here.
