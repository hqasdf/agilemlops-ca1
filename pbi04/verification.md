# PBI 04 AWS RDS verification

Verified on 17 November 2026 in AWS Academy Learner Lab, us-east-1.

The private MySQL instance `amlops-ca1-mysql` hosts database `amlops_ca1`. A dedicated CloudShell VPC client reached port 3306 using security group `sg-0280aedebe84e31d1`. The RDS security group permits that group. The connection authenticated using the RDS-managed secret and verified the server certificate and hostname against the AWS RDS CA. The reported TLS cipher was `TLS_AES_256_GCM_SHA384`.

The three cleaned CA1 CSVs were uploaded to a private staging bucket. SHA-256 hashes of the copies downloaded into the VPC client matched the laptop files. The transactional import committed with the following counts.

| Table | Source rows | Database rows | Re-import inserted | Re-import unchanged |
| --- | ---: | ---: | ---: | ---: |
| customers | 901,312 | 901,312 | 0 | 901,312 |
| orders | 949,889 | 949,889 | 0 | 949,889 |
| labels | 950,318 | 950,318 | 0 | 950,318 |

Relationship checks found 444 labels without matching orders, 15 orders without matching labels and 14 labels without matching customers. These counts describe the imported data and require review by the data team. A test order was inserted inside a transaction and was absent after rollback.

The laptop schema and loader are in `001_schema.sql` and `load_cleaned.py`. For the live import, a separate equivalent staging-based script was run in the temporary CloudShell environment. The live script used the CSV header for column names and performed the import within a single transaction.

Ephraim deferred the full new-batch import test to ETL integration. Access and query verification by Han Qian and Nithish from their own environments, screenshots, setup instructions, and team review remain outstanding. The database is private, so teammates need an approved VPC-connected client and authorized credentials. On 17 November AWS accepted deletion of the temporary Secrets Manager interface endpoint and reported its state as deleting. The managed secret, RDS instance, and staging resources remain and may incur charges.

