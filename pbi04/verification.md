# PBI 04 AWS RDS verification

Verified on 17 November 2026 in AWS Academy Learner Lab, us-east-1.

The private MySQL instance `amlops-ca1-mysql` hosts database `amlops_ca1`. A dedicated CloudShell VPC client reached port 3306 using security group `sg-0280aedebe84e31d1`. The RDS security group permits that group. The connection authenticated using the RDS-managed secret and verified the server certificate and hostname against the AWS RDS CA. The reported TLS cipher was `TLS_AES_256_GCM_SHA384`.

The three cleaned CA1 CSVs were uploaded to a private staging bucket. SHA-256 hashes of the copies downloaded into the VPC client matched the laptop files. The transactional import committed with the following counts.

| Table | Source rows | Database rows | Re-import inserted | Re-import unchanged |
| --- | ---: | ---: | ---: | ---: |
| customers | 901,312 | 901,312 | 0 | 901,312 |
| orders | 949,889 | 949,889 | 0 | 949,889 |
| labels | 950,318 | 950,318 | 0 | 950,318 |

Independent comparisons of the cleaned CSVs produce the same counts as RDS: 444 labels without matching orders, 15 orders without matching labels and 14 labels without matching customers. The records are present in the cleaned inputs and were preserved on import. Their original cause has not been established. A test order was inserted inside a transaction and was absent after rollback.

## Schema decisions

Customers use the composite primary key `(country_code, customer_id)`. Orders and labels use `(country_code, order_id)`. These keys identify records within the country namespace and reject repeated keys. Customer and order identifiers are text to preserve their source representation. The labels table relates orders and customers using the country and respective identifiers. These links are checked by queries because the cleaned input includes unmatched records. Enforcing foreign keys during this import would reject them. Labels additionally index `(country_code, customer_id)` for customer joins. The fraud value is constrained to 0 or 1. `DECIMAL(38,18)` stores financial amounts, integer columns store counts, and date/time fields use MySQL date types. Nullable fields are declared in the schema, while keys and required values are not nullable.

## Connection and teammate review

The successful connection used a CloudShell VPC environment in the same VPC as the private RDS instance. The database security group allowed TCP 3306 from the dedicated client security group. The client retrieved the RDS-managed master credential using AWS Secrets Manager and validated the RDS certificate and hostname against the AWS CA. The temporary Secrets Manager VPC endpoint has since been deleted. The managed secret remains. A laptop or separate lab account does not automatically have a route to this private database, and direct teammate access has not yet been configured.

After an approved network path and individual credentials are configured, each teammate can connect using a MySQL client with server certificate validation and run `SELECT COUNT(*) FROM customers;`, `SELECT COUNT(*) FROM orders;`, and `SELECT COUNT(*) FROM labels;`. Expected counts are 901312, 949889, and 950318. Each teammate should capture their query output and record their name and environment. They can independently review this branch and its schema now. Training-environment access requires its own successful query and evidence.

The laptop schema and loader are in `001_schema.sql` and `load_cleaned.py`. For the live import, a separate equivalent staging-based script was run in the temporary CloudShell environment. The live script used the CSV header for column names and performed the import within a single transaction.

Ephraim deferred the full new-batch import test to ETL integration. Access and query verification by Han Qian and Nithish from their own environments, training access, and team review remain outstanding. AWS confirmed the temporary Secrets Manager interface endpoint no longer exists. The managed secret, RDS instance, and staging resources remain and may incur charges.

