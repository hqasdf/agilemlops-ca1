# PBI 04 — schema and source reconciliation

Review date: 22 November 2026. Scope: the versioned PBI 04 schema and the local cleaned historical CSVs. This report does not assert a fresh live database query or a team review.

## Schema contract

Source of truth for this branch: [001_schema.sql](001_schema.sql). The migration uses CREATE TABLE IF NOT EXISTS; it does not migrate an already existing incompatible table. Verify live DDL with SHOW CREATE TABLE before treating it as identical.

| Table | Columns and nullability | Key and purpose |
| --- | --- | --- |
| customers | country_code CHAR(2), customer_id VARCHAR(32), counts INT, total_payment_last_50days DECIMAL(38,18): NOT NULL. mobile_verified VARCHAR(5), first_order_datetime DATETIME: nullable. | PRIMARY KEY (country_code, customer_id); namespacing customer IDs by country. |
| orders | country_code CHAR(2), order_id VARCHAR(32), collect_type VARCHAR(32), order_value and refund_value DECIMAL(38,18), order_date DATE: NOT NULL. payment_method VARCHAR(64), num_items_ordered INT: nullable. | PRIMARY KEY (country_code, order_id); namespacing order IDs by country. |
| labels | country_code CHAR(2), order_id VARCHAR(32), customer_id VARCHAR(32), is_fraud TINYINT: NOT NULL. Fraud value is constrained to 0 or 1. | PRIMARY KEY (country_code, order_id); index (country_code, customer_id) supports the customer join. |

Country uses ASCII binary collation; IDs use a binary utf8mb4 collation and are stored as text to preserve their representations. Amounts use decimal rather than binary floating-point storage. Date and datetime types distinguish order dates and customer first-order timestamps.

The label-to-order relationship uses (country_code, order_id); label-to-customer uses (country_code, customer_id). There are intentionally no enforced foreign keys because the supplied cleaned historical files include unmatched records. An inner join removes those records from joined results; retaining the source rows does not make them eligible for training. Investigate missing references before deciding whether to repair, quarantine or exclude them.

## Historical reconciliation

| Measure | Cleaned CSV | Prior RDS evidence, 17 Nov |
| --- | ---: | ---: |
| Customers | 901,312 | 901,312 |
| Orders | 949,889 | 949,889 |
| Labels | 950,318 | 950,318 |
| Labels without orders | 444 | 444 |
| Orders without labels | 15 | 15 |
| Labels without customers | 14 | 14 |

The local cleaned files were independently recounted on 22 November. There were zero duplicate country-qualified primary keys in each file. The 17 November RDS figures and repeat-import evidence are recorded in [verification.md](verification.md); their agreement is historical evidence, not proof of the database's current contents. Matched order/label pairs total 949,874. The arithmetic is 949,889 − 15 = 950,318 − 444 = 949,874.

| Country | Labels without orders | Orders without labels | Labels without customers |
| --- | ---: | ---: | ---: |
| BD | 6 | 2 | 1 |
| MY | 144 | 6 | 5 |
| PH | 212 | 4 | 5 |
| PK | 20 | 2 | 2 |
| TH | 62 | 1 | 1 |
| Total | 444 | 15 | 14 |

Twelve of the 14 labels without customers also lack an order; categories overlap and must not be added as independent record counts. The unmatched records occur in multiple countries, not just one. The cleaned files establish that these gaps are present in the supplied cleaned data. They do not establish whether extraction, source-system timing, or another cause produced them. Obtain and compare the original source records and extraction rules with the data owner before attributing a cause; do not invent matches or delete rows.

## Queries to repeat in MySQL

Run against the intended database using read-only credentials:

```sql
SELECT 'customers' AS table_name, COUNT(*) AS n FROM customers
UNION ALL SELECT 'orders', COUNT(*) FROM orders
UNION ALL SELECT 'labels', COUNT(*) FROM labels;

SELECT COUNT(*) AS labels_without_orders
FROM labels l LEFT JOIN orders o
  ON l.country_code = o.country_code AND l.order_id = o.order_id
WHERE o.order_id IS NULL;

SELECT COUNT(*) AS orders_without_labels
FROM orders o LEFT JOIN labels l
  ON o.country_code = l.country_code AND o.order_id = l.order_id
WHERE l.order_id IS NULL;

SELECT COUNT(*) AS labels_without_customers
FROM labels l LEFT JOIN customers c
  ON l.country_code = c.country_code AND l.customer_id = c.customer_id
WHERE c.customer_id IS NULL;
```

Counts are a historical baseline and can change when new batches are accepted. The SQL is provided for re-verification and was not executed during this documentation update.

## Batch rows that do not fit the schema

The PBI 02 cleaner keeps missing or invalid feature values as missing, while several columns here are NOT NULL. The PBI 10 pipeline now checks each cleaned batch before loading it (`src/pbi10_db_contract.py`): missing required columns, nulls in NOT NULL columns and over-length text stop the whole batch with a message naming the table, column and count. Nothing is substituted and no part of that batch is loaded. Loading runs in strict SQL mode inside one transaction (`src/pbi10_load.py`), so numeric or date values the database rejects also roll the batch back.

The current loader is `src/pbi04_load.py`. `pbi04/load_cleaned.py` is the original 17 November version, kept as a record of the initial import.

## Status at 28 November

- The figures above rest on the 17 November live verification and the 22 November recount of the cleaned files. The live database was not queried again for this update.
- Network access has changed since the 17 November note; check the current access rules in the console before connecting.
- The owner reports that teammates reviewed the schema and reconciliation. No written review is attached to this document.
- The cause of the unmatched records is still unknown. They are kept in the database and excluded wherever a query or training step joins the tables.
