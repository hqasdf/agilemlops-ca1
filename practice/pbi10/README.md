# PBI 10 synthetic rehearsal batch

All records in these three CSVs are generated, fictional data. Each file has 120 records; identifiers are unique and the labels refer to the corresponding customer and order. Dates range from 2026-01-01 onwards and labels include both fraud classes.

Use only in an isolated rehearsal database. Do not load these records into the shared amlops_ca1 tables. The pipeline and three-minute requirement have not yet been verified against this batch. A chronological split may assign every new row to validation; successful loading alone does not prove retraining used new records.
