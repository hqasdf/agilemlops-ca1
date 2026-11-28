# PBI 10 and PBI 11 - full-history verification on a local replica

Run on 23-24 Nov 2026 on a Windows laptop (16 logical cores, 15 GB RAM) with Python 3.13 and the pinned requirements. The database was a throwaway local MySQL 8.0.44 instance with TLS and certificate-plus-hostname verification, loaded with the **complete** cleaned history using the PBI 04 loader and schema. MLflow was a local server on 127.0.0.2. Nothing was run against AWS RDS or the shared MLflow service, because no AWS session was available. Treat timings as indicative of this laptop only.

## Data loaded

| Table | Rows loaded from cleaned CSVs | Same count as RDS (PBI 04 evidence) |
| --- | ---: | --- |
| customers | 901,312 | yes |
| orders | 949,889 | yes |
| labels | 950,318 | yes |

Unmatched relationships also match RDS: 444 labels without orders, 15 orders without labels, 14 labels without customers. The sealed manifest built from the same history has 144,126 keys, matching the production readiness check of 23 Nov.

## PBI 10 pipeline on full history (synthetic 120-row batch, `practice/pbi10`)

| Run | Result | Total | Download+validate+load | Read+feature prep | Han Qian | Nithish | Ephraim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (no `PYTHONUTF8`) | 3 of 3 routines failed on Windows | 230.5 s | - | - | `UnicodeEncodeError` | `MlflowException` (loopback URL rejected by guard) | `UnicodeEncodeError` |
| 2 (new batch) | success, 7 runs | 280.4 s | 4.4 s | 69.2 s | 25.0 s | 25.3 s | 160.4 s |
| 3 (repeat batch, config file only) | success, 7 runs, 120 unchanged per table | 512.9 s | 4.5 s | 63.5 s | 26.9 s | 143.7 s | 278.4 s |

- Best full-history time was 280 s, above the 180 s target. Run 3 was 513 s: Nithish's second model took 121 s instead of 2 s and Ephraim's decision tree 132 s instead of 11 s, with only 1.4 GB of RAM free, so contention on this laptop dominates the variance.
- Profiling in isolation: reading the three tables 27-30 s; `prepare_training_data` about 42 s (about 16 s iterating Arrow strings and about 10 s on 3.5 million per-row SHA-256 key hashes); Ephraim's two models 36 s. Summing quiet-machine figures gives about 155 s, so 180 s is plausible but **not demonstrated**.
- Training used 805,847 rows (historical rows before the sealed boundary plus 96 new rows). Validation held only 24 rows, all from the synthetic batch, because the code takes the validation split from the new batch. A real batch with more dates gives a larger validation set.
- Repeat batch: 0 inserted, 120 unchanged per table; ledger unchanged.

## Failure scenarios (PBI 10 task "Test repeat batches and failure scenarios")

Database rows before and after were identical (901,432 / 950,009 / 950,438) and the ledger file was unchanged after all three.

| Scenario | Outcome |
| --- | --- |
| Unreachable URL (HTTP 404) | `failed` at stage `prepare` (0.16 s) |
| Same order key with a changed value | `failed` at stage `load`; whole batch rolled back (0.63 s) |
| Batch containing a sealed historical order | `failed` at stage `prepare` (0.38 s) |

## PBI 11 acceptance criteria

| Criterion | Evidence |
| --- | --- |
| Read DB host, port, name, user, password, MLflow URL and other settings from environment variables or config files | `src/pbi10_config.py` reads both (`--config` or `CA1_CONFIG_FILE`; environment overrides file). Run 3 above used a settings file only. The PBI 04 loader reads the same names as defaults. |
| Example configuration and instructions, no live credentials, no passwords in logs | `.env.example`, `docs/PBI11_CONFIGURATION.md`. The password is masked in the config object, never in error text; a search of the error output for the real password found 0 matches. |
| A connection setting can change without editing Python | Same code with port 33307 connected; the same code with `CA1_DB_PORT=33999` in the file failed to connect (`OperationalError`). Unit test `test_changing_a_setting_changes_the_connection_without_code_edits`. |
| Missing configuration gives an actionable error | With no settings: `Missing required settings: CA1_DB_HOST, ... Set them as environment variables, or put KEY=VALUE lines in a config file and pass it with --config PATH or CA1_CONFIG_FILE=PATH.` Also tested for a partially filled file and a missing file path. |

Open: the teammate notebooks still hold a default MLflow URL in a cell; teammate review and PR evidence are not yet recorded.

## Tests

`tests/test_pbi10_config.py` 9 tests pass. All other PBI 10 tests (80 in total, excluding `test_pbi06_tracking.py`, which imports POSIX-only `fcntl`) pass; 5 tests in `tests/test_train_hq.py` fail on Windows only because a temporary SQLite file is still locked when the temp directory is deleted.
