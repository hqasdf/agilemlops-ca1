# PBI 06 — AWS MLflow publication

## Current AWS publication

**Published and verified on 19 November 2026.** The original shared experiment was absent. The approved recovery used the existing AWS endpoint, with a synthetic readiness probe in Default before publication. The probe's parameter, metric and artifact bytes matched in two independent clients. The probe continued to pass between batches and at completion. This is operational verification of the current endpoint. Persistence across a server replacement was outside this check.

| Item | Result |
| --- | --- |
| Live experiment | [`PBI06_HanQian`, experiment 2](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/2) |
| Live parent run | `c56b220577344dfabc42cd4e9f897112` |
| Source parent | `939506bd3eb6434c8489d5871ea9778e` |
| Shared run count | 491 finished runs, one parent and 490 children |
| Run identity and metadata | Verified source IDs, names, parameters, metrics, timestamps and remapped parent links |
| Metric histories | 6,426 source events compared during publication |
| Logical artifacts | 1,046 files, 217,546,438 bytes |
| Physical artifact payload | 1,268 files and 217,572,967 bytes including part manifests |
| Large artifacts | 38, each reassembled and checksum-checked |
| Readiness probe | `7603f9db4de6453fb9e005d2221e2dd1` |
| Notebook display | AWS status cell verified with saved evidence, without running any data-loading cells |
| Automated tests | 15 focused tests passed |

All seven final runs were uploaded first. Their models and fitted preprocessors loaded from AWS with the saved thresholds. The remaining runs used batches of up to 24 with four workers and a three-second pause between batches. The publisher checked the readiness probe, experiment identity and prior run IDs between batches. The endpoint remained reachable and retained the records throughout this attempt. An independent read after publication confirmed exactly 491 unique source runs, the remapped links, the authorized parent artifacts, and the presence of all seven final artifact sets. The publisher successfully loaded all seven model, preprocessor and threshold combinations again at completion. The shared ID map reconciled every original run. Experiment 1, `PBI08_Ephraim`, remained present.

| Model family | Verified AWS run |
| --- | --- |
| catboost | [`c5a7fd06bf474c3cb727508268dc0811`](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/2/runs/c5a7fd06bf474c3cb727508268dc0811) |
| gradient_boosting | [`f172ec192c684e98b20f756b6fb15732`](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/2/runs/f172ec192c684e98b20f756b6fb15732) |
| hist_gradient_boosting | [`3fd4769de8d949ad811925cad24497e8`](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/2/runs/3fd4769de8d949ad811925cad24497e8) |
| lightgbm | [`94b6aa1dc4bb40b4ad1e8a2f62d35490`](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/2/runs/94b6aa1dc4bb40b4ad1e8a2f62d35490) |
| logistic_regression | [`66697e2d543a4556955bd06ef92899ad`](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/2/runs/66697e2d543a4556955bd06ef92899ad) |
| random_forest | [`e522b7d726884f14a38077354a808aa3`](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/2/runs/e522b7d726884f14a38077354a808aa3) |
| xgboost | [`983fa63c1ba246ddb0a1b87da98ebe7f`](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/2/runs/983fa63c1ba246ddb0a1b87da98ebe7f) |

The source has 512 runs. The 21 superseded runs were excluded. `predictions.npz` stayed local, and the 70 exported Random Forest copies omit retained per-row training weights. The original binaries were preserved. Publications can resume by source tags and the recorded destination. This verification read the shared tree without repeating the upload. Models were not fitted again and the sealed holdout was not accessed.

## Prior outage and rename

**Tracking fix implemented. AWS retention and final verification are blocked.** AWS acknowledged and returned the 491-run tree and all approved files before the endpoint stopped responding. The final model-load check failed on the seventh model with connection refused. A direct read then returned HTTP 502. After health recovered, the server listed only `Default`; both experiment 2 and the pre-existing Nithish experiment 1 were missing. The earlier assigned IDs are retained below as attempt evidence. The current endpoint cannot retrieve them.

## Experiment rename and recovery

The local experiment was renamed in place from `PBI06_SevenFamily_Rebuild` to `PBI06_HanQian`. The old AWS publication used the former name. Both names are absent on the current AWS server. The local experiment kept ID **5** and all **512 runs**, including the **491-run** publication tree. Before and after checks matched for the run records, tags, parameters, metric history and artifact inventory. The rename and recovery checks passed **13 focused tests**, and the updated notebook ran **46 cells** in **29.7 seconds** in reuse mode. The shared rename command confirmed the original experiment is missing and created no replacement. Confirm the replacement backend and obtain approval before using `--recover-replacement`. If the original store returns, `--rename-original` checks the archived AWS IDs before renaming it.

## Tracking fix

`research_stages()` passed an explicit SQLite destination to `_tracking()`, so it won over the notebook's AWS environment value. Legacy tuning and threshold code also specified SQLite explicitly, and three evidence readers changed the fluent writer's global destination. `src/model_evaluation.py` had no tracking state.

The explicit write destination or `MLFLOW_TRACKING_URI` now selects the writer, with the local URI used as a fallback when neither is configured. An unavailable configured server raises an error. Evidence readers use explicit clients. The historical evidence migration pins the exact prior parent and fingerprint and both exact post-patch file hashes. It recomputes the prior digest with the current authorized development input, package versions and unchanged source files. Other evidence follows the ordinary strict fingerprint check.

## Publication scope and results

| Check | Result |
| --- | --- |
| Local experiment | `PBI06_HanQian`, SQLite experiment 5 |
| Local source parent | `939506bd3eb6434c8489d5871ea9778e` |
| Local records | 512 total, 491 in scope and 21 superseded records omitted |
| AWS records acknowledged before interruption | 491, one parent and 490 children |
| AWS records retrievable from the current endpoint | 0 for this experiment |
| AWS experiment ID originally assigned | 2, currently unavailable |
| AWS parent ID originally assigned | `632079cd81914408b10f6a84202a5baa` |
| Source inventory for the completed tree | 1,044 files, 238,521,106 bytes |
| Approved logical payload | 1,046 files, 217,546,450 bytes |
| Full AWS artifact load | 6 of 7 final models, with HistGradientBoosting's final load interrupted |
| Repeat idempotence | Interrupted attempts resumed on existing tags. Final zero-duplicate repeat is pending restoration. |
| Notebook | 46 code cells in 29.55 seconds, zero errors |
| Automated tests | 9 focused tests passed, with no model fitting |

The run upload checked all 491 source references, remapped the parent relationships and compared 5,474 parameters, 6,426 metric history events and downloaded artifact checksums before the final seven-model reload. The logical payload includes two current reports and the source-to-shared ID map. Total physical file count after chunking and network retry bytes could not be rechecked after the backend disappeared.

[Previously assigned AWS dashboard (currently unavailable)](http://mlflow-aws-eb.eba-9bnrxizu.us-east-1.elasticbeanstalk.com/#/experiments/2).

| Model family | Local run ID | Previously assigned AWS run ID | Threshold |
| --- | --- | --- | ---: |
| catboost | `652ffc4da1894319aa859515e14c2fce` | `22c7b3c521c74f3fa494ffb6dce28f3d` | 0.5762911561144178 |
| gradient_boosting | `5e4b1bd4257f4e98920647519f59550c` | `e5cef5ab61004e1d844e2beb46fb4ab1` | 0.528222223685986 |
| hist_gradient_boosting | `bec883e5805e4614aaabe32dc0db1f15` | `89d536d653f544bc85c5d8161de8e59a` | 0.5457061396298276 |
| lightgbm | `8fd15510396444e29e0308f423fe97df` | `dea0033b278446febcca5e4846ed9baf` | 0.5805976492906365 |
| logistic_regression | `2a91d0ecab36428989a0efecc96c80e6` | `51fc435ee7534ff4a1d210216e25f15a` | 0.4893435537815094 |
| random_forest | `adc11d5eedb54dc9a3086cad7e6c4270` | `f0f86bb458a24acc80fd5ab4d146074f` | 0.5109967277392646 |
| xgboost | `289945f86f2449ea8fe7ff2421249eaa` | `1471f002db7b4995b22a27e7710e4525` | 0.5479568839073181 |

## Privacy and transfer handling

- `predictions.npz` (26,286,946 bytes) stays local because it contains individual calibration labels and model scores.
- All 70 exported Random Forest binaries omit the retained per-row training weight array. The unchanged fitted tree state and synthetic inference were checked. The original local bytes remain intact.
- JSON and run metadata were screened for identifier fields and credential patterns. No raw CSVs or sealed test features or labels were read by the publisher. The historical appendix was omitted from the shared copy of the comparison report.
- The proxy rejected an oversized artifact with HTTP 413. The publisher resumed on existing run IDs and split oversized files into 768 KiB parts with checksum manifests. I corrected the downloader's local directory handling for the AWS SDK and verified reassembly during the transfer.

## Verification and recovery

A live check resolved the writer to AWS and confirmed that finding the local experiment via an explicit SQLite client left the write destination active. The updated notebook ran in default reuse mode after the incident and showed that the shared experiment was missing, along with the previously assigned IDs. It did not create any run. The local store still contains all 512 rebuild-experiment records.

Continued writes were paused after the endpoint reported only `Default`. I did not change the Nithish experiment or AWS infrastructure. The publisher now records the shared destination and parent and stops before creating replacement runs if the recorded experiment or parent is missing. Restore or reconnect the original backend, then run `python -m src.pbi06_publish` with the shared URI configured to finish the model reload and live duplicate check. If the store was permanently replaced, establish the intended destination for a new publication first. Actual service durability and PBI 10 remain outside this verification.

## Changed files

`src/train_hq.py`, `src/pbi06_research.py`, `src/pbi06_tracking.py`, `src/pbi06_publish.py`, `tests/test_pbi06_tracking.py`, `ca1 hq file.ipynb`, `README.md`, and this report. PBI 01–03 notebook cells, the kernel setting, shared preprocessing, model results, local MLflow and teammate code were preserved.
