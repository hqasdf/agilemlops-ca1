# PBI 09 Fraud Model Comparison and Registry Review

**Owner:** Nithish (`@nith-ish22`)
**Evidence snapshot:** 19 November 2026, 16:28–17:20 Singapore time
**Status:** Team review and registry verification pending
**Purpose:** Compare fraud models for human review, identify two defensible candidates and record what must be verified before choosing and registering a champion.

## Decision at a glance

The provisional shortlist is **Han Qian’s calibrated Random Forest** and **Han Qian’s calibrated CatBoost**, both from PBI 06. Random Forest is the provisional first choice when the team values catching more fraud and can handle roughly 814 reviews per day. CatBoost is the lower-workload alternative at roughly 628 reviews per day. These are recommendations, not a recorded Scrum decision or a production approval. The best business choice depends on actual review capacity, the cost of missed fraud and whether a real-time customer friction cost applies. Neither has been independently loaded in this review.

Ephraim’s tuned Random Forest is a high-priority challenger: the supplied draft reports average precision (AP) 0.5390 and recall 0.805 on an earlier run, but its fit protocol, current run and retrievable artifact need confirmation. Do not exclude it solely because its artifact was previously unavailable. Nithish’s v3 is also retained for team review; the source report and screenshots show different versions and row counts that need reconciliation.

**Current live state is unverified.** The shared server could not be reached from this review environment on 20 November. The uploaded images depict an earlier snapshot; the original draft also contains later text-only updates. Do not describe the registry as currently empty or any historical run as currently available without checking again. Historical run links below may stop working if the server is replaced. Local results and run IDs may differ from shared ones.

## Business comparison on the reported validation window

The source draft describes 144,978 orders from 15 March to 13 April 2023, including 19,740 fraud orders. The 30-day average is descriptive, not a forecast of future workload. Reported rows are from the 19 November draft and PBI 06 report; identical row counts and dates alone do not prove identical order IDs, labels, cleaning, feature timing or fitted preprocessing.

| Candidate and operating point | F1 | Precision | Recall | AP | False alarms | Missed fraud | Reviews per day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Han Qian Random Forest, calibrated threshold 0.511 | 0.5435 | 0.4915 | 0.6078 | 0.5233 | 12,410 | 7,743 | 814 |
| Han Qian CatBoost, calibrated threshold 0.576 | 0.5208 | 0.5330 | 0.5091 | 0.5250 | 8,804 | 9,690 | 628 |
| Nithish histogram gradient boosting v3, threshold approximately 0.711 | 0.5221 | 0.526 | 0.519 | 0.5197 | 9,244 | 9,500 | 650 |
| Ephraim tuned Random Forest, fixed threshold 0.5 | 0.5346 | 0.400 | 0.805 | 0.5390 | 23,796 | 3,856 | 1,323 |

Review counts are (true positives + false positives) divided by 30 and rounded. The Ephraim row uses a different or unconfirmed fitting and threshold protocol; its metrics are historical, from the supplied draft, and must not be ranked as a controlled like-for-like comparison.

Random Forest versus CatBoost: the reported confusion counts imply **1,947 more fraud orders caught** and **3,606 more legitimate orders flagged**, requiring **5,553 more reviews in the 30-day window** (about 185 more daily). If every review costs the same and each additional fraud caught avoids the same net loss, the illustrative break-even avoided loss is about **2.85 times the cost of one review**. This simplified comparison assumes equal costs and capacity, comparable rows, and no additional customer impact; it is not an agreed business cost estimate. At a 650-review daily capacity, the reported Random Forest operating point would exceed capacity; the team would need a separately calibrated capacity-constrained threshold and fresh evaluation rather than assume the table’s recall still applies.

AP and ROC-AUC measure score ranking across thresholds; F1, precision, recall and workload describe a particular threshold. AP does not make comparisons automatically fair when data, features or fitting differ. No confidence interval is available; do not treat a fixed F1 difference such as 0.02 as automatically statistical noise. Repeated selection on this validation period can make results optimistic. The historical project report also notes that full-history exploratory analysis happened before the holdout was established, so final test evidence has that limitation.

## Candidate evidence and limitations

### Proposed candidate 1: Han Qian calibrated Random Forest

The PBI 06 report records F1 0.543502, AP 0.523255, 11,997 true positives, 12,410 false positives and 7,743 false negatives. The provided screenshots show an earlier FAILED run with these metrics; the later 17:10 update in the supplied draft reports a FINISHED replacement run `def735e86a7f431bb03fc244d349f3d0` with a model folder, but that later update has no supplied screenshot and has not been rechecked. A folder listing alone does not prove a model can be loaded. Confirm the current run, model and preprocessing artifacts and perform a teammate load test. [PBI 06 report](../reports/PBI06_model_comparison.md); [earlier screenshot](pbi09_evidence/ss12_han_rf_failed.jpg).

### Proposed candidate 2: Han Qian calibrated CatBoost

The PBI 06 report records F1 0.520806, AP 0.524951, 10,050 true positives, 8,804 false positives and 9,690 false negatives. It is a useful lower-workload alternative to the Random Forest, not necessarily a better ranking model: the AP difference is small and uncertainty has not been measured. The supplied draft reports a later run `e27c9dcdda72428ca6b42e68bfceaf46`; the run and artifact load are not independently verified in this review.

### Challenger: Ephraim tuned Random Forest

The original draft reports AP 0.5390, recall approximately 0.805 and 23,796 false positives, with an earlier logged-model upload failure. A later draft update says the experiment was republished and training was still in progress. Neither status establishes the present state. Verify fit dates and row counts, feature list, threshold rule, current run, model URI and loadability. Re-evaluate the same candidate under an agreed calibration and validation protocol before final selection. Its higher recall may be valuable if review capacity and fraud-loss costs justify the additional workload.

### Challenger: Nithish v3 and logistic baseline

The supplied screenshot records v3 F1 approximately 0.5221 and AP 0.5197; the project report contains a different shared rerun and historical result. The draft also notes a development-loader row-count discrepancy. Trace code commit, data identity, preprocessing, split and model settings before attributing the differences or using an old number as the current result. Nithish v1 and v2 are documented in PBI 07 as the same fitted logistic model with different thresholds. Equal rounded AP alone is not proof that two model artifacts are identical. [PBI 07 report](../reports/PBI07_model_comparison.md); [Nithish metrics screenshot](pbi09_evidence/ss07_nithish_metrics.jpg).

## Fair selection protocol

1. Freeze the eligible development data and identify validation records by order key and a privacy-safe fingerprint. Compare the actual rows and labels, not only dates and totals.
2. Confirm point-in-time availability of each feature, including `first_order_datetime`, and record the same preprocessing and fit/calibration periods. Resolve the loader discrepancy before declaring the runs comparable.
3. Use training or calibration data to choose thresholds. Pre-agree an operating objective with the team: a review capacity and/or defensible costs for missed fraud, review and customer friction. Report F1, AP, precision, recall and confusion counts at the chosen threshold.
4. Compare at the same review budget where possible, or report the workload differences. Assess subgroup error rates where sample sizes support them. Use the same locked validation window and note repeated model selection.
5. Confirm each candidate’s run, code version, parameters, threshold, fitted preprocessor and actual retrievable model artifact. Treat v1/v2 or threshold-optimised variants as the same fitted model only when provenance or artifacts establish this.
6. Record the team’s actual selection, including who attended and the reason. Freeze the champion decision before a final holdout assessment. Historical full-data EDA limits how strongly the test can be described as untouched.

## Scrum meeting record

The user confirmed the Scrum meeting was held on **19 November 2026**, attended by **Nithish, Ephraim and Han Qian**. Time, meeting evidence, which first model each member presented, the metrics actually discussed, the team’s top-two decision and its champion have not been supplied or independently verified. Add these from the real meeting notes or ask the participants; do not turn the provisional shortlist into a claimed meeting decision.

## Registry and verification record

The supplied screenshot `ss05_registry_empty.jpg` shows an empty registry at the earlier snapshot, not necessarily now. The shared server’s current state could not be checked during this review. Do not claim model registration, promotion or a successful load until demonstrated on the active service.

| Evidence required | Status | Record when verified |
| --- | --- | --- |
| Team-reviewed top two | Pending confirmation | Meeting evidence, exact model and threshold for each |
| First registered candidate | Not verified | Name, version, model URI, source run, metrics and artifact |
| Second registered candidate | Not verified | Name, version, model URI, source run, metrics and artifact |
| Champion designation | Not verified | Team decision, registry alias/version and timestamp |
| Teammate load test | Not verified | Tester, environment, model URI and prediction outcome |
| Later model review | Not verified | Date, new candidate, comparison and decision |
| Sprint/Scrum and review evidence | Incomplete | Assigned tasks, outputs, reviews, owners and follow-up |

For MLflow 3 use the actual logged model URI returned by the logging process where applicable (`models:/<model_id>`); some older runs may instead store a model under `runs:/<run_id>/...`. Inspect the source before registration. A threshold is operational configuration and must travel with the model; registering an estimator without its threshold does not by itself reproduce the decision policy.

## Next actions

| Owner | Action | Evidence needed |
| --- | --- | --- |
| Nithish | Confirm Scrum time, meeting notes and actual team decision with the group | Notes or board/review links |
| Han Qian | Confirm current Random Forest and CatBoost run IDs, preprocessing, artifact URIs and load tests | Run links and successful model loads |
| Ephraim | Republish or identify current tuned RF, then confirm fit rows, features, threshold and artifact | Current run link, protocol and successful model load |
| Nithish | Reconcile PBI 07 loader counts and differing v3 metrics with current code and data | Reproducible run manifest and corrected comparison |
| Team | Agree capacity and cost assumptions, compare under a common protocol and record a decision | Dated comparison and decision |
| PBI 12 owner | Improve and verify durable metadata and artifact storage before relying on shared history | Actual RDS/S3 configuration and replacement test |

When the team chooses, register the actual two models, set the agreed champion alias and verify that another member can load the champion and apply its threshold. Repeat and document the review for future model versions. Do not mark PBI 9 complete until its issue criteria and actual evidence are satisfied.

## Evidence provenance

- Source supplied by Nithish: original `PBI09_Model_Comparison.docx`, with initial 19 November screenshots and text-only updates until approximately 17:20. Its historical states are not current-state proof.
- Screenshot archive: `reports/pbi09_evidence/`; the files are dated evidence from the supplied ZIP. Screenshot 05 shows the registry empty in that earlier snapshot; screenshots 10 and 12 show earlier failed uploads. Later claims must be verified separately.
- [PBI 09 issue](https://github.com/hqasdf/agilemlops-ca1/issues/9) and [PBI 12 issue](https://github.com/hqasdf/agilemlops-ca1/issues/12).
- [PBI 06 model comparison](PBI06_model_comparison.md) and [PBI 07 model comparison](PBI07_model_comparison.md), as present on the team integration branch.
- The draft’s November 19 live links and the currently terminated or unreachable service are not durable evidence. Save run IDs and team decisions outside the service.
