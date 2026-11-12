# PBI 01 findings and handoff

The evidence is in `lab1.ipynb`; `data_quality_register.csv` lists issue counts, impacts and proposed handling. These results describe the supplied historical snapshot, not new batches. The CA1 specification defines fraud as 1 and non-fraud as 0, and describes review before shipment. No source data was altered.

## Scope and evidence

The CSVs are next to the notebook: customers has **901,312 × 9**, orders **949,889 × 8**, and labels **950,318 × 4**. No table has exact duplicate rows. Customer IDs are unique in this snapshot. Orders and labels each have three extra rows with repeated order IDs: the repeats occur in different countries. After country case normalization, `(country_code, order_id)` is unique in each order-grain table, and `(country_code, customer_id)` is unique for customers. One repeated label order ID has different fraud labels across countries; these are different composite keys and should not be treated as a label contradiction.

A validated composite outer join yields **949,874 matching order/label records**, **444 label-only records**, and **15 order-only records**. A validated customer join yields **950,304 matching label rows**, **14 label rows without a matching customer**, and **279,054 customers without labels**. The last group may simply have no order in the extract. Country agrees on unambiguous IDs after normalization; without normalization casing can appear to be a conflict. There are no invalid nonmissing dates in either date column.

Missing values comprise **712 first-order timestamps (0.079% of customers)**, **21,501 payment methods (2.264% of orders)**, and **one item count**. Country codes contain 390 lowercase customer entries and 301 lowercase label entries; collection type has 85 lowercase `delivery` entries alongside `Delivery`; mobile verification has 165 `yes` entries. There are **199,687 orders (21.022%)** with refund value greater than order value; this is a business-rule anomaly, not proof of an invalid transaction. Date and currency semantics, adjustments and source lineage require confirmation.

The label distribution is **104,504 fraud (10.997%)** and **845,814 non-fraud (89.003%)**. Several amounts and counts are right-skewed: order value skewness is approximately 6.62, customer total payment 8.68, and item count 41.04. Extreme values and IQR flags need investigation rather than automatic deletion, especially where IQR is zero. Fraud-rate charts show denominators and use matched labeled orders; they are descriptive associations, not causal evidence.

## PBI 02: recommended cleaning

Normalize country and collection casing on working copies, validate permissible categories and confirm whether `yes` means true. Validate composite keys before joins, investigate and quarantine unresolved unmatched records, and preserve missingness indicators. Do not merge on order ID alone or infer missing labels. Confirm refund and currency rules before changing values, and review extreme values with domain owners. Treat the unused-customer count as coverage information, not a deletion rule. Build repeatable validation for later batches; no cleaning pipeline is implemented here.

## PBI 03: recommended transformations and modeling safeguards

Establish feature timestamps for the pre-shipment decision. Exclude the target, refund value, refund-related history and any snapshot aggregates that may include current/future events until availability is verified. Construct historical aggregates strictly before each prediction and assess temporal validation and repeat-customer overlap. Keep identifiers for joins instead of treating them as numeric features. Consider `log1p`, robust scaling, explicit missing indicators and categorical encoding, fitted only on training folds. Compare class weighting and training-only resampling; retain natural prevalence in validation/test and report precision, recall, precision-recall performance and review capacity alongside accuracy.

## Scope of completion

This PBI supplies exploration and recommendations. The specification also requires actual cleaning, transformations and automated processing across later project work; those implementations remain for their respective PBIs. Issue #1 asks for a commit and execution evidence: execution is recorded in the notebook, while a commit is intentionally left pending explicit authorization.
