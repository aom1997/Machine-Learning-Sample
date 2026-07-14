# Model Card – DME Fraud Detection (LightGBM)

## Model Overview

| Item | Detail |
|---|---|
| **Model name** | `dme_lgbm` |
| **Type** | Binary classifier — outputs P(fraud) ∈ [0, 1] |
| **Algorithm** | LightGBM (gradient-boosted decision trees) |
| **Version in production** | v2 (`@champion`) |
| **Baseline anchor** | v1 (`@baseline`) |
| **Registry** | MLflow Unity Catalog (Databricks) |

---

## Problem Statement

Durable Medical Equipment (DME) fraud is a significant source of loss in Medicare billing. Fraudulent providers overbill, bill for equipment never delivered, or bill for medically unnecessary items. Manual auditing is expensive and slow.

This model scores every DME claim with a fraud probability so that audit teams can **prioritize the highest-risk claims first**, dramatically increasing the efficiency of fraud investigations.

---

## Training Data

| Item | Detail |
|---|---|
| **Source** | Medicare Part B DME claim line items (BigQuery) |
| **Date range** | 2021-01-01 – 2025-12-31 |
| **Label source** | `fraud_flag` — pre-computed from a known fraudulent NPI/company list, joined upstream |
| **Split method** | Time-based quantile split on `clm_thru_dt` (no data leakage) |
| **Train / Validation / Test** | 70% / 15% / 15% |
| **Test set** | Fully out-of-time — no overlap with training period |

### Class Imbalance
Fraud cases are a small minority of all claims. `scale_pos_weight` was computed from the training set class ratio and passed to LightGBM to compensate.

---

## Features

100+ features engineered across six dimensions:

| Dimension | Examples |
|---|---|
| **Claim-level** | Charge amount, claim duration, high-risk CPT flag, day-of-week |
| **Beneficiary** | Rolling 30/90/365-day unique provider count, paid amounts, NPI concentration ratio |
| **Billing NPI** | Rolling claim volume, new patient rate, top-CPT concentration, avg charge per patient |
| **Ordering NPI** | Same as billing NPI for the ordering provider |
| **Billing NPI × Beneficiary** | Repeat-relationship claim count and paid amount per provider-patient pair |
| **Ordering NPI × Beneficiary** | Same for ordering provider-patient pairs |

**Leakage prevention:** payment amount columns (`paid_amt`), patient-level payment ratios, and downstream risk scores were excluded — these are only available after adjudication, not at real-time scoring.

---

## Performance

### Test Set (fully out-of-time)

| Metric | Value |
|---|---|
| **AUC** | **0.9989** |
| **PR-AUC** | **0.97** |
| **KS Statistic** | **0.97** |

### Validation Set

| Metric | Value |
|---|---|
| AUC | 1.00 |
| PR-AUC | 0.98 |
| KS Statistic | 0.97 |
| Accuracy | 1.00 |
| F1-optimal threshold | 0.50 |

### Top-K Recall (Audit Prioritization)

This is the most operationally relevant metric: **what fraction of all fraud cases are captured if auditors review only the top-K% highest-scored claims?**

| Review % of Claims | Fraud Captured (Test) | Fraud Captured (Validation) |
|---|---|---|
| Top 1% | 52% | 27% |
| Top 3% | **98%** | — |
| Top 5% | **99%** | — |

**Business interpretation:** By auditing only the top 3% of highest-scored claims, the model surfaces **98% of all fraudulent cases** — allowing audit teams to focus effort where it matters most instead of reviewing claims at random.

---

## Model Configuration

```python
LGBMClassifier(
    objective          = "binary",
    learning_rate      = 0.05,
    num_leaves         = 63,
    n_estimators       = 2000,       # with early stopping (patience=100)
    min_child_samples  = 50,
    reg_alpha          = 0.1,
    reg_lambda         = 0.1,
    feature_fraction   = 0.8,
    bagging_fraction   = 0.8,
    bagging_freq       = 5,
    scale_pos_weight   = <class ratio>,
    eval_metric        = "average_precision",
)
```

---

## Production Usage

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/your-gcp-project.dme_fraud.dme_lgbm@champion")
scores = model.predict(claims_df)  # returns P(fraud=1) per claim, shape (n,)
```

The pyfunc wrapper handles preprocessing automatically:
- `bene_days_last_claim` NaN → 999
- inf / -inf / NaN in numeric columns → 0
- Extra input columns silently dropped
- Missing trained features filled with 0

---

## Known Limitations

- **Label quality dependency:** The `fraud_flag` label is derived from a known fraudulent provider list. Novel fraud patterns not yet on the list will not be captured in training.
- **Temporal drift:** Provider billing behavior changes over time. The model should be retrained periodically as new labeled data becomes available.
- **Top-1% recall gap (valid vs. test):** Validation top-1% recall (0.27) is lower than test (0.52), suggesting score distribution differences between the two splits. This is expected with time-based splits and does not indicate leakage.
- **Scope:** Trained on DME claims only. Not applicable to other claim types (Part A inpatient, Part B professional, etc.).

---

## Versioning

| Alias | Version | Notes |
|---|---|---|
| `@champion` | v2 | Current production model |
| `@baseline` | v1 | Frozen anchor for regression testing — never retargeted |
