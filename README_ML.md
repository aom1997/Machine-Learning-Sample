# DME Fraud Detection – Machine Learning Pipeline

An end-to-end machine learning pipeline for detecting fraud in Durable Medical Equipment (DME) Medicare insurance claims. Built on Databricks with PySpark and LightGBM, the system engineers 100+ features from raw claims data, trains a gradient-boosted classifier, and registers the model to MLflow for production serving.

---

## Pipeline Overview

```
BigQuery (Raw Claims)
        │
        ▼
┌─────────────────────────────┐
│  DME_Features_Pipeline      │  ← PySpark feature engineering
│  (DME Features Pipeline.ipynb) │
└─────────────────────────────┘
        │  ~100+ features per claim
        ▼
  BigQuery: DME_Fraud_Training
        │
        ▼
┌─────────────────────────────┐
│  DME_Model_Training_Pipeline│  ← LightGBM training + MLflow
│  (DME Model Training Pipeline.ipynb) │
└─────────────────────────────┘
        │
        ▼
  MLflow Model Registry
  (@champion alias → production scoring)
```

---

## Notebooks

### 1. `DME Features Pipeline.ipynb` — Feature Engineering (PySpark)

Reads raw Medicare Part B DME claim line items and engineers six feature tables, joined into one wide feature set keyed by `CUR_CLM_UNIQ_ID`.

**Feature groups:**

| Group | Description |
|---|---|
| **Claim-level** | Claim month, day-of-week, duration, charge/paid amounts, high-risk CPT flag, original vs. adjusted claim |
| **Beneficiary-level** | Rolling 30/90/365-day counts of unique billing NPIs, ordering NPIs, claims, and paid amounts per beneficiary; ratio features |
| **Billing NPI-level** | Rolling claim volumes, unique patient counts, new patient rates, top-CPT concentration, charge/paid ratios over 30/90/365 days |
| **Ordering NPI-level** | Same as billing NPI, but for the ordering provider |
| **Billing NPI × Beneficiary** | Repeat-relationship features: rolling counts/amounts for each (provider, patient) pair |
| **Ordering NPI × Beneficiary** | Same as above for ordering provider × patient pairs |

**Key engineering techniques:**
- **Range-join windowing** (`hint("range_join")`) for efficient 30/90/365-day rolling aggregations without Cartesian blowup
- **Two-step daily aggregation** — collapse to daily summaries first, then roll up over time windows to control shuffle cost
- **Window functions** (`ROW_NUMBER`, `PARTITION BY`) to rank top CPT codes by frequency, charge, and paid amount per provider
- **Derived ratios** — new patient %, risk CPT %, short-window vs. 365-day volume ratios
- **DBFS checkpointing** to Parquet after each major step to truncate Spark DAG lineage
- **Label join** (training mode) — `fraud_flag` left-joined from known fraudulent provider list; null → 0

**Output:** BigQuery table with 100+ features per claim, ready for model training.

---

### 2. `DME Model Training Pipeline.ipynb` — Model Training & Registry (LightGBM + MLflow)

Reads the engineered feature table from BigQuery, trains a LightGBM binary classifier, evaluates it, and registers the model to the MLflow Unity Catalog registry.

**Data split:**
- Time-based 70/15/15 quantile split on `clm_thru_dt` — test set is fully out-of-time to prevent leakage

**Leakage prevention:**
- Drops payment amount columns (`paid_amt`) — known after claim adjudication, not at scoring time
- Drops patient-level payment ratio columns for the same reason
- Drops all `risk_*` columns (downstream derived scores)

**Model configuration:**

```python
LGBMClassifier(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    n_estimators=2000,
    min_child_samples=50,
    reg_alpha=0.1, reg_lambda=0.1,
    feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5,
    scale_pos_weight=<computed from class imbalance>,
)
```

**Evaluation metrics:**
- AUC, PR-AUC (primary — handles class imbalance better than accuracy)
- KS statistic
- F1-optimal threshold search across [0.0, 1.0]
- Top-K recall analysis at 1%, 3%, 5%, 10%, 20%, 50%

**MLflow pyfunc wrapper (`DMEFraudModel`):**
- Bundles preprocessing + LightGBM booster into a single deployable artifact
- At inference: fills `bene_days_last_claim` NaN with 999, replaces inf/NaN with 0, selects only trained features by name
- Extra columns silently dropped; missing columns filled with 0 (production-safe)
- Input/output signature inferred and stored with the model

**Model registry:**
- Registered to MLflow Unity Catalog as `your-gcp-project.dme_fraud.dme_lgbm`
- `@champion` alias → current production version
- `@baseline` alias → v1 anchor for regression testing

**Production load:**
```python
model = mlflow.pyfunc.load_model("models:/your-gcp-project.dme_fraud.dme_lgbm@champion")
scores = model.predict(claims_df)  # returns P(fraud=1) per claim
```

---

## Technical Stack

| Layer | Technology |
|---|---|
| Distributed processing | PySpark (Databricks) |
| Data warehouse | Google BigQuery |
| Model training | LightGBM (`lgb.LGBMClassifier`) |
| Experiment tracking | MLflow (Databricks Unity Catalog) |
| Data manipulation | pandas, numpy |
| Evaluation | scikit-learn (AUC, PR-AUC, confusion matrix, F1) |

---

## Setup

### Prerequisites
- Databricks workspace with PySpark
- Google BigQuery access
- Google Cloud service account key file (see below)

### Credentials
**Never hardcode credentials.** Set the environment variable before running:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service_account.json"
```

Or in Databricks, store the key file path in a Databricks Secret and reference it via `dbutils.secrets.get(...)`.

### Configuration (top of each notebook)
```python
PROJECT_ID    = "your-gcp-project"
BQ_TABLE      = "your-gcp-project.ML_DATASETS.DME_Fraud_Training"
EXPERIMENT_PATH = "/Users/your-email/machine_learning/fraud_dme/dme_lgbm_experiment"
```
