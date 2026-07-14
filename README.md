# Medicare DME Fraud Detection — ML Pipeline

> Detecting fraudulent Durable Medical Equipment (DME) claims in Medicare using machine learning. The project includes two modeling approaches: a **Positive-Unlabeled (PU) Learning** pipeline for settings where confirmed fraud labels are scarce, and a production-grade **LightGBM** classifier registered to MLflow.

---

## Problem Statement

Traditional supervised fraud detection requires a clean set of both positive (fraud) and negative (legitimate) labels. In practice, **we only have a small set of confirmed fraud claims** — the rest are unlabeled, not confirmed clean. Training a standard binary classifier on this data introduces massive label noise.

This project addresses that challenge in two stages:

1. **PU Learning** (`DME_TrainingData.ipynb` + `DME_PULearning.ipynb`) — treats confirmed fraud as reliable positives and all unlabeled claims as an unknown mixture, learning fraud patterns without assuming unlabeled = legitimate.
2. **LightGBM production model** (`DME_Features_Pipeline.ipynb` + `DME_Model_Training_Pipeline.ipynb`) — trained on labeled data from BigQuery at scale using PySpark on Databricks, with MLflow model registry for production serving.

---

## Pipeline Architecture

```
Medicare Claims Data
        │
        ├── SQL Server (CCLF6 Part B, 2019–2025)        ← PU Learning pipeline
        └── Google BigQuery (ML_DATASETS, 2021–2025)     ← LightGBM pipeline
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              Feature Engineering (5 Levels)          │
│                                                      │
│  • Claim Level        — amounts, CPT codes, dates    │
│  • Beneficiary Level  — utilization history          │
│  • Billing NPI Level  — provider behavior patterns   │
│  • Ordering NPI Level — referral patterns            │
│  • NPI × Bene Level   — provider-patient concentration│
└─────────────────────────────────────────────────────┘
        │
        ├── PU Learning Model    (label-scarce setting)
        └── LightGBM Classifier  (production, MLflow)
        │
        ▼
   Fraud Risk Scores  — P(fraud) ∈ [0, 1] per claim
```

---

## Feature Engineering

Features are engineered at **5 levels** with **30 / 90 / 365-day lookback windows**:

| Level | Key Signals |
|---|---|
| **Claim** | Charge amount, line count, CPT mix, claim duration, adjustment status, high-risk HCPCS flag |
| **Beneficiary** | DME utilization history, provider shopping behavior, days since last claim |
| **Billing NPI** | Volume trends, patient concentration, high-risk CPT share, new patient rate |
| **Ordering NPI** | Referral volume, top CPT patterns, new patient acquisition rate |
| **NPI × Beneficiary** | Repeat-relationship claim counts and paid amounts per provider-patient pair |

**Provider-patient visit verification:** Cross-checks Part A/B claims to verify whether the ordering provider ever had a clinical encounter with the patient (`bene_seen_ordnpi_5years`) — a strong fraud signal when absent.

---

## LightGBM Model Performance

**Test set (fully out-of-time, time-based 70/15/15 split):**

| Metric | Value |
|---|---|
| AUC | **0.9989** |
| PR-AUC | **0.97** |
| KS Statistic | **0.97** |

**Top-K Recall — Audit Prioritization:**

| Review % of Claims | Fraud Captured |
|---|---|
| Top 1% | 52% |
| Top 3% | **98%** |
| Top 5% | **99%** |

> By auditing only the top 3% of highest-scored claims, the model surfaces **98% of all fraudulent cases**.

---

## Repository Structure

```
├── DME_TrainingData.ipynb             # Feature engineering (SQL Server / pandas)
├── DME_PULearning.ipynb               # PU Learning model training
├── DME_Features_Pipeline.ipynb        # Feature engineering at scale (PySpark / BigQuery)
├── DME_Model_Training_Pipeline.ipynb  # LightGBM training + MLflow registry
├── model_card.md                      # Model performance, limitations, versioning
└── README.md
```

---

## Tech Stack

| Tool | Usage |
|---|---|
| Python (pandas, numpy) | Feature engineering & data processing |
| PySpark | Distributed feature engineering on Databricks |
| SQL Server + SQLAlchemy | Data extraction and staging (PU Learning pipeline) |
| T-SQL | Complex lookback window aggregations |
| Google BigQuery | Data warehouse for production pipeline |
| LightGBM | Gradient-boosted classifier |
| MLflow | Experiment tracking, model registry, production serving |
| PU Learning | Fraud model training under label scarcity |

---

## Setup

See `model_card.md` for full model versioning, known limitations, and production usage details.
Credentials are always loaded from external files or environment variables — never hardcoded.
