# 🏥 Medicare DME Fraud Detection — PU Learning Pipeline

> Detecting fraudulent Durable Medical Equipment (DME) claims in Medicare using **Positive-Unlabeled (PU) Learning** — a machine learning approach designed for real-world fraud detection where confirmed fraud labels are scarce but reliable.

---

## 🔍 Problem Statement

Traditional supervised fraud detection requires a clean set of both positive (fraud) and negative (legitimate) labels. In practice, **we only have a small set of confirmed fraud claims** — the rest are unlabeled, not confirmed clean. Training a standard binary classifier on this data would introduce massive label noise.

This project applies **PU Learning** to treat confirmed fraud as the positive class and all unlabeled claims as an unknown mixture, allowing the model to learn fraud patterns without assuming unlabeled = legitimate.

---

## 🏗️ Pipeline Architecture

```
Medicare Claims Data (SQL Server)
        │
        ▼
┌─────────────────────────────┐
│   Data Extraction & Staging  │  ← CCLF6 Part B DME claims (2019–2025)
│   temp_zeming.dbo.dme        │    Filtered to assigned beneficiaries
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              Feature Engineering (5 Levels)          │
│                                                       │
│  • Claim Level        — amounts, CPT codes, dates    │
│  • Beneficiary Level  — utilization history          │
│  • Billing NPI Level  — provider behavior patterns   │
│  • Ordering NPI Level — referral patterns            │
│  • NPI × Bene Level   — provider-patient concentration│
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   PU Learning Model      │
│   (Positive-Unlabeled)   │
└─────────────────────────┘
        │
        ▼
   Fraud Risk Scores
```

---

## ✨ Key Features

### Multi-Level Feature Engineering
Features are engineered at **5 levels** with **30 / 90 / 365 day lookback windows** to capture both short-term spikes and long-term behavioral patterns:

| Level | Key Signals |
|---|---|
| **Claim** | Charge amount, line count, CPT mix, claim duration, adjustment status |
| **Beneficiary** | DME utilization history, provider shopping behavior, days since last claim |
| **Billing NPI** | Volume trends, patient concentration, high-risk CPT share, new patient rate |
| **Ordering NPI** | Referral volume, top CPT patterns, new patient acquisition rate |
| **NPI × Beneficiary** | What % of a provider's total volume is concentrated on a single patient |

### High-Risk CPT Flagging
Claims containing HCPCS codes from a curated high-risk DME code list are flagged at both the claim and provider level.

### Provider-Patient Visit Verification
A cross-check against Part A and Part B claims verifies whether the **ordering provider has ever had a clinical encounter** with the patient — a strong fraud signal when no prior relationship exists (`bene_seen_ordnpi_5years`).

### PU Learning
Standard supervised learning assumes unlabeled = legitimate, which leads to a heavily contaminated negative class. PU Learning addresses this by:
- Treating confirmed fraud claims as **reliable positives**
- Treating all other claims as **unlabeled** (not negative)
- Using a two-step approach to iteratively identify likely negatives and train the final classifier

---

## 📁 Repository Structure

```
├── DME_TrainingData.ipynb     # Feature engineering pipeline
├── DME_PULearning.ipynb       # PU Learning model training
├── data/
│   ├── features_claim.csv
│   ├── features_bene.csv
│   ├── features_billnpi.csv
│   ├── features_ordenpi.csv
│   ├── features_billnpibene.csv
│   ├── features_ordenpibene.csv
│   └── raw_training_data.csv  # Final merged training dataset
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Usage |
|---|---|
| **Python** (pandas, numpy) | Feature engineering & data processing |
| **SQL Server + SQLAlchemy** | Data extraction and staging |
| **T-SQL** | Complex lookback window aggregations |
| **PU Learning** | Fraud model training under label scarcity |

---

## 💡 Why PU Learning?

| Approach | Problem |
|---|---|
| Standard Binary Classifier | Treats unlabeled claims as clean — introduces massive label noise |
| Anomaly Detection | Ignores known fraud patterns entirely |
| **PU Learning** ✅ | Leverages confirmed fraud labels without assuming unlabeled = legitimate |

In real-world healthcare fraud detection, **confirmed fraud cases represent only a fraction of actual fraud**. PU Learning is specifically designed for this setting, making it a more principled and realistic approach than standard supervised methods.

---

## ⚙️ Setup

### Prerequisites
- Python 3.12+
- Access to SQL Server (`_Internal_Reporting` database)
- `pandas`, `sqlalchemy`, `pyodbc`

### Credentials
Create a `Credentials.txt` file with one value per line:
```
server_name
database_name
username
password
```

---

## 📌 Notes
- Claims data starts from **2019** to support 2-year lookback windows for 2021+ flagged claims
- All lookback features use `clm_thru_dt` as the anchor date to prevent data leakage
- Beneficiary population is restricted to **assigned members** (2021–2025)
- Credentials are loaded from an external file and never hardcode
