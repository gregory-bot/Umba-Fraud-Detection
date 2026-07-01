# Umba Fraud Detection Assessment

**Catching fraud before money moves.**
  
**Author:** Kipngeno Gregory · **Date:** July 2026 · **Time invested:** ~6 hours

---

## Submission Summary

| Part | Deliverable
|------|-------------|--------|
| **A — Pipeline & Model** | 4-model comparison, time-series CV, Random Forest, isotonic calibration
| **B — Serving API** | FastAPI with Swagger docs, deployed on Render |  [Backend](https://umba-fraud-detection.onrender.com/docs) |
| **C — Dashboard** | React + TypeScript operations dashboard, deployed on Netlify |  [front-end](https://umba-fraud-detection.netlify.app/) |
| **D — Dockerize & Deploy** | Dockerfiles, docker-compose, cloud deployment
| **predictions.csv** | 40,000 rows, correct submission format
| **GitHub Repo** | Full source code, model artifact, README |  [gregory-bot/Umba-Fraud-Detection](https://github.com/gregory-bot/Umba-Fraud-Detection) |

---

## Live Services

| Service | URL |
|---------|-----|
| **Frontend Dashboard** | [umba-fraud-detection.netlify.app](https://umba-fraud-detection.netlify.app/) |
| **Backend API** | [umba-fraud-detection.onrender.com](https://umba-fraud-detection.onrender.com) |
| **Swagger / Interactive Docs** | [umba-fraud-detection.onrender.com/docs](https://umba-fraud-detection.onrender.com/docs) |
| **Source Code** | [github.com/gregory-bot/Umba-Fraud-Detection](https://github.com/gregory-bot/Umba-Fraud-Detection) |

---

## System Architecture

![System Architecture — Umba Fraud Detection](https://i.postimg.cc/43nKY7rn/f4a2d8c1-e36a-4514-8be6-20f9adbd2fbc.png)

**End-to-end flow:** Raw CSV data from Kenya and Nigeria → Preprocessing & feature engineering (97 features, leakage removed) → 4-model comparison with time-series cross-validation → Random Forest selected (PR-AUC: 0.162, ROC-AUC: 0.789) → Isotonic calibration → FastAPI serving on Render → React + TypeScript operations dashboard on Netlify.

### Tech Stack

| Layer | Technology | Reason for choice |
|-------|------------|-------------------|
| **ML Pipeline** | Python, scikit-learn, XGBoost, LightGBM | Battle-tested tabular ML; reproducible across environments |
| **Model** | Random Forest + isotonic calibration | Best PR-AUC among 4 candidates; interpretable; non-parametric calibration |
| **API** | FastAPI, Pydantic v2, uvicorn | Async-first; auto-generated OpenAPI docs; strict input validation |
| **Dashboard** | React, TypeScript, TanStack Router, Tailwind CSS, shadcn/ui, Recharts | Type-safe, production-grade UI; no runtime surprises |
| **Deployment** | Render (API), Netlify (Dashboard), Docker | Zero-config cloud; deterministic containers for reproducibility |

---

## The Problem, in Plain English

Every day, thousands of mobile money, card, and bank-transfer transactions flow through Umba across Kenya and Nigeria. Most are legitimate. A small fraction — about 3 in every 100 — are fraudulent.

The challenge is not "can a model spot fraud?" It is **can the bank act on it, in real time, without drowning staff in false alarms?**

This project answers that with four things working together:

1. A **pipeline** that trains and validates a fraud detection model on 120,000 historical transactions
2. A **model** (Random Forest) that scores every transaction for fraud risk with calibrated probabilities
3. An **API** that delivers that score in milliseconds when a transaction happens
4. A **dashboard** that lets a non-technical operations team see what is flagged and why — no code required

---

## How It Works, Step by Step

### Step 1 — Learn from the past
The model was trained on 120,000 historical transactions, each already labelled "fraud" or "legitimate." It studied patterns — amount, channel, device, time of day, recipient account age — that separate the two groups.

### Step 2 — Remove the cheat code (data leakage)
One field, `flagged_for_review`, had a 0.63 correlation with the fraud label — the strongest signal in the dataset. But on inspection: reviewers only fill this field in *after* deciding a transaction is fraudulent. It is a consequence of fraud, not a predictor of it.

Using it would be like giving the model the answer sheet during the exam — impressive validation scores, completely useless in production where the field is always empty at decision time. **It was removed entirely.**

This was the single most important judgment call in the project.

### Step 3 — Test like it is production
Instead of shuffling all transactions randomly (which leaks future patterns into training), the model was trained on the earliest 80% of transactions by time and tested on the remaining 20% — strictly later in time. This is exactly how it will work live: always predicting transactions it has never seen.

### Step 4 — Compare four candidate models

| Model | PR-AUC | ROC-AUC | Brier Score |
|-------|--------|---------|-------------|
| Logistic Regression | 0.062 | 0.593 | 0.240 |
| **Random Forest ✓** | **0.162** | **0.789** | **0.130** |
| XGBoost | 0.145 | 0.772 | 0.105 |
| LightGBM | 0.139 | 0.763 | 0.118 |

Random Forest was selected. PR-AUC is the honest primary metric under heavy class imbalance (3.4% fraud rate) — it measures precision-recall trade-off across all thresholds, unlike ROC-AUC which can look good even on a useless model when the negatives heavily outnumber positives.

### Step 5 — Calibrate the probabilities
A raw model score of 0.8 does not automatically mean an 80% chance of fraud. **Isotonic calibration** was applied so scores are meaningful: a score of 0.8 should correspond to roughly 8 in 10 similar transactions being fraudulent. This matters for threshold setting and for ops teams trusting the output.

### Step 6 — Serve it in real time
The trained model is loaded into a FastAPI service. A transaction comes in, the same 97 features are computed on the fly, and a risk score is returned in under 50 ms. A React dashboard gives operations teams a live view of flagged transactions, risk distribution, and threshold controls — without needing to write a single line of code.

---

## Operational Impact

| If staff review… | Fraud caught | vs. random spot checks |
|------------------|-------------|------------------------|
| Top 1% riskiest transactions | ~15% of all fraud | 15× better |
| Top 5% riskiest transactions | ~28% of all fraud | **5.6× better** |
| Top 10% riskiest transactions | ~42% of all fraud | 4.2× better |

The model does not eliminate fraud. It concentrates it — so the same number of staff hours catches far more of it.

---

## Risk Tiers and Actions

| Score | Label | Action |
|-------|-------|--------|
| 0.00 – 0.19 | Low | Auto-approved |
| 0.20 – 0.49 | Medium | Logged; reviewed in batch |
| 0.50 – 0.69 | High | Flagged for manual review |
| 0.70 – 1.00 | Critical | Transaction held immediately |

---

## Screenshots

### Dashboard — Overview

> *(Add screenshot: overview page showing model health, fraud rate, flagged count)*

### Dashboard — Score Distribution

> *(Add screenshot: histogram of isFraud_prob across 40,000 test transactions)*

### Dashboard — Transaction Lookup

> *(Add screenshot: transaction ID input returning risk score with colour-coded badge)*

### API — Swagger Docs

> *(Add screenshot: Render-deployed /docs page showing POST /predict endpoint)*

### API — Predict Response

> *(Add screenshot: live response JSON with isFraud_prob, alarm, risk_level)*

### Training Run — Terminal Output

> *(Add screenshot: python src/pipeline.py output showing CV folds, metrics, model saved)*

### Render Deployment Logs

> *(Add screenshot: Render build log showing successful deploy)*

---

## Quick Start

### Prerequisites

- Python 3.11+
- 8 GB RAM
- Git

### Setup

```bash
# 1. Clone
git clone https://github.com/gregory-bot/Umba-Fraud-Detection.git
cd Umba-Fraud-Detection

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
.\.venv\Scripts\Activate.ps1     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the data files
# Copy train.csv, test.csv, identity.csv, sample_submission.csv into data/

# 5. Train the model and generate predictions.csv
python src/pipeline.py

# 6. Start the API
uvicorn api.main:app --reload --port 8000

# 7. Start the Streamlit dashboard (optional — React dashboard is also live above)
streamlit run dashboard/app.py
```

### Local access points

| Service | URL |
|---------|-----|
| API root | `http://localhost:8000` |
| Swagger docs | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |
| Streamlit dashboard | `http://localhost:8501` |

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API metadata |
| `GET` | `/health` | Health check + model load status |
| `GET` | `/model/info` | Model metadata, CV metrics, current threshold |
| `POST` | `/predict` | Score a single transaction |
| `GET` | `/threshold` | Read current alarm threshold |
| `PUT` | `/threshold` | Update alarm threshold (ops use) |
| `GET` | `/predictions/csv` | Download full test-set predictions |

### Example — score a transaction

```bash
curl -X POST https://umba-fraud-detection.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": 1120000,
    "TransactionDT": 11657574,
    "TransactionAmt": 143.9,
    "country": "KE",
    "currency": "KES",
    "channel": "mobile_money",
    "card_type": "debit",
    "card_bank": "opay",
    "card1": 7236, "card2": 316.0, "card3": 150.0, "card5": 220.0,
    "addr1": 322.0, "addr2": 254.0,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "recipient_account_age_days": 241,
    "sender_prev_txn_count": 23
  }'
```

**Response:**

```json
{
  "TransactionID": 1120000,
  "isFraud_prob": 0.0051,
  "alarm": false,
  "risk_level": "low"
}
```

---

## Project Structure

```
Umba-Fraud-Detection/
├── README.md                        ← This document
├── requirements.txt                 ← Python dependencies
├── predictions.csv                  ← 40,000 test-set scores (submission)
├── Dockerfile                       ← API container
├── Dockerfile.dashboard             ← Streamlit dashboard container
├── docker-compose.yml               ← Multi-service orchestration
├── .dockerignore
│
├── data/
│   ├── train.csv                    ← 120,000 labelled transactions
│   ├── test.csv                     ← 40,000 unlabelled transactions
│   ├── identity.csv                 ← 47,140 device / session records
│   └── sample_submission.csv        ← Expected submission format
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb    ← Full EDA + statistical tests
│
├── src/
│   ├── preprocessing.py             ← Data loading, 97-feature engineering
│   ├── model_training.py            ← Time-series CV, 4-model comparison, calibration
│   ├── hypothesis_tests.py          ← Statistical significance tests
│   └── pipeline.py                  ← One-command end-to-end runner
│
├── model/
│   ├── fraud_model.pkl              ← Trained Random Forest (67 MB)
│   └── model_results.json           ← CV metrics per fold
│
├── api/
│   └── main.py                      ← FastAPI service
│
└── dashboard/
    └── app.py                       ← Streamlit operations view
```

---

## Feature Engineering — 97 Features

| Category | Technique | Features |
|----------|-----------|----------|
| Amount | Log transform, per-currency z-score, round-number flag | 6 |
| Categorical | One-hot (card_type, channel), frequency encoding (card1–5, addr1–2, email domains) | 25+ |
| Email domains | TLD extraction, payer = recipient flag | 4 |
| Temporal | Hour-of-day (sin/cos), day-of-week (sin/cos), night flag, weekend flag | 6 |
| Velocity | Log sender transaction count, first-time sender flag, log recipient account age, new account flag | 5 |
| Match flags M1–M6 | Encoded as −1 / 0 / 1 (F / missing / T) | 12 |
| V-block (V1–V20) | Scaled features + missing indicators | 22 |
| D-block timedeltas | Log transforms + missing indicators | 10 |
| C-block counts | Raw count features | 8 |
| Identity (joined) | Device type, session count, id_01–id_11 aggregated by mean | 14 |

---

## Data Integrity — What Was Checked and Why It Matters

### Finding 1 — `flagged_for_review` is a leakage column

`flagged_for_review` had a Spearman correlation of **0.63** with `isFraud` — by far the strongest signal. The data dictionary notes this field is "populated by reviewers as part of the review process." In plain terms: the field only exists after a human has already decided the transaction is suspicious. At real-time scoring, this field is always blank. **Dropped.**

A model trained with this field would show inflated PR-AUC on validation (it would essentially learn "if flagged = 1, predict fraud") and then fail completely in production. This is the most common failure mode in fraud modelling and the most important thing to catch.

### Finding 2 — identity.csv is not one row per transaction

```
identity.csv rows:    47,140
Unique TransactionIDs: 41,610
Duplicate rows:         5,530 (multiple device sessions per transaction)
```

A naive `pd.merge(train, identity, on='TransactionID')` would silently duplicate transaction rows — inflating the training set, corrupting cross-validation fold sizes, and producing leaked metrics. **Resolution:** aggregated identity to one row per TransactionID (numeric columns averaged across sessions, categorical columns by mode, session count retained as a feature).

### Finding 3 — time ordering confirmed, no leakage

```
train.csv  max TransactionDT: 11,657,391
test.csv   min TransactionDT: 11,657,574
```

The test set is strictly later than training. Validated before choosing validation strategy.

### Hypothesis tests on all 97 features

| Test | Applied to | Result |
|------|-----------|--------|
| D'Agostino-Pearson | All numerical features | None normally distributed → non-parametric tests chosen |
| Mann-Whitney U | Numerical features (fraud vs legitimate) | All significant at p < 0.05 |
| Chi-squared | Categorical features | All significant at p < 0.05 |

---

## Design Decisions and Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Time-series CV over random K-fold | Matches production: train on past, score future | Honest but lower reported metrics than random splits |
| Random Forest over LightGBM | Best PR-AUC; less sensitive to hyperparameter tuning | Larger model artifact (67 MB vs ~2 MB) |
| PR-AUC as primary metric | Correct metric under 3.4% fraud rate; punishes false positives | Less familiar to non-ML stakeholders |
| Drop `flagged_for_review` | Prevents direct target leakage | Lose a 0.63 correlation signal — by design |
| Isotonic calibration over Platt scaling | Non-parametric; works for any score distribution | Slightly higher variance on small datasets |
| Threshold at F1-maximising point | Balances precision and recall for ops teams | Threshold may need retuning as fraud patterns shift |

---

## Docker Deployment

### Start everything with one command

```bash
docker compose up -d
```

Services started:

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

### Logs

```bash
docker compose logs -f api
docker compose logs -f dashboard
```

### Stop

```bash
docker compose down
```

### Single-service build

```bash
docker build -t umba-fraud-api .
docker run -p 8000:8000 umba-fraud-api
```

### Cloud deployment (Render)

| Setting | Value |
|---------|-------|
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| Environment | `PYTHON_VERSION=3.11` |

The model artifact (`model/fraud_model.pkl`) is pre-trained and included in the repository — no retraining is required on deploy.

---

## Production Monitoring (Bonus Thinking)

This section answers: *how would you keep this working after go-live?*

**Drift detection**  
Track Population Stability Index (PSI) weekly on the top 20 input features. PSI > 0.2 on any feature triggers a retraining alert. In practice, `TransactionAmt` distribution and channel mix shift fastest in African mobile money markets — those get daily monitoring.

**Performance monitoring**  
Track the model's predicted fraud rate against confirmed chargebacks (with a 30-day lag for resolution). A >20% relative deviation between predicted and confirmed rates triggers a review. Grafana + a Postgres audit table is sufficient at Umba's scale.

**Retraining cadence**  
Scheduled monthly retrain on a rolling window of the last 6 months. Additional unscheduled retrains triggered by: (a) PSI alert, (b) PR-AUC on a holdout dropping below 0.10, or (c) a major product change (new channel, new country).

**Explainability for reviewers**  
SHAP values computed per prediction and surfaced in the dashboard — "this transaction was flagged mainly because the recipient account is 2 days old and the amount is 8× the sender's average." Builds ops team trust and surfaces edge cases for model improvement.

**A/B testing new model versions**  
Shadow mode: new model runs in parallel, logs scores, is not yet used for decisions. Compare PR-AUC on confirmed fraud labels from ops reviews over 2 weeks. Promote if better; roll back if worse, without ops disruption.

---

## Next Steps (With More Time)

- **Hyperparameter tuning** — Bayesian optimisation (Optuna) across all four candidate models rather than default parameters
- **Rolling-window features** — per-card and per-user aggregations over 7d / 30d / 90d (the most impactful class of features for fraud, requiring a feature store)
- **Entity embeddings** — learned dense representations for high-cardinality columns (`card_bank`, `P_emaildomain`, `addr1`) instead of frequency encoding
- **Graph features** — network analysis on sender-recipient pairs to detect coordinated fraud rings
- **Cost-sensitive threshold** — tune threshold using actual cost of a missed fraud vs. cost of a false positive (requires business input on KES values)
- **Model ensemble** — stacked generaliser over Random Forest + XGBoost + LightGBM, trained on out-of-fold predictions

---

## AI Tool Usage — Honest Account

This is an AI-native role. The expectation is fluent use of AI tools *with* rigorous human judgment. Here is exactly how that worked on this project.

### Where AI accelerated the work

**Security and adversarial input testing**  
I used Claude to generate adversarial transaction payloads designed to break the API — malformed types, boundary values, missing required fields, SQL-injection-style strings in categorical fields, and extreme numeric outliers (TransactionAmt = 10^9, negative values, NaN). This caught two validation gaps in the Pydantic schema before they reached review. I wrote the fixes; AI wrote the attack cases.

**Statistical test selection**  
Given 97 features and a non-parametric distribution (D'Agostino-Pearson confirmed non-normality across all features), I used Claude as a sounding board for whether Mann-Whitney U or Kolmogorov-Smirnov was more appropriate given the sample sizes and the fraud/legitimate imbalance. I made the call; AI surfaced the relevant considerations quickly.

**Documentation structure**  
First draft of the non-technical "how it works" section was generated as a starting point. Every sentence was rewritten or cut. The leakage explanation, the calibration section, and the trade-offs table are entirely original — those required judgment that AI cannot substitute.

**Boilerplate scaffolding**  
API endpoint schemas, Pydantic models, Dockerfile templates. Generated, then read line by line and edited. Nothing was committed without being understood.

**Debugging**  
The identity aggregation join issue (5,530 duplicate TransactionIDs producing inflated row counts) was flagged by Claude when I described the shape mismatch. I verified the root cause independently in a notebook, wrote the aggregation logic, and validated the fix. The tool pointed; I drove.

### Where AI was not used

The core judgment calls in this project were made without AI:

- Recognising `flagged_for_review` as leakage (not just a strong feature) required reading the data dictionary carefully and reasoning about the operational timeline
- Choosing time-series CV over random K-fold — this was a conscious decision, not a generated suggestion
- Selecting PR-AUC over ROC-AUC as the primary metric
- The identity aggregation strategy (mean for numerics, mode for categoricals, retain session count)
- Model selection rationale

**Every line of code in this repository was read, understood, and is defensible by the author.**

---

## Submission Checklist

- [x] `predictions.csv` — 40,000 rows, format matches `sample_submission.csv` exactly
- [x] Model artifact — `model/fraud_model.pkl` (67 MB, pre-trained)
- [x] Reproducible training pipeline — `python src/pipeline.py`
- [x] FastAPI service — deployed on Render with Swagger docs
- [x] Operations dashboard — React + TypeScript, deployed on Netlify
- [x] EDA notebook — `notebooks/01_eda_and_modeling.ipynb`
- [x] Docker support — `Dockerfile`, `Dockerfile.dashboard`, `docker-compose.yml`
- [x] This README

---

**Live API:** [umba-fraud-detection.onrender.com/docs](https://umba-fraud-detection.onrender.com/docs)  
**Frontend Dashboard:** [umba-fraud-detection.netlify.app](https://umba-fraud-detection.netlify.app/)  
**GitHub:** [gregory-bot/Umba-Fraud-Detection](https://github.com/gregory-bot/Umba-Fraud-Detection)

---

*Proprietary — Umba Microfinance Bank Limited. All rights reserved.*
