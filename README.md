# Umba Fraud Detection — Take-Home Assessment

## Project Structure

```
.
├── README.md                          # This file — approach, metrics, trade-offs
├── requirements.txt                   # Python dependencies
├── predictions.csv                    # Test set predictions (submission format)
├── create_notebook.py                 # Notebook builder script
├── data/
│   ├── train.csv                      # Labelled transactions (120k rows)
│   ├── test.csv                       # Unlabelled transactions (40k rows)
│   ├── identity.csv                   # Device/session feed
│   └── sample_submission.csv          # Expected output format
├── notebooks/
│   └── 01_eda_and_modeling.ipynb      # Full EDA + modeling walkthrough
├── src/
│   ├── preprocessing.py               # Data loading, feature engineering, encoding
│   ├── model_training.py              # CV, model comparison, calibration
│   ├── hypothesis_tests.py            # Statistical tests for feature selection
│   └── pipeline.py                    # End-to-end pipeline runner
├── model/
│   ├── fraud_model.pkl                # Trained & calibrated model artifact
│   └── model_results.json             # CV metrics per model
├── api/
│   └── main.py                        # FastAPI serving (health, /predict, /predict_batch)
└── dashboard/
    └── app.py                         # Streamlit operations dashboard
```

---

## How to Run Everything

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Run the pipeline (train + predict)

```bash
source .venv/bin/activate
python src/pipeline.py
```

This loads data, engineers features, runs 4-model comparison with 5-fold time-series CV, trains the best model (Random Forest), calibrates it, and writes `predictions.csv`.

### 2. Run the Jupyter notebook

```bash
source .venv/bin/activate
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

### 3. Serve the API

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

Endpoints:
- `GET /health` — health check
- `POST /predict` — score a single transaction
- `POST /predict_batch` — score multiple transactions

### 4. Launch the dashboard

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

---

## Approach

### Data Integrity & Leakage

The most critical finding was **`flagged_for_review`**: it has a 0.63 correlation with `isFraud`, but this field is populated *after* manual review — it would not be available at prediction time. I dropped it entirely.

The `identity.csv` join is another pitfall: it has 47k rows for 41.6k unique `TransactionID`s (some have 2 session rows). I aggregated by taking the mode for categorical fields and the mean for numeric fields.

### Handling Class Imbalance

Fraud is 3.44% of the training set. I used:
- **PR-AUC** as the primary metric (reports precision-recall trade-off honestly under imbalance)
- **`class_weight='balanced'`** / `scale_pos_weight` in all models
- Time-series cross-validation to evaluate on future data

### Feature Engineering

- Log-transformed `TransactionAmt` and created deviation from currency-group mean
- One-hot encoded `card_type` and `channel`
- Frequency encoding for high-cardinality columns (`card1`, `card_bank`, `addr1`, `DeviceInfo`, email domains)
- Extracted email domain prefixes
- Binned `recipient_account_age_days`
- Interaction features: amount × channel, amount × sender_txn_count
- Missing-count features for the V block
- Scaled D features (timedeltas)

### Hypothesis Testing for Feature Selection

Before using features, I tested for discriminative power:
- **Numerical features**: Checked normality (D'Agostino-Pearson), then applied Welch's t-test (normal) or Mann-Whitney U (non-normal)
- **Categorical features**: Chi-squared test of independence
- All features showed statistically significant differences between fraud and legitimate transactions (p < 0.05), so all were retained.

### Validation Design

Used **TimeSeriesSplit** (5 folds) rather than random K-fold — the test set occurs strictly later in time than the training set, and random splits would leak future information into training.

### Model Selection

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Logistic Regression | 0.062 | 0.594 | 0.240 |
| Random Forest | **0.162** | **0.789** | 0.130 |
| XGBoost | 0.145 | 0.771 | **0.105** |
| LightGBM | 0.139 | 0.763 | 0.118 |

**Random Forest** was selected as the final model based on best PR-AUC. After training on the full dataset, I applied **isotonic calibration** (5-fold cross-validated) to produce well-calibrated probabilities. The calibrated model achieves a predicted fraud rate of 3.8% on the test set, closely matching the training rate of 3.4%.

### Trade-offs & Next Steps

**What I'd improve with more time:**
1. **Feature engineering**: Rolling-window aggregations per card/user, RFM-style features, graph-based features linking sender-recipient pairs
2. **Hyperparameter tuning**: Grid search or Bayesian optimization for the best model
3. **Ensemble**: Stacking or weighted averaging of the best models
4. **Threshold optimization**: Choose the alarm threshold based on operational cost (false positives vs missed fraud)
5. **Unsupervised features**: Isolation Forest scores as additional signals
6. **Model monitoring**: Population stability index (PSI) for drift detection, scheduled retraining

---

## AI Usage Note

This solution was built with AI assistance (Claude Code). AI was used only in generating the final readme, and debugging:   

I own every line and can explain all decisions made.

---

## Predictions Format

`predictions.csv`:
```csv
TransactionID,isFraud_prob
1120000,0.0131
1120001,0.8742
...
```

## Docker Deployment (Part D - Bonus)

### Quick Start with Docker Compose

\\\ash
# Build and start all services
docker compose up -d

# Check logs
docker compose logs -f api

# Stop
docker compose down
\\\

Services:
- **API**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

### Docker (Single Service)

\\\ash
# Build
docker build -t umba-fraud-api .

# Run
docker run -p 8000:8000 umba-fraud-api
\\\

### Cloud Deployment

**Render:**
- Build Command: pip install -r requirements.txt
- Start Command: uvicorn api.main:app --host 0.0.0.0 --port 
- The model artifact is pre-trained and included in the repo.

**AWS/GCP/Azure:**
- Push Docker image to container registry
- Deploy as container service (ECS, Cloud Run, ACI)
- Mount persistent volume for model retraining artifacts

### Production Monitoring

- **Drift Detection**: Track PSI on feature distributions weekly
- **Retraining**: Trigger when PR-AUC drops below threshold or every 30 days
- **Alerting**: Set up alerts for prediction rate changes >20%
