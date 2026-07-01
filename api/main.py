"""
Umba Fraud Detection API - Real-time Transaction Scoring Service.

Endpoints:
- GET  /              - API information
- GET  /health        - Health check with model status
- GET  /model/info    - Model metadata and performance metrics
- POST /predict       - Score a single transaction
- GET  /threshold     - Get current alarm threshold
- PUT  /threshold     - Update alarm threshold
"""
import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing import load_data, preprocess

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Pydantic Schemas
# ============================================================================

class TransactionInput(BaseModel):
    """Single transaction for fraud scoring.

    Only TransactionID, TransactionDT, and TransactionAmt are required.
    All other fields default to None/NaN matching training behavior.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "P_emaildomain": "gmail.com", "R_emaildomain": "gmail.com",
                "recipient_account_age_days": 241,
                "sender_prev_txn_count": 23
            }
        }
    )

    TransactionID: int = Field(..., description="Unique transaction identifier")
    TransactionDT: int = Field(..., description="Transaction timestamp (seconds offset from reference)")
    TransactionAmt: float = Field(..., description="Transaction amount in local currency")
    country: Optional[str] = Field("KE", description="Country: KE (Kenya) or NG (Nigeria)")
    currency: Optional[str] = Field("KES", description="Currency: KES or NGN")
    channel: Optional[str] = Field(None, description="Channel: mobile_money, p2p, bank_transfer, card, airtime, bill_pay")
    card_type: Optional[str] = Field(None, description="Card type: debit, credit, prepaid")
    card_bank: Optional[str] = Field(None, description="Anonymised card issuer / wallet provider")
    card1: Optional[float] = Field(None, description="Anonymised card attribute")
    card2: Optional[float] = Field(None, description="Anonymised card attribute")
    card3: Optional[float] = Field(None, description="Anonymised card attribute")
    card5: Optional[float] = Field(None, description="Anonymised card attribute")
    addr1: Optional[float] = Field(None, description="Anonymised address code")
    addr2: Optional[float] = Field(None, description="Anonymised address code")
    dist1: Optional[float] = Field(None, description="Distance measure")
    dist2: Optional[float] = Field(None, description="Distance measure")
    P_emaildomain: Optional[str] = Field(None, description="Payer email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    recipient_account_age_days: Optional[int] = Field(None, description="Age of recipient account in days")
    sender_prev_txn_count: Optional[int] = Field(None, description="Number of prior sender transactions")
    C1: Optional[int] = None; C2: Optional[int] = None; C3: Optional[int] = None; C4: Optional[int] = None
    C5: Optional[int] = None; C6: Optional[int] = None; C7: Optional[int] = None; C8: Optional[int] = None
    D1: Optional[float] = None; D2: Optional[float] = None; D3: Optional[float] = None
    D4: Optional[float] = None; D5: Optional[float] = None
    M1: Optional[str] = None; M2: Optional[str] = None; M3: Optional[str] = None
    M4: Optional[str] = None; M5: Optional[str] = None; M6: Optional[str] = None
    V1: Optional[float] = None; V2: Optional[float] = None; V3: Optional[float] = None
    V4: Optional[float] = None; V5: Optional[float] = None; V6: Optional[float] = None
    V7: Optional[float] = None; V8: Optional[float] = None; V9: Optional[float] = None
    V10: Optional[float] = None; V11: Optional[float] = None; V12: Optional[float] = None
    V13: Optional[float] = None; V14: Optional[float] = None; V15: Optional[float] = None
    V16: Optional[float] = None; V17: Optional[float] = None; V18: Optional[float] = None
    V19: Optional[float] = None; V20: Optional[float] = None


class PredictionOutput(BaseModel):
    """Prediction result for a single transaction."""
    TransactionID: int = Field(..., description="Transaction identifier")
    isFraud_prob: float = Field(..., description="Fraud probability (0-1)", ge=0, le=1)
    alarm: bool = Field(..., description="Whether alarm threshold was exceeded")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("ok", description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime: float = Field(..., description="Service uptime in seconds")


class ModelInfoResponse(BaseModel):
    """Model metadata and performance."""
    model_type: str = Field(..., description="Algorithm type")
    cv_pr_auc: float = Field(..., description="Cross-validation PR-AUC")
    cv_roc_auc: float = Field(..., description="Cross-validation ROC-AUC")
    features: int = Field(..., description="Number of engineered features")
    threshold: float = Field(..., description="Current alarm threshold")
    calibration: str = Field(..., description="Calibration method")


class ThresholdUpdate(BaseModel):
    """Threshold update request."""
    threshold: float = Field(..., description="New alarm threshold (0-1)", ge=0, le=1)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Umba Fraud Detection API",
    description="""
## Real-time Transaction Fraud Scoring Service

This API serves a trained fraud detection model for scoring financial
transactions in real-time. The model was trained on anonymised transaction
data from Kenya and Nigeria.

### Model Details
- **Algorithm**: Random Forest with isotonic calibration
- **Validation**: 5-fold time-series cross-validation
- **PR-AUC**: 0.162 (honest metric for imbalanced data ~3.4% fraud)
- **ROC-AUC**: 0.789
- **Features**: 97 engineered features

### Data Integrity Notes
- `flagged_for_review` intentionally excluded (post-hoc label, data leakage)
- Time-based CV ensures no future information leaks into training
- Identity data aggregated to one row per transaction
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "Gregory", "email": "kipngenogregory@gmail.com"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State
# ============================================================================

MODEL = None
TRAIN_DATA = None
TRAIN_COLS = None          # 97 engineered feature columns — set during startup
ALARM_THRESHOLD = 0.5
START_TIME = time.time()


@app.on_event("startup")
async def startup():
    """Load model and derive the 97 engineered feature columns from a small
    preprocessing run so _prepare() can align columns correctly at predict time.
    """
    global MODEL, TRAIN_DATA, TRAIN_COLS

    model_dir = Path(__file__).parent.parent / "model"
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )

    MODEL = joblib.load(model_dir / "fraud_model.pkl")

    # Load raw data — needed so _prepare() can fill missing raw columns
    train, test, _ = load_data(data_dir)
    TRAIN_DATA = train

    # Run preprocess on a small sample purely to discover the 97 engineered
    # column names. We use the *test* output (second return value) because
    # the test slice is the "unseen row" path — exactly what _prepare() does.
    identity_path = os.path.join(data_dir, "identity.csv")
    identity_sample = (
        pd.read_csv(identity_path).head(100)
        if os.path.exists(identity_path)
        else pd.DataFrame()
    )
    _, X_sample, _ = preprocess(
        train.head(500), test.head(100), identity_sample
    )
    TRAIN_COLS = X_sample.columns.tolist()

    print(f"Model loaded. Engineered features: {len(TRAIN_COLS)}.")


def _prepare(tx_dict: dict) -> pd.DataFrame:
    """Transform a single raw transaction dict into the 97 engineered features
    the model expects.

    Key rules:
    - DO NOT pop TransactionID before calling preprocess() — preprocess() needs
      it for the identity merge and drops it internally.
    - Fill any raw columns the caller omitted with NaN so preprocess() can run.
    - Concatenate with a small slice of training data so frequency-encoders and
      categorical encoders have context (they were fit on training distribution).
    - Return only the 97 columns in the exact order used at training time.
    """
    df = pd.DataFrame([tx_dict])

    # Fill in any raw columns the caller didn't supply (except label columns)
    skip = {"isFraud", "flagged_for_review"}
    for col in TRAIN_DATA.columns:
        if col not in df.columns and col not in skip:
            df[col] = np.nan

    # Prepend a small training reference slice so encoders have context
    combined = pd.concat([TRAIN_DATA.head(100), df], ignore_index=True)

    # preprocess() signature: (train_df, test_df, identity_df)
    # We pass the whole combined df as "train" and an empty df as "test"
    # so the new row is the last row of the returned X_train
    X, _, _ = preprocess(
        combined,
        pd.DataFrame(columns=TRAIN_DATA.columns),
        pd.DataFrame(),
    )

    # Take only the last row (our transaction) and align to training columns
    return X.iloc[[-1]][TRAIN_COLS]


def _risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "critical"
    elif probability >= 0.5:
        return "high"
    elif probability >= 0.2:
        return "medium"
    return "low"


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=dict, summary="API Root", tags=["General"])
async def root():
    return {
        "service": "Umba Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["General"],
)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        uptime=time.time() - START_TIME,
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    tags=["Model"],
)
async def model_info():
    if MODEL is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    return ModelInfoResponse(
        model_type="Random Forest",
        cv_pr_auc=0.162,
        cv_roc_auc=0.789,
        features=len(TRAIN_COLS) if TRAIN_COLS else 97,
        threshold=ALARM_THRESHOLD,
        calibration="Isotonic (5-fold CV)",
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Predict Fraud Probability",
    tags=["Prediction"],
)
async def predict(tx: TransactionInput):
    if MODEL is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    try:
        X = _prepare(tx.model_dump())
        p = float(MODEL.predict_proba(X)[0, 1])
        return PredictionOutput(
            TransactionID=tx.TransactionID,
            isFraud_prob=round(p, 4),
            alarm=p >= ALARM_THRESHOLD,
            risk_level=_risk_level(p),
        )
    except Exception as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))


@app.get("/threshold", summary="Get Alarm Threshold", tags=["Configuration"])
async def get_threshold():
    return {"threshold": ALARM_THRESHOLD}


@app.put("/threshold", summary="Update Alarm Threshold", tags=["Configuration"])
async def set_threshold(body: ThresholdUpdate):
    global ALARM_THRESHOLD
    ALARM_THRESHOLD = body.threshold
    return {"threshold": ALARM_THRESHOLD, "message": "Threshold updated"}