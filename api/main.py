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
    country: Optional[str] = Field("KE", description="Country: KE or NG")
    currency: Optional[str] = Field("KES", description="Currency: KES or NGN")
    channel: Optional[str] = Field(None, description="mobile_money, p2p, bank_transfer, card, airtime, bill_pay")
    card_type: Optional[str] = Field(None, description="debit, credit, prepaid")
    card_bank: Optional[str] = Field(None, description="Anonymised card issuer")
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card5: Optional[float] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    recipient_account_age_days: Optional[int] = None
    sender_prev_txn_count: Optional[int] = None
    C1: Optional[int] = None; C2: Optional[int] = None; C3: Optional[int] = None; C4: Optional[int] = None
    C5: Optional[int] = None; C6: Optional[int] = None; C7: Optional[int] = None; C8: Optional[int] = None
    D1: Optional[float] = None; D2: Optional[float] = None; D3: Optional[float] = None
    D4: Optional[float] = None; D5: Optional[float] = None
    M1: Optional[str] = None; M2: Optional[str] = None; M3: Optional[str] = None
    M4: Optional[str] = None; M5: Optional[str] = None; M6: Optional[str] = None
    V1: Optional[float] = None;  V2: Optional[float] = None;  V3: Optional[float] = None
    V4: Optional[float] = None;  V5: Optional[float] = None;  V6: Optional[float] = None
    V7: Optional[float] = None;  V8: Optional[float] = None;  V9: Optional[float] = None
    V10: Optional[float] = None; V11: Optional[float] = None; V12: Optional[float] = None
    V13: Optional[float] = None; V14: Optional[float] = None; V15: Optional[float] = None
    V16: Optional[float] = None; V17: Optional[float] = None; V18: Optional[float] = None
    V19: Optional[float] = None; V20: Optional[float] = None


class PredictionOutput(BaseModel):
    TransactionID: int
    isFraud_prob: float = Field(..., ge=0, le=1)
    alarm: bool
    risk_level: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime: float


class ModelInfoResponse(BaseModel):
    model_type: str
    cv_pr_auc: float
    cv_roc_auc: float
    features: int
    threshold: float
    calibration: str


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0, le=1)


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Umba Fraud Detection API",
    description="""
## Real-time Transaction Fraud Scoring Service

- **Algorithm**: Random Forest with isotonic calibration
- **PR-AUC**: 0.162 | **ROC-AUC**: 0.789
- **Features**: 97 engineered features
- `flagged_for_review` excluded to prevent data leakage
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
TRAIN_COLS = None
ALARM_THRESHOLD = 0.5
START_TIME = time.time()


def _extract_feature_names(model) -> list:
    """
    Pull the feature column list directly from the trained model object.
    This is the single source of truth — never try to reconstruct it from
    a preprocessing run.

    Handles three common wrappers:
      1. Plain sklearn estimator with feature_names_in_
      2. CalibratedClassifierCV (wraps estimator in calibrated_classifiers_)
      3. Pipeline where the final step is the estimator
    """
    # Plain estimator
    if hasattr(model, "feature_names_in_"):
        return model.feature_names_in_.tolist()

    # CalibratedClassifierCV
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_names_in_"):
            return base.feature_names_in_.tolist()
        # Some sklearn versions store it as base_estimator
        if hasattr(model, "estimator") and hasattr(model.estimator, "feature_names_in_"):
            return model.estimator.feature_names_in_.tolist()

    # sklearn Pipeline
    if hasattr(model, "steps"):
        final_step = model.steps[-1][1]
        if hasattr(final_step, "feature_names_in_"):
            return final_step.feature_names_in_.tolist()

    raise RuntimeError(
        "Cannot extract feature names from model. "
        "Ensure the model was trained with sklearn >= 1.0 "
        "so feature_names_in_ is set, or save feature names separately."
    )


@app.on_event("startup")
async def startup():
    """Load model and derive feature column list from the model itself."""
    global MODEL, TRAIN_DATA, TRAIN_COLS

    model_dir = Path(__file__).parent.parent / "model"
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )

    MODEL = joblib.load(model_dir / "fraud_model.pkl")

    # SOURCE OF TRUTH: feature names live in the model, not the CSV
    TRAIN_COLS = _extract_feature_names(MODEL)

    # Load raw training data so _prepare() can fill in missing raw columns
    train, _, _ = load_data(data_dir)
    TRAIN_DATA = train

    print(f"Model loaded. Engineered features: {len(TRAIN_COLS)}. Ready.")


def _prepare(tx_dict: dict) -> pd.DataFrame:
    """
    Convert a raw API payload into the 97 engineered features the model needs.

    Rules:
    - Keep TransactionID in the dict — preprocess() needs it for the identity
      merge and drops it internally. Never pop it before calling preprocess().
    - Fill any raw columns the caller omitted with NaN.
    - Concat with a small training slice so freq-encoders have context.
    - Use .reindex() (not []) to align to TRAIN_COLS — missing cols become 0,
      extra cols are silently dropped. No KeyError possible.
    """
    df = pd.DataFrame([tx_dict])

    # Fill missing raw columns (skip label/leakage columns)
    skip = {"isFraud", "flagged_for_review"}
    for col in TRAIN_DATA.columns:
        if col not in df.columns and col not in skip:
            df[col] = np.nan

    # Prepend training slice so categorical/freq encoders have context
    combined = pd.concat([TRAIN_DATA.head(100), df], ignore_index=True)

    # preprocess() treats arg1 as train, arg2 as test
    # Our transaction is the last row of arg1 (X_train)
    X, _, _ = preprocess(
        combined,
        pd.DataFrame(columns=TRAIN_DATA.columns),
        pd.DataFrame(),
    )

    # Grab our row and align to model's expected columns
    # reindex fills any gap with 0 instead of raising KeyError
    return X.iloc[[-1]].reindex(columns=TRAIN_COLS, fill_value=0)


def _risk_level(p: float) -> str:
    if p >= 0.7:   return "critical"
    elif p >= 0.5: return "high"
    elif p >= 0.2: return "medium"
    return "low"


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=dict, tags=["General"])
async def root():
    return {"service": "Umba Fraud Detection API", "version": "1.0.0",
            "docs": "/docs", "status": "running"}


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        uptime=time.time() - START_TIME,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    if MODEL is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    return ModelInfoResponse(
        model_type="Random Forest", cv_pr_auc=0.162, cv_roc_auc=0.789,
        features=len(TRAIN_COLS) if TRAIN_COLS else 97,
        threshold=ALARM_THRESHOLD, calibration="Isotonic (5-fold CV)",
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
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
        # Re-raise with full message so you can see the real error
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))


@app.get("/threshold", tags=["Configuration"])
async def get_threshold():
    return {"threshold": ALARM_THRESHOLD}


@app.put("/threshold", tags=["Configuration"])
async def set_threshold(body: ThresholdUpdate):
    global ALARM_THRESHOLD
    ALARM_THRESHOLD = body.threshold
    return {"threshold": ALARM_THRESHOLD, "message": "Threshold updated"}