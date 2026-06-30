from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = FastAPI(title="Umba Fraud Detection API", version="1.0.0")

model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'fraud_model.pkl')
model = joblib.load(model_path)

features = None


class TransactionInput(BaseModel):
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float
    country: str
    currency: str
    channel: str
    card_type: str
    card_bank: str
    card1: int
    card2: float = None
    card3: float = None
    card5: float = None
    addr1: float = None
    addr2: float = None
    dist1: float = None
    dist2: float = None
    P_emaildomain: str = None
    R_emaildomain: str = None
    recipient_account_age_days: int = None
    sender_prev_txn_count: int = None
    C1: int = None
    C2: int = None
    C3: int = None
    C4: int = None
    C5: int = None
    C6: int = None
    C7: int = None
    C8: int = None
    D1: float = None
    D2: float = None
    D3: float = None
    D4: float = None
    D5: float = None
    M1: str = None
    M2: str = None
    M3: str = None
    M4: str = None
    M5: str = None
    M6: str = None
    V1: float = None
    V2: float = None
    V3: float = None
    V4: float = None
    V5: float = None
    V6: float = None
    V7: float = None
    V8: float = None
    V9: float = None
    V10: float = None
    V11: float = None
    V12: float = None
    V13: float = None
    V14: float = None
    V15: float = None
    V16: float = None
    V17: float = None
    V18: float = None
    V19: float = None
    V20: float = None
    DeviceType: str = None
    DeviceInfo: str = None
    id_01: float = None
    id_02: float = None
    id_03: float = None
    id_04: float = None
    id_05: float = None
    id_06: float = None
    id_07: float = None
    id_08: float = None
    id_09: float = None
    id_10: float = None
    id_11: float = None


class BatchInput(BaseModel):
    transactions: list[TransactionInput]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/")
def root():
    return {"app": "Umba Fraud Detection API", "version": "1.0.0"}


@app.post("/predict")
def predict(txn: TransactionInput):
    try:
        df = pd.DataFrame([txn.model_dump()])
        from preprocessing import create_features
        df_feat, _ = create_features(df, is_train=False)
        expected_cols = getattr(model, 'feature_names_in_', None)
        if expected_cols is not None:
            for c in expected_cols:
                if c not in df_feat.columns:
                    df_feat[c] = 0
            df_feat = df_feat[expected_cols]

        prob = model.predict_proba(df_feat.values)[0, 1]
        return {
            "TransactionID": txn.TransactionID,
            "isFraud_prob": float(prob),
            "alarm": bool(prob >= 0.5),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
def predict_batch(batch: BatchInput):
    try:
        rows = [t.model_dump() for t in batch.transactions]
        df = pd.DataFrame(rows)
        from preprocessing import create_features
        df_feat, _ = create_features(df, is_train=False)
        expected_cols = getattr(model, 'feature_names_in_', None)
        if expected_cols is not None:
            for c in expected_cols:
                if c not in df_feat.columns:
                    df_feat[c] = 0
            df_feat = df_feat[expected_cols]

        probs = model.predict_proba(df_feat.values)[:, 1]
        return [
            {
                "TransactionID": t.TransactionID,
                "isFraud_prob": float(p),
                "alarm": bool(p >= 0.5),
            }
            for t, p in zip(batch.transactions, probs)
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
