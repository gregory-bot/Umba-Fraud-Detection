import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import json


def time_based_split(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(X, y)


def evaluate_model(y_true, y_prob, threshold=0.5):
    prauc = average_precision_score(y_true, y_prob)
    rocauc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    return {
        'PR_AUC': prauc,
        'ROC_AUC': rocauc,
        'Brier': brier,
    }


def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]


def train_and_evaluate_cv(X, y, model, model_name, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_metrics = []
    thresholds = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        m = model.__class__(**model.get_params())
        m.fit(X_tr, y_tr)

        y_prob = m.predict_proba(X_val)[:, 1]
        metrics = evaluate_model(y_val, y_prob)
        metrics['fold'] = fold
        cv_metrics.append(metrics)

        thresh = find_optimal_threshold(y_val, y_prob)
        thresholds.append(thresh)

    cv_df = pd.DataFrame(cv_metrics)
    result = {
        'model': model_name,
        'mean_PR_AUC': cv_df['PR_AUC'].mean(),
        'std_PR_AUC': cv_df['PR_AUC'].std(),
        'mean_ROC_AUC': cv_df['ROC_AUC'].mean(),
        'mean_Brier': cv_df['Brier'].mean(),
        'per_fold': cv_metrics,
        'mean_threshold': np.mean(thresholds),
    }
    print(f'\n=== {model_name} ===')
    print(f'PR-AUC: {result["mean_PR_AUC"]:.4f} +/- {result["std_PR_AUC"]:.4f}')
    print(f'ROC-AUC: {result["mean_ROC_AUC"]:.4f}')
    print(f'Brier: {result["mean_Brier"]:.4f}')
    return result


def train_final_model(X, y, X_test, model_class, best_params=None):
    model = model_class(**best_params) if best_params else model_class()
    model.fit(X, y)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, y_prob


def run_model_comparison(X_train, X_test, y):
    results = {}

    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=0.1)
    results['LogisticRegression'] = train_and_evaluate_cv(X_train, y, lr, 'LogisticRegression')

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=50,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1
    )
    results['RandomForest'] = train_and_evaluate_cv(X_train, y, rf, 'RandomForest')

    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
    results['XGBoost'] = train_and_evaluate_cv(X_train, y, xgb_model, 'XGBoost')

    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        class_weight='balanced', subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    results['LightGBM'] = train_and_evaluate_cv(X_train, y, lgb_model, 'LightGBM')

    best_model_name = max(results, key=lambda k: results[k]['mean_PR_AUC'])
    print(f'\n=== BEST MODEL: {best_model_name} (PR-AUC: {results[best_model_name]["mean_PR_AUC"]:.4f}) ===')

    model_map = {
        'LogisticRegression': (LogisticRegression, {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42, 'C': 0.1}),
        'RandomForest': (RandomForestClassifier, {'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 50,
                                                   'class_weight': 'balanced_subsample', 'random_state': 42, 'n_jobs': -1}),
        'XGBoost': (xgb.XGBClassifier, {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03,
                                          'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
                                          'subsample': 0.8, 'colsample_bytree': 0.8,
                                          'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1}),
        'LightGBM': (lgb.LGBMClassifier, {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03,
                                           'class_weight': 'balanced', 'subsample': 0.8, 'colsample_bytree': 0.8,
                                           'random_state': 42, 'n_jobs': -1, 'verbose': -1}),
    }

    model_class, best_params = model_map[best_model_name]

    raw_model = model_class(**best_params)
    raw_model.fit(X_train, y)

    calibrated = CalibratedClassifierCV(raw_model, method='isotonic', cv=5)
    calibrated.fit(X_train, y)
    test_probs = calibrated.predict_proba(X_test)[:, 1]

    joblib.dump(calibrated, 'model/fraud_model.pkl')

    with open('model/model_results.json', 'w') as f:
        def convert(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
            if isinstance(o, (np.bool_,)): return bool(o)
            return o
        json.dump({k: {kk: convert(vv) if not isinstance(vv, (str, int, float, bool)) and kk != 'per_fold'
                       else vv for kk, vv in v.items() if kk != 'per_fold'}
                   for k, v in results.items()}, f, indent=2, default=str)

    return calibrated, test_probs, results, best_model_name
