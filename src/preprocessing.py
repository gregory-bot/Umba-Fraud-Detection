import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_data(data_dir='data'):
    train = pd.read_csv(f'{data_dir}/train.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')
    identity = pd.read_csv(f'{data_dir}/identity.csv')
    return train, test, identity


def aggregate_identity(identity):
    if identity.empty or 'TransactionID' not in identity.columns:
        return pd.DataFrame()
    agg = identity.groupby('TransactionID').agg({
        'DeviceType': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        'DeviceInfo': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        'id_01': 'mean', 'id_02': 'mean', 'id_03': 'mean', 'id_04': 'mean',
        'id_05': 'mean', 'id_06': 'mean', 'id_07': 'mean', 'id_08': 'mean',
        'id_09': 'mean', 'id_10': 'mean', 'id_11': 'mean',
    }).reset_index()
    agg['identity_rows'] = identity.groupby('TransactionID').size().values
    return agg


def extract_email_domain_features(df):
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            df[f'{col}_prefix'] = df[col].fillna('missing').apply(
                lambda x: x.split('.')[0] if '.' in str(x) else str(x)
            )
    return df


def create_features(df, is_train=True, freq_encodings=None):
    df = df.copy()
    df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
    df['TransactionAmt_sqrt'] = np.sqrt(df['TransactionAmt'])
    df['currency_KE'] = (df['currency'] == 'KES').astype(int)
    df['country_KE'] = (df['country'] == 'KE').astype(int)
    df['amt_per_currency'] = df.groupby('currency')['TransactionAmt'].transform('mean')
    df['amt_deviation'] = df['TransactionAmt'] / (df['amt_per_currency'] + 1)
    df['amt_x_sender_txn'] = df['TransactionAmt'] * (df['sender_prev_txn_count'] + 1)
    if 'card_type' in df.columns:
        card_type_dummies = pd.get_dummies(df['card_type'], prefix='card_type', dummy_na=True)
        df = pd.concat([df, card_type_dummies], axis=1)
    if 'channel' in df.columns:
        channel_dummies = pd.get_dummies(df['channel'], prefix='channel', dummy_na=True)
        df = pd.concat([df, channel_dummies], axis=1)
    for ch in ['mobile_money', 'p2p', 'bank_transfer', 'card']:
        col = f'channel_{ch}'
        if col in df.columns:
            df[f'amt_x_{ch}'] = df['TransactionAmt'] * df[col]
    df['recipient_account_age_days'] = df['recipient_account_age_days'].fillna(-1)
    df['sender_prev_txn_count'] = df['sender_prev_txn_count'].fillna(-1)
    df['account_age_binned'] = pd.cut(df['recipient_account_age_days'],
                                       bins=[-2, 0, 7, 30, 365, 99999],
                                       labels=[0, 1, 2, 3, 4]).astype(int)
    df = extract_email_domain_features(df)
    for c in ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']:
        if c in df.columns: df[c] = df[c].fillna(-1)
    for c in ['dist1', 'dist2']:
        if c in df.columns: df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
    for c in [f'C{i}' for i in range(1, 9)]:
        if c in df.columns: df[c] = df[c].fillna(0)
    for c in [f'D{i}' for i in range(1, 6)]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
            if df[c].nunique() > 1:
                p = (df[c] - df[c].min()) / (df[c].max() - df[c].min() + 1e-10)
                df[f'{c}_scaled'] = p
    for c in [f'M{i}' for i in range(1, 7)]:
        if c in df.columns: df[c] = df[c].fillna('missing')
    for c in [f'V{i}' for i in range(1, 21)]:
        if c in df.columns: df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
    v_cols = [f'V{i}' for i in range(1, 21) if f'V{i}' in df.columns]
    if v_cols:
        df['V_missing_count'] = df[v_cols].isna().sum(axis=1)
        df['V_zero_count'] = (df[v_cols] == 0).sum(axis=1)
    for c in ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06',
              'id_07', 'id_08', 'id_09', 'id_10', 'id_11']:
        if c in df.columns: df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
    if 'DeviceType' in df.columns: df['DeviceType'] = df['DeviceType'].fillna('unknown')
    if 'DeviceInfo' in df.columns: df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown')
    if 'identity_rows' in df.columns: df['identity_rows'] = df['identity_rows'].fillna(0).astype(int)
    if 'card_bank' in df.columns: df['card_bank'] = df['card_bank'].fillna('unknown')
    high_card_cols = ['card1', 'card_bank', 'addr1', 'DeviceInfo', 'P_emaildomain', 'R_emaildomain']
    if is_train:
        freq_encodings = {}
        for col in high_card_cols:
            if col in df.columns:
                vc = df[col].value_counts()
                freq_encodings[col] = vc.to_dict()
                df[f'{col}_freq'] = df[col].map(vc).fillna(0)
        return df, freq_encodings
    else:
        if freq_encodings is not None:
            for col in high_card_cols:
                if col in df.columns and col in freq_encodings:
                    df[f'{col}_freq'] = df[col].map(freq_encodings[col]).fillna(0)
        return df, None


def encode_categoricals(train_df, test_df, cat_cols):
    if not cat_cols:
        return train_df, test_df
    full = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    for col in cat_cols:
        if col in full.columns:
            le = LabelEncoder()
            full[col] = le.fit_transform(full[col].astype(str))
    train_enc = full.iloc[:len(train_df)].copy()
    test_enc = full.iloc[len(train_df):].copy()
    return train_enc, test_enc


def preprocess(train, test, identity):
    identity_agg = aggregate_identity(identity)
    if not identity_agg.empty:
        train = train.merge(identity_agg, on='TransactionID', how='left')
        test = test.merge(identity_agg, on='TransactionID', how='left')
    train, freq_enc = create_features(train, is_train=True)
    test, _ = create_features(test, is_train=False, freq_encodings=freq_enc)
    cat_cols = ['card_type', 'channel', 'card_bank', 'P_emaildomain',
                'R_emaildomain', 'P_emaildomain_prefix', 'R_emaildomain_prefix',
                'DeviceType', 'DeviceInfo',
                'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    existing_cats = [c for c in cat_cols if c in train.columns]
    train, test = encode_categoricals(train, test, existing_cats)
    drop_cols = ['TransactionID', 'TransactionDT', 'isFraud',
                 'flagged_for_review', 'currency', 'country',
                 'P_emaildomain', 'R_emaildomain',
                 'P_emaildomain_prefix', 'R_emaildomain_prefix',
                 'card_type', 'channel', 'card_bank', 'DeviceType',
                 'DeviceInfo', 'amt_per_currency']
    drop_cols = [c for c in drop_cols if c in train.columns]
    y = train['isFraud'].values if 'isFraud' in train.columns else None
    X_train = train.drop(columns=drop_cols)
    X_test = test.drop(columns=[c for c in drop_cols if c != 'isFraud'], errors='ignore')
    test_cols = [c for c in X_train.columns if c in X_test.columns]
    X_test = X_test[test_cols]
    X_train = X_train[test_cols]
    return X_train, X_test, y
