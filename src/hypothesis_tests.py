import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def check_normality(data, feature_name):
    if len(data) < 5000:
        _, p = stats.shapiro(data)
        test = 'Shapiro-Wilk'
    else:
        _, p = stats.normaltest(data)
        test = 'D\'Agostino-Pearson'
    return p > 0.05, p, test


def hypothesis_test_feature_selection(X_train, y, numerical_features, alpha=0.05):
    results = []

    for feat in numerical_features:
        fraud_vals = X_train.loc[y == 1, feat].dropna()
        legit_vals = X_train.loc[y == 0, feat].dropna()

        if len(fraud_vals) < 3 or len(legit_vals) < 3:
            results.append({
                'feature': feat,
                'test': 'skipped',
                'statistic': np.nan,
                'p_value': np.nan,
                'reject_null': np.nan,
                'conclusion': 'insufficient_data'
            })
            continue

        fraud_normal, _, fraud_test = check_normality(fraud_vals, feat)
        legit_normal, _, legit_test = check_normality(legit_vals, feat)

        n_fraud = len(fraud_vals)
        n_legit = len(legit_vals)

        if fraud_normal and legit_normal:
            stat, p = stats.ttest_ind(fraud_vals, legit_vals, equal_var=False)
            test_name = 'Welch t-test'
        else:
            stat, p = stats.mannwhitneyu(fraud_vals, legit_vals, alternative='two-sided')
            test_name = 'Mann-Whitney U'

        results.append({
            'feature': feat,
            'test': test_name,
            'statistic': stat,
            'p_value': p,
            'reject_null': p < alpha,
            'conclusion': 'significant_difference' if p < alpha else 'no_significant_difference'
        })

    return pd.DataFrame(results).sort_values('p_value')


def chi2_feature_selection(X_train, y, categorical_features):
    from sklearn.feature_selection import chi2
    results = []

    for feat in categorical_features:
        ct = pd.crosstab(X_train[feat], y)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            results.append({
                'feature': feat,
                'chi2_stat': np.nan,
                'p_value': np.nan,
                'reject_null': np.nan,
                'conclusion': 'insufficient_dof'
            })
            continue
        try:
            chi2_stat, p = stats.chi2_contingency(ct)[:2]
            results.append({
                'feature': feat,
                'chi2_stat': chi2_stat,
                'p_value': p,
                'reject_null': p < 0.05,
                'conclusion': 'dependent' if p < 0.05 else 'independent'
            })
        except Exception as e:
            results.append({
                'feature': feat,
                'chi2_stat': np.nan,
                'p_value': np.nan,
                'reject_null': np.nan,
                'conclusion': f'error: {str(e)}'
            })

    return pd.DataFrame(results).sort_values('p_value')
