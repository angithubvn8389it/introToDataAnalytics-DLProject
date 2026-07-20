import pandas as pd

def engineer_features(df):
    """
    Create new meaningful features based on financial logic.
    """
    df_feat = df.copy()
    
    # 1. Balance to Credit Limit Ratio
    if 'BALANCE' in df_feat.columns and 'CREDIT_LIMIT' in df_feat.columns:
        df_feat['BALANCE_TO_CREDIT_LIMIT_RATIO'] = df_feat['BALANCE'] / (df_feat['CREDIT_LIMIT'] + 1e-6)
        
    # 2. Payments to Minimum Payments Ratio
    if 'PAYMENTS' in df_feat.columns and 'MINIMUM_PAYMENTS' in df_feat.columns:
        df_feat['PAYMENTS_TO_MINIMUM_PAYMENTS_RATIO'] = df_feat['PAYMENTS'] / (df_feat['MINIMUM_PAYMENTS'] + 1e-6)
        
    # 3. Cash Advance to Balance Ratio
    if 'CASH_ADVANCE' in df_feat.columns and 'BALANCE' in df_feat.columns:
        df_feat['CASH_ADVANCE_TO_BALANCE_RATIO'] = df_feat['CASH_ADVANCE'] / (df_feat['BALANCE'] + 1e-6)

    return df_feat
