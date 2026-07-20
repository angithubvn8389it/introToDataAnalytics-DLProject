import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_outlier_bounds(df, columns):
    bounds = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = {
            'lower': Q1 - 1.5 * IQR,
            'upper': Q3 + 1.5 * IQR
        }
    return bounds

def apply_outliers(df, bounds):
    df_out = df.copy()
    for col, b in bounds.items():
        if col in df_out.columns:
            df_out[col] = np.clip(df_out[col], b['lower'], b['upper'])
    return df_out

def get_preprocessing_params(df):
    params = {
        'median_min_payments': df['MINIMUM_PAYMENTS'].median(),
        'median_credit_limit': df['CREDIT_LIMIT'].median()
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude CUST_ID from outlier bounds if it hasn't been dropped yet
    if 'CUST_ID' in numeric_cols:
        numeric_cols = numeric_cols.drop('CUST_ID')
    params['outlier_bounds'] = get_outlier_bounds(df, numeric_cols)
    return params

def apply_preprocessing(df, params):
    df_clean = df.copy()
    if 'CUST_ID' in df_clean.columns:
        df_clean = df_clean.drop('CUST_ID', axis=1)
        
    df_clean['MINIMUM_PAYMENTS'] = df_clean['MINIMUM_PAYMENTS'].fillna(params['median_min_payments'])
    df_clean['CREDIT_LIMIT'] = df_clean['CREDIT_LIMIT'].fillna(params['median_credit_limit'])
    
    # Cap outliers
    df_clean = apply_outliers(df_clean, params['outlier_bounds'])
    
    return df_clean

def scale_features_fit(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    return scaled_features, scaler

def scale_features_transform(df, scaler):
    scaled_features = scaler.transform(df)
    return scaled_features