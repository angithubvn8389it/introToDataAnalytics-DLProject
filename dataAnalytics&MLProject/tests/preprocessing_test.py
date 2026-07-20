import pandas as pd
import numpy as np
from src.preprocessing import get_preprocessing_params, apply_preprocessing, scale_features_fit

def test_preprocess_data():
    # Create dummy data
    data = {
        'CUST_ID': ['C1', 'C2', 'C3'],
        'MINIMUM_PAYMENTS': [100.0, np.nan, 300.0],
        'CREDIT_LIMIT': [1000.0, 2000.0, np.nan],
        'BALANCE': [500.0, 1500.0, 2500.0]
    }
    df = pd.DataFrame(data)
    
    params = get_preprocessing_params(df)
    processed_df = apply_preprocessing(df, params)
    scaled_features, scaler = scale_features_fit(processed_df)
    
    # Check that CUST_ID is dropped
    assert 'CUST_ID' not in processed_df.columns
    
    # Check missing values are filled (median of [100, 300] is 200, median of [1000, 2000] is 1500)
    assert processed_df['MINIMUM_PAYMENTS'].iloc[1] == 200.0
    assert processed_df['CREDIT_LIMIT'].iloc[2] == 1500.0
    
    # Check scaled features shape
    assert scaled_features.shape == processed_df.shape
