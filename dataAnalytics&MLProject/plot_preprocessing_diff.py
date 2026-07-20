import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.preprocessing import get_preprocessing_params, apply_preprocessing

def generate_before_after_boxplots():
    df_raw = pd.read_csv('data/raw_data/creditcardDataset.csv')
    
    # Apply preprocessing to the whole dataset for visualization
    params = get_preprocessing_params(df_raw)
    df_clean = apply_preprocessing(df_raw, params)
    
    features = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']
    
    plt.figure(figsize=(16, 12))
    
    for i, col in enumerate(features):
        # Before preprocessing
        plt.subplot(4, 2, 2*i + 1)
        sns.boxplot(x=df_raw[col], color='lightcoral')
        plt.title(f'{col} (Before Preprocessing - Outliers Present)')
        
        # After preprocessing
        plt.subplot(4, 2, 2*i + 2)
        sns.boxplot(x=df_clean[col], color='lightgreen')
        plt.title(f'{col} (After Preprocessing - Outliers Clipped)')
        
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out_path = os.path.join('figures', 'boxplot_before_after_preprocessing.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved '{out_path}'")

if __name__ == '__main__':
    generate_before_after_boxplots()
