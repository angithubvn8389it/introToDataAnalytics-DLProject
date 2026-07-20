import os
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df, output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)
    print("Performing EDA...")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 2. Distributions
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']):
        if col in df.columns:
            plt.subplot(2, 2, i+1)
            sns.histplot(df[col], bins=30, kde=True, color='skyblue')
            plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()
    
    # 3. Boxplots
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']):
        if col in df.columns:
            plt.subplot(2, 2, i+1)
            sns.boxplot(x=df[col], color='lightgreen')
            plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_boxplots.png'))
    plt.close()
    
    print("EDA complete. Plots saved to 'figures/'.")
