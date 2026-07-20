import os
import pandas as pd
import numpy as np

from src.data_loader import load_data
from src.preprocessing import get_preprocessing_params, apply_preprocessing, scale_features_fit
from src.feature_engineering import engineer_features
from sklearn.cluster import KMeans

# 1. Re-run data pipeline up to clustering
data_path = os.path.join('data', 'raw_data', 'creditcardDataset.csv')
df = load_data(data_path)
params = get_preprocessing_params(df)
df_clean = apply_preprocessing(df, params)
df_engineered = engineer_features(df_clean)
scaled_features, _ = scale_features_fit(df_engineered)

# Load Autoencoder latent features
from tensorflow.keras.models import load_model
encoder = load_model(os.path.join('outputs', 'encoder.h5'))
latent_features = encoder.predict(scaled_features)

# Re-run K-Means (random_state ensures identical clusters)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(latent_features)

df_engineered['Cluster'] = cluster_labels

# Calculate the mean of each feature for each cluster
cluster_profiles = df_engineered.groupby('Cluster').mean().round(2)
print("\n--- Cluster Profiles ---")
# Print some of the most distinguishing columns
important_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'TENURE']
print(cluster_profiles[important_cols].to_string())
