import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.preprocessing import get_preprocessing_params, apply_preprocessing, scale_features_fit
from src.feature_engineering import engineer_features
from sklearn.cluster import KMeans
from src.clustering import visualize_clusters

# Load data and encoder
df = pd.read_csv('data/raw_data/creditcardDataset.csv').dropna()
params = get_preprocessing_params(df)
df_clean = apply_preprocessing(df, params)
df_eng = engineer_features(df_clean)
scaled, scaler = scale_features_fit(df_eng)
encoder = load_model('outputs/encoder.h5')
latent = encoder.predict(scaled)

# Get clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(latent)

# Visualize
visualize_clusters(latent, labels, output_path='outputs/customer_segments_pca.png')
print("Done!")
