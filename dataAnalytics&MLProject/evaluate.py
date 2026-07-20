import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import load_model

from src.data_loader import load_data
from src.preprocessing import get_preprocessing_params, apply_preprocessing, scale_features_transform
from src.feature_engineering import engineer_features

def evaluate_models():
    print("=== Testing Saved Models ===")
    
    # 1. Load Data
    data_path = os.path.join('data', 'raw_data', 'creditcardDataset.csv')
    df = load_data(data_path)
    
    # 2. Train Val Test Split First (To prevent data leakage, must match main.py)
    df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
    
    # 3. Preprocessing (Fit on Train, Transform on test)
    preprocessing_params = get_preprocessing_params(df_train)
    df_test_clean = apply_preprocessing(df_test, preprocessing_params)
    
    # 4. Feature Engineering
    df_test_engineered = engineer_features(df_test_clean)
    
    # 5. Scaling
    scaler_path = os.path.join('outputs', 'scaler.pkl')
    if not os.path.exists(scaler_path):
        print("Error: scaler.pkl not found")
        return
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    X_test_scaled = scale_features_transform(df_test_engineered, scaler)
    
    # 6. Autoencoder
    encoder_path = os.path.join('outputs', 'encoder.h5')
    if not os.path.exists(encoder_path):
        print(f"Error: Encoder not found at {encoder_path}")
        return
    encoder = load_model(encoder_path)
    
    X_test_latent = encoder.predict(X_test_scaled)
    
    # We do not have ground truth y_test since K-Means was not saved in main.py,
    # and re-running clustering might result in swapped labels. 
    # For demonstration, we just predict using the saved model and show its class distributions.
    
    # 7. Load the Best Models
    model_path = os.path.join('outputs', 'best_models.pkl')
    if not os.path.exists(model_path):
        print(f"Error: Saved model not found at {model_path}")
        return
        
    with open(model_path, 'rb') as f:
        best_models = pickle.load(f)
        
    print(f"\nLoaded Model Type: {best_models['type']}")
    
    if best_models['type'] == 'RandomForest':
        model = best_models['best_model']
        preds = model.predict(X_test_latent)
        unique, counts = np.unique(preds, return_counts=True)
        print("Predicted Class Distribution on Test Set:", dict(zip(unique, counts)))
        
    elif best_models['type'] == 'DNN':
        dnn_path = os.path.join('outputs', best_models['best_model_path'])
        model = load_model(dnn_path)
        
        preds_prob = model.predict(X_test_latent)
        preds = np.argmax(preds_prob, axis=1)
        unique, counts = np.unique(preds, return_counts=True)
        print("Predicted Class Distribution on Test Set:", dict(zip(unique, counts)))

if __name__ == "__main__":
    evaluate_models()

