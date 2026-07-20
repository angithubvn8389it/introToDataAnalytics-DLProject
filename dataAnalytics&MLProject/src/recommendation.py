import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.feature_engineering import engineer_features

def get_recommendations(cluster_id, input_data):
    """
    Returns personalized recommendations based on the predicted cluster 
    and specific data points of the user.
    """
    recommendations = []
    
    # Base recommendations
    if cluster_id == 0:
        recommendations.append("💳 **Product:** Standard Cash-Back Credit Card")
        recommendations.append("📈 **Action:** Offer a moderate credit limit increase based on consistent usage.")
        recommendations.append("🏦 **Service:** Basic High-Yield Savings Account.")
        
    elif cluster_id == 1:
        recommendations.append("🛫 **Product:** Premium Travel Rewards Card (e.g., Chase Sapphire Reserve, Amex Platinum)")
        recommendations.append("🎩 **Service:** 24/7 Concierge Service and VIP Airport Lounge Access.")
        recommendations.append("💼 **Service:** Priority Wealth Management & Investment Advisory.")
        
        if input_data.get('PURCHASES', 0) > 5000:
            recommendations.append("⭐ **Personalized:** Customer has exceptionally high purchases. Offer luxury brand point multipliers.")
            
    elif cluster_id == 2:
        recommendations.append("📉 **Product:** Low-APR Balance Transfer Credit Card")
        recommendations.append("🤝 **Service:** Debt Consolidation Loan.")
        recommendations.append("📊 **Service:** Complimentary Financial Planning / Debt Management Consultation.")
        
        if input_data.get('CASH_ADVANCE', 0) > 2000:
            recommendations.append("🚨 **Personalized:** Customer relies heavily on cash advances. Offer a fixed-rate personal loan to replace high-interest cash advance fees.")
            
    elif cluster_id == 3:
        recommendations.append("🆓 **Product:** No-Annual-Fee Rewards Card")
        recommendations.append("🤖 **Service:** Automated Investing (Robo-Advisor) linked to checking account.")
        recommendations.append("📱 **Feature:** Spending trackers and automated 'pay-in-full' autopay setup.")
        
        if input_data.get('PAYMENTS', 0) > 1000:
            recommendations.append("💸 **Personalized:** Customer pays off large amounts rapidly. Offer high-yield checking accounts to park their cash.")
            
    return recommendations

def predict_and_recommend(input_dict, base_dir):
    """
    Takes raw input dictionary, runs the pipeline, predicts cluster, and returns recommendations.
    """
    # 1. Convert to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Ensure all required columns exist (fill missing with 0 for demo purposes)
    expected_cols = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
       'TENURE']
       
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    # 2. Feature Engineering
    df_engineered = engineer_features(df.copy())
    
    # 3. Scaling
    scaler_path = os.path.join(base_dir, 'outputs', 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Ensure column order matches exactly what scaler expects
    df_engineered = df_engineered[scaler.feature_names_in_]
        
    scaled_features = scaler.transform(df_engineered)
    
    # 4. Autoencoder
    encoder_path = os.path.join(base_dir, 'outputs', 'encoder.h5')
    encoder = load_model(encoder_path)
    latent_features = encoder.predict(scaled_features)
    
    # 5. Model Prediction
    model_path = os.path.join(base_dir, 'outputs', 'best_models.pkl')
    with open(model_path, 'rb') as f:
        best_models = pickle.load(f)
        
    model_type = best_models.get('type')
    if model_type == 'RandomForest':
        model = best_models['best_model']
        cluster_pred = model.predict(latent_features)[0]
    else:
        # Load DNN
        dnn_path = os.path.join(base_dir, 'outputs', best_models['best_model_path'])
        model = load_model(dnn_path)
        pred_probs = model.predict(latent_features)
        cluster_pred = np.argmax(pred_probs, axis=1)[0]
        
    # 6. Get Recommendations
    recs = get_recommendations(cluster_pred, input_dict)
    
    return cluster_pred, recs
