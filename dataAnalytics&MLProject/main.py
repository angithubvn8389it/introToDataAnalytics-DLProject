import os
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import get_preprocessing_params, apply_preprocessing, scale_features_fit, scale_features_transform
from src.feature_engineering import engineer_features
from src.eda import perform_eda
from src.autoencoder import build_and_train_autoencoder
from src.clustering import perform_clustering, visualize_clusters, evaluate_clusters
from src.models import train_baseline_model, train_deep_learning_model

def main():
    print("=== Bank Customer Classification & Segmentation Pipeline ===")
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # 1. Load Data
    data_path = os.path.join('data', 'raw_data', 'creditcardDataset.csv')
    df = load_data(data_path)
    
    # 2. EDA
    perform_eda(df)
    
    # 3. Train Val Test Split First (To prevent data leakage)
    df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
    
    # Save the splits to CSV for verification
    os.makedirs(os.path.join('data', 'processed_data'), exist_ok=True)
    df_train.to_csv(os.path.join('data', 'processed_data', 'train_split.csv'), index=False)
    df_val.to_csv(os.path.join('data', 'processed_data', 'val_split.csv'), index=False)
    df_test.to_csv(os.path.join('data', 'processed_data', 'test_split.csv'), index=False)
    
    # 4. Preprocessing (Fit on Train, Transform on all)
    preprocessing_params = get_preprocessing_params(df_train)
    df_train_clean = apply_preprocessing(df_train, preprocessing_params)
    df_val_clean = apply_preprocessing(df_val, preprocessing_params)
    df_test_clean = apply_preprocessing(df_test, preprocessing_params)
    
    # Plot Data Amount Before and After Preprocessing
    plt.figure(figsize=(8, 5))
    data_sizes = [len(df_train) + len(df_val) + len(df_test), len(df_train_clean) + len(df_val_clean) + len(df_test_clean)]
    labels = ['Before Preprocessing', 'After Preprocessing']
    bars = plt.bar(labels, data_sizes, color=['#ff9999', '#66b3ff'])
    plt.ylabel('Number of Samples')
    plt.title('Data Amount Before and After Preprocessing')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')
    plt.savefig(os.path.join('figures', 'data_amount_preprocessing.png'))
    plt.close()

    
    # 5. Feature Engineering
    df_train_engineered = engineer_features(df_train_clean)
    df_val_engineered = engineer_features(df_val_clean)
    df_test_engineered = engineer_features(df_test_clean)
    
    # 6. Scaling (Fit on Train, Transform on all)
    X_train_scaled, scaler = scale_features_fit(df_train_engineered)
    X_val_scaled = scale_features_transform(df_val_engineered, scaler)
    X_test_scaled = scale_features_transform(df_test_engineered, scaler)
    
    # Save scaler
    with open(os.path.join('outputs', 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    # 7. Autoencoder for Dimensionality Reduction (Train on Train only)
    autoencoder, encoder, X_train_latent = build_and_train_autoencoder(X_train_scaled, encoding_dim=10, epochs=30)
    autoencoder.save(os.path.join('outputs', 'autoencoder.h5'))
    encoder.save(os.path.join('outputs', 'encoder.h5'))
    
    X_val_latent = encoder.predict(X_val_scaled)
    X_test_latent = encoder.predict(X_test_scaled)
    
    # 8. Clustering Models (Fit on Train Latent Features)
    evaluate_clusters(X_train_latent, max_k=10, output_dir='figures')
    kmeans, df_train_engineered, y_train, sil, db, ch = perform_clustering(X_train_latent, df_train_engineered, n_clusters=4)
    visualize_clusters(X_train_latent, y_train, output_path=os.path.join('outputs', 'customer_segments_pca.png'))
    
    # Predict clusters for Val and Test sets
    y_val = kmeans.predict(X_val_latent)
    y_test = kmeans.predict(X_test_latent)
    
    # 9. Classification Models & Metrics
    # Train Random Forest
    rf_model, rf_acc, rf_f1, rf_roc = train_baseline_model(X_train_latent, X_test_latent, y_train, y_test)
    
    # Train DNN
    dnn_model, dnn_acc, dnn_f1, dnn_roc, history = train_deep_learning_model(X_train_latent, X_val_latent, X_test_latent, y_train, y_val, y_test)
    
    # Save training history as JSON
    with open(os.path.join('outputs', 'training_history.json'), 'w') as f:
        json.dump(history.history, f, indent=4)
        
    # Save training history as CSV
    import pandas as pd
    hist_dict = history.history
    epochs = len(list(hist_dict.values())[0])
    hist_df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_accuracy': hist_dict.get('accuracy', []),
        'train_loss': hist_dict.get('loss', []),
        'val_accuracy': hist_dict.get('val_accuracy', []),
        'val_loss': hist_dict.get('val_loss', []),
        'learning_rate': hist_dict.get('learning_rate', hist_dict.get('lr', []))
    })
    hist_df.to_csv(os.path.join('outputs', 'training_history.csv'), index=False)
    
    # Plot Metrics

    
    # 2.5 ROC Curves (Separate for RF and DNN) with Micro/Macro averages
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import numpy as np
    
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    n_classes = y_test_bin.shape[1]
    
    rf_probs = rf_model.predict_proba(X_test_latent)
    dnn_probs = dnn_model.predict(X_test_latent)
    
    colors = ['cyan', 'orange', 'cornflowerblue', 'green']
    
    def plot_roc_for_model(probs, model_name, filename):
        plt.figure(figsize=(8, 6))
        
        # Compute micro-average ROC curve and ROC area
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, label=f'Micro-avg (AUC = {roc_auc_micro:.3f})',
                 color='deeppink', linestyle=':', linewidth=4)
                 
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([roc_curve(y_test_bin[:, i], probs[:, i])[0] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], probs[:, i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        mean_tpr /= n_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, label=f'Macro-avg (AUC = {roc_auc_macro:.3f})',
                 color='navy', linestyle=':', linewidth=4)
                 
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], probs[:, i])
            roc_auc_i = auc(fpr_i, tpr_i)
            plt.plot(fpr_i, tpr_i, color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc_i:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - Multi-class ROC Curve')
        plt.legend(loc="lower right", fontsize='small')
        plt.savefig(os.path.join('figures', filename))
        plt.close()

    plot_roc_for_model(rf_probs, "Random Forest", "roc_curve_rf.png")
    plot_roc_for_model(dnn_probs, "Deep Neural Network", "roc_curve_dnn.png")
    
    # 2.7 Confusion Matrices
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    rf_preds = rf_model.predict(X_test_latent)
    dnn_preds = np.argmax(dnn_probs, axis=1)
    
    def plot_confusion_matrix(y_true, y_pred, model_name, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
                    yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.savefig(os.path.join('figures', filename))
        plt.close()
        
    plot_confusion_matrix(y_test, rf_preds, "Random Forest", "cm_rf.png")
    plot_confusion_matrix(y_test, dnn_preds, "Deep Neural Network", "cm_dnn.png")
    
    # 2.8 Feature Importances (Auxiliary Model for Interpretability)
    print("\n--- Training Auxiliary RF for Feature Importance ---")
    from sklearn.ensemble import RandomForestClassifier
    aux_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    aux_rf.fit(X_train_scaled, y_train)
    
    importances = aux_rf.feature_importances_
    feature_names = scaler.feature_names_in_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances (Original Features)")
    plt.bar(range(len(importances)), importances[indices], align="center", color='teal')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'feature_importances.png'))
    plt.close()
    
    # 3. F1-Score
    plt.figure(figsize=(8, 5))
    plt.bar(['Random Forest', 'Deep Neural Network'], [rf_f1, dnn_f1], color=['blue', 'green'])
    plt.ylabel('F1-Score (Weighted)')
    plt.title('Model vs F1-Score')
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join('figures', 'model_vs_f1_score.png'))
    plt.close()
    
    # 4. Epoch vs Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Epoch vs Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('figures', 'epoch_vs_loss.png'))
    plt.close()
    
    # 5. Epoch vs Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('figures', 'epoch_vs_accuracy.png'))
    plt.close()
    
    # Save best models
    best_models = {}
    if rf_roc >= dnn_roc:
        best_models['best_model'] = rf_model
        best_models['type'] = 'RandomForest'
    else:
        dnn_model.save(os.path.join('outputs', 'best_dnn.h5'))
        best_models['best_model_path'] = 'best_dnn.h5'
        best_models['type'] = 'DNN'
        
    with open(os.path.join('outputs', 'best_models.pkl'), 'wb') as f:
        pickle.dump(best_models, f)
    print("Saved best models to 'outputs/best_models.pkl'")
    
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()
