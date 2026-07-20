import numpy as np
import tensorflow as tf
import csv
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def train_baseline_model(X_train, X_test, y_train, y_test):
    print("\n--- Training Baseline Model: Random Forest with GridSearchCV ---")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    rf_model = grid_search.best_estimator_
    
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, rf_preds)
    f1 = f1_score(y_test, rf_preds, average='weighted')
    precision = precision_score(y_test, rf_preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, rf_preds, average='weighted', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, rf_probs, multi_class='ovr', labels=rf_model.classes_)
    except Exception as e:
        print(f"Warning: ROC-AUC could not be computed ({e}). Setting to NaN.")
        roc_auc = float('nan')
    
    target_names = ["Budget Customers", "Premium Customers", "High-Risk Customers", "Regular Customers"]
    print("Random Forest Accuracy:", acc)
    print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_preds, target_names=target_names))
    
    # Save metrics to CSV
    import pandas as pd
    metrics_path = 'outputs/model_metrics.csv'
    os.makedirs('outputs', exist_ok=True)
    new_row = {'Model_Name': 'Random Forest', 'Model_Type': 'Classification', 'Accuracy': f"{acc:.4f}", 'Precision': f"{precision:.4f}", 'Recall': f"{recall:.4f}", 'F1_Score': f"{f1:.4f}", 'ROC_AUC': f"{roc_auc:.4f}", 'Silhouette_Score': '', 'Davies_Bouldin_Index': '', 'Calinski_Harabasz_Index': '', 'Notes': str(grid_search.best_params_)}
    
    if os.path.exists(metrics_path):
        try:
            df_m = pd.read_csv(metrics_path)
            if 'Random Forest' in df_m['Model_Name'].values:
                for k, v in new_row.items(): df_m.loc[df_m['Model_Name'] == 'Random Forest', k] = v
            else:
                df_m = pd.concat([df_m, pd.DataFrame([new_row])], ignore_index=True)
        except Exception:
            df_m = pd.DataFrame([new_row])
    else:
        df_m = pd.DataFrame([new_row])
    df_m.to_csv(metrics_path, index=False)
    return rf_model, acc, f1, roc_auc

def train_deep_learning_model(X_train, X_val, X_test, y_train, y_val, y_test, num_classes=4):
    print("\n--- Training Deep Learning Model: DNN ---")
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    
    dnn_model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])
    
    custom_adam = Adam(learning_rate=0.0005)
    dnn_model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    
    history = dnn_model.fit(
        X_train, y_train_cat, 
        epochs=100, 
        batch_size=64, 
        validation_data=(X_val, y_val_cat),
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Evaluate DNN
    dnn_loss, dnn_acc = dnn_model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nDeep Neural Network Accuracy: {dnn_acc:.4f}")
    
    # DNN Classification Report
    dnn_preds_prob = dnn_model.predict(X_test)
    dnn_preds = np.argmax(dnn_preds_prob, axis=1)
    f1 = f1_score(y_test, dnn_preds, average='weighted')
    precision = precision_score(y_test, dnn_preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, dnn_preds, average='weighted', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test_cat, dnn_preds_prob, multi_class='ovr')
    except Exception as e:
        print(f"Warning: DNN ROC-AUC could not be computed ({e}). Setting to NaN.")
        roc_auc = float('nan')
    
    target_names = ["Budget Customers", "Premium Customers", "High-Risk Customers", "Regular Customers"]
    print("\nDNN Classification Report:\n", classification_report(y_test, dnn_preds, target_names=target_names))
    
    # Save metrics to CSV
    import pandas as pd
    metrics_path = 'outputs/model_metrics.csv'
    os.makedirs('outputs', exist_ok=True)
    new_row = {'Model_Name': 'Deep Neural Network', 'Model_Type': 'Classification', 'Accuracy': f"{dnn_acc:.4f}", 'Precision': f"{precision:.4f}", 'Recall': f"{recall:.4f}", 'F1_Score': f"{f1:.4f}", 'ROC_AUC': f"{roc_auc:.4f}", 'Silhouette_Score': '', 'Davies_Bouldin_Index': '', 'Calinski_Harabasz_Index': '', 'Notes': '100 epochs, Adam lr=0.0005'}
    
    if os.path.exists(metrics_path):
        try:
            df_m = pd.read_csv(metrics_path)
            if 'Deep Neural Network' in df_m['Model_Name'].values:
                for k, v in new_row.items(): df_m.loc[df_m['Model_Name'] == 'Deep Neural Network', k] = v
            else:
                df_m = pd.concat([df_m, pd.DataFrame([new_row])], ignore_index=True)
        except Exception:
            df_m = pd.DataFrame([new_row])
    else:
        df_m = pd.DataFrame([new_row])
    df_m.to_csv(metrics_path, index=False)
    return dnn_model, dnn_acc, f1, roc_auc, history
