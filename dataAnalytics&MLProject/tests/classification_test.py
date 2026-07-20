import numpy as np
from src.models import train_baseline_model, train_deep_learning_model

def test_models():
    # Generate dummy data for classification
    np.random.seed(42)
    X_train = np.random.rand(80, 5)
    y_train = np.random.randint(0, 4, size=(80,))
    
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 4, size=(20,))
    
    # Test Baseline Model
    rf_model, rf_acc, rf_f1 = train_baseline_model(X_train, X_test, y_train, y_test)
    assert rf_model is not None
    assert 0.0 <= rf_acc <= 1.0
    
    # Test Deep Learning Model
    dnn_model, dnn_acc, dnn_f1, history = train_deep_learning_model(X_train, X_test, y_train, y_test, num_classes=4)
    assert dnn_model is not None
    assert 0.0 <= dnn_acc <= 1.0
