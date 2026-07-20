import pandas as pd
import numpy as np
from src.clustering import perform_clustering

def test_perform_clustering():
    # Create dummy scaled features
    np.random.seed(42)
    scaled_features = np.random.rand(10, 3)
    df = pd.DataFrame(scaled_features, columns=['F1', 'F2', 'F3'])
    
    # Perform clustering
    df_result, cluster_labels, sil_score = perform_clustering(scaled_features, df, n_clusters=2)
    
    # Check if 'Cluster' column is added
    assert 'Cluster' in df_result.columns
    
    # Check if we have the correct number of labels
    assert len(cluster_labels) == 10
    
    # Check if cluster labels are 0 and 1
    assert set(cluster_labels).issubset({0, 1})
