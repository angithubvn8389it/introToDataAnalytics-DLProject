import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

def assign_cluster_labels(df, cluster_labels):
    df_temp = df.copy()
    df_temp['Cluster'] = cluster_labels
    
    # Calculate means for each cluster
    if 'PURCHASES' in df_temp.columns and 'CASH_ADVANCE' in df_temp.columns and 'BALANCE' in df_temp.columns:
        cluster_means = df_temp.groupby('Cluster')[['BALANCE', 'PURCHASES', 'CASH_ADVANCE']].mean()
        
        sorted_purchases = cluster_means['PURCHASES'].sort_values()
        sorted_cash = cluster_means['CASH_ADVANCE'].sort_values()
        
        premium_id = sorted_purchases.index[-1]
        
        high_risk_candidates = [idx for idx in sorted_cash.index[::-1] if idx != premium_id]
        high_risk_id = high_risk_candidates[0] if high_risk_candidates else premium_id
        
        budget_candidates = [idx for idx in sorted_purchases.index if idx not in [premium_id, high_risk_id]]
        budget_id = budget_candidates[0] if budget_candidates else premium_id
        
        regular_candidates = [idx for idx in cluster_means.index if idx not in [premium_id, high_risk_id, budget_id]]
        regular_id = regular_candidates[0] if regular_candidates else premium_id
        
        id_map = {
            budget_id: 0,
            premium_id: 1,
            high_risk_id: 2,
            regular_id: 3
        }
        
        new_cluster_labels = [id_map.get(lbl, lbl) for lbl in cluster_labels]
        
        label_map = {
            0: "Budget Customers",
            1: "Premium Customers",
            2: "High-Risk Customers",
            3: "Regular Customers"
        }
    else:
        new_cluster_labels = cluster_labels
        label_map = {i: f"Cluster {i}" for i in np.unique(cluster_labels)}
        id_map = {i: i for i in np.unique(cluster_labels)}
        
    named_labels = [label_map[label] for label in new_cluster_labels]
    return new_cluster_labels, named_labels, label_map, id_map

def perform_clustering(features_for_clustering, df, n_clusters=4):
    print("Performing K-Means Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_for_clustering)
    
    # Add labels back to dataframe
    df['Cluster'] = cluster_labels
    
    # Calculate metrics
    sil_score = silhouette_score(features_for_clustering, cluster_labels)
    db_score = davies_bouldin_score(features_for_clustering, cluster_labels)
    ch_score = calinski_harabasz_score(features_for_clustering, cluster_labels)
    
    print(f"Clustering complete. Cluster counts:\n{df['Cluster'].value_counts()}")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    print(f"Calinski-Harabasz Index: {ch_score:.4f}")
    
    # Save metrics to CSV
    import pandas as pd
    metrics_path = 'outputs/model_metrics.csv'
    os.makedirs('outputs', exist_ok=True)
    new_row = {'Model_Name': 'K-Means', 'Model_Type': 'Clustering', 'Accuracy': '', 'Precision': '', 'Recall': '', 'F1_Score': '', 'ROC_AUC': '', 'Silhouette_Score': f"{sil_score:.4f}", 'Davies_Bouldin_Index': f"{db_score:.4f}", 'Calinski_Harabasz_Index': f"{ch_score:.4f}", 'Notes': f"n_clusters={n_clusters}"}
    
    if os.path.exists(metrics_path):
        try:
            df_m = pd.read_csv(metrics_path)
            if 'K-Means' in df_m['Model_Name'].values:
                for k, v in new_row.items(): df_m.loc[df_m['Model_Name'] == 'K-Means', k] = v
            else:
                df_m = pd.concat([df_m, pd.DataFrame([new_row])], ignore_index=True)
        except Exception:
            df_m = pd.DataFrame([new_row])
    else:
        df_m = pd.DataFrame([new_row])
    df_m.to_csv(metrics_path, index=False)
    new_cluster_labels, named_labels, label_map, id_map = assign_cluster_labels(df, cluster_labels)
    cluster_labels = np.array(new_cluster_labels)
    df['Cluster'] = cluster_labels
    df['Customer_Type'] = named_labels
    print("Assigned Labels based on cluster profiles:")
    for cid, cname in label_map.items():
        print(f"Cluster {cid}: {cname}")

    return kmeans, df, cluster_labels, sil_score, db_score, ch_score, id_map

def evaluate_clusters(features, max_k=10, output_dir='figures'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    inertias = []
    silhouette_scores = []
    K = range(2, max_k + 1)
    
    print("Evaluating optimal k (Elbow Method & Silhouette)...")
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, labels))
        
    # Plot Elbow (Inertia)
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig(os.path.join(output_dir, 'elbow_method.png'))
    plt.close()
    
    # Plot Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(K, silhouette_scores, 'rx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs k')
    plt.savefig(os.path.join(output_dir, 'silhouette_vs_k.png'))
    plt.close()
    print("Saved Elbow and Silhouette plots.")


def visualize_clusters(latent_features, cluster_labels, output_path='outputs/customer_segments_pca.png'):
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(latent_features)
    
    # Format labels for the plot legend
    plot_labels = [f"Cluster {label}" for label in cluster_labels]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=pca_features[:, 0], 
        y=pca_features[:, 1], 
        hue=plot_labels, 
        palette='viridis',
        s=70,
        edgecolor="black",
        linewidth=0.3
    )
    plt.title('Customer Segments (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
 
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved PCA plot as '{output_path}'")
    plt.close()

    # Now create and save a t-SNE plot
    tsne_path = output_path.replace('pca', 'tsne')
    print("Generating t-SNE visualization (this may take a moment)...")
    tsne = TSNE(
        n_components=2, 
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )
    tsne_features = tsne.fit_transform(latent_features)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=tsne_features[:, 0], 
        y=tsne_features[:, 1], 
        hue=plot_labels, 
        palette='viridis',
        s=70,
        edgecolor="black",
        linewidth=0.3
    )
    plt.title('Customer Segments (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(tsne_path)
    print(f"Saved t-SNE plot as '{tsne_path}'")
    plt.close()
