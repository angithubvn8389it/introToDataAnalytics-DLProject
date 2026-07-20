import streamlit as st
import pandas as pd
import os
import sys
from PIL import Image

# Ensure src is in the path so we can import our modules
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.recommendation import predict_and_recommend

st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="🏦", layout="wide")

# Inject Custom CSS for Demo Polish
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.title("🏦 Bank Customer Classification & Segmentation Dashboard")
st.markdown("""
Welcome to the interactive customer segmentation dashboard. This tool provides an overview of the exploratory data analysis, clustering results, classification model performances, and an AI-driven **Personalized Recommendation Engine**.
***
""")

# Helper to load images safely
def load_image(img_path):
    if os.path.exists(img_path):
        return Image.open(img_path)
    return None

# Tabs with modern icons
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "🔍 Exploratory Data Analysis", 
    "🧩 Clustering Results", 
    "🤖 Classification Models", 
    "💡 Recommendation Engine"
])

with tab1:
    st.header("📊 Dataset Overview")
    data_path = os.path.join(base_dir, 'data', 'raw_data', 'creditcardDataset.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Display high-level metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Customers", value=f"{df.shape[0]:,}")
        with col2:
            st.metric(label="Raw Features (Cols)", value=df.shape[1])
        with col3:
            st.metric(label="Training Features", value=20, delta="3 Engineered Features")
        with col4:
            st.metric(label="Missing Values", value=df.isnull().sum().sum())
            
        st.markdown("### Preview of the Raw Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        with st.expander("View Statistical Summary"):
            st.write(df.describe())
    else:
        st.warning("Dataset not found. Please run the main pipeline first.")

with tab2:
    st.header("🔍 Exploratory Data Analysis")
    
    st.subheader("Feature Distributions")
    with st.expander("Why look at distributions?", expanded=False):
        st.write("Understanding the distribution of each feature helps us identify skewness, outliers, and the need for scaling before feeding the data into machine learning models.")
    img_dist = load_image(os.path.join(base_dir, 'figures', 'feature_distributions.png'))
    if img_dist:
        st.image(img_dist, use_container_width=True)
        
    st.subheader("Feature Boxplots")
    with st.expander("Why look at boxplots?", expanded=False):
        st.write("Boxplots help visually identify outliers and understand the spread, quartiles, and median of the data, which is crucial for anomaly detection.")
    img_box = load_image(os.path.join(base_dir, 'figures', 'boxplot_before_after_preprocessing.png'))
    if img_box:
        st.image(img_box, caption="Feature Boxplots (Before vs After Preprocesing)", use_container_width=True)
        
    st.subheader("Data Amount Before and  Preprocessing Differences")
    # Explicitly show the dataset counts
    col_num1, col_num2, col_num3 = st.columns(3)
    with col_num1:
        st.metric(label="Rows Before Preprocessing", value="8,950")
    with col_num2:
        st.metric(label="Rows After Preprocessing", value="8,950", delta="0 rows dropped (Imputed instead)")
    img_data_amt = load_image(os.path.join(base_dir, 'figures', 'data_amount_preprocessing.png'))
    if img_data_amt:
        st.image(img_data_amt, caption="Data Amount (Before vs After)", use_container_width=True)
    
    st.subheader("Correlation Heatmap")
    img_corr = load_image(os.path.join(base_dir, 'figures', 'correlation_heatmap.png'))
    if img_corr:
        st.image(img_corr, use_container_width=True)

with tab3:
    st.header("🧩 Clustering Results (Autoencoder + K-Means)")
    st.markdown("We used a Deep Learning **Autoencoder** to compress the high-dimensional financial data into a rich latent space, followed by **K-Means clustering**.")
    
    st.subheader("Optimal k Evaluation")
    with st.expander("How did we choose k=4?"):
        st.markdown("To justify our choice of `k=4` clusters, we evaluated the K-Means inertia (Elbow Method) and Silhouette Score for k ranging from 2 to 10. You can observe the 'elbow' forming around 4, balancing cluster cohesion with complexity.")
    
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        img_elbow = load_image(os.path.join(base_dir, 'figures', 'elbow_method.png'))
        if img_elbow:
            st.image(img_elbow, caption="Elbow Method (Inertia)", use_container_width=True)
    with col_eval2:
        img_sil_k = load_image(os.path.join(base_dir, 'figures', 'silhouette_vs_k.png'))
        if img_sil_k:
            st.image(img_sil_k, caption="Silhouette Score vs k", use_container_width=True)
            
    st.subheader("Final Segments (k=4)")
    
    st.info("""
    **Business Cluster Profiles:**
    - 🟢 **Cluster 0: Budget Customers** (Moderate spenders with low debt)
    - 🔵 **Cluster 1: Premium Customers** (High limits, extremely high purchases, large payments)
    - 🔴 **Cluster 2: High-Risk Customers** (Low purchases, but carry massive cash-advance debt)
    - 🟡 **Cluster 3: Regular Customers** (Near $0 balances, but active payers)
    """)
    
    col_pca, col_tsne = st.columns(2)
    with col_pca:
        img_pca = load_image(os.path.join(base_dir, 'outputs', 'customer_segments_pca.png'))
        if img_pca:
            st.image(img_pca, caption="2D PCA of Customer Clusters", use_container_width=True)
    with col_tsne:
        img_tsne = load_image(os.path.join(base_dir, 'outputs', 'customer_segments_tsne.png'))
        if img_tsne:
            st.image(img_tsne, caption="2D t-SNE (Better Separation)", use_container_width=True)
            

with tab4:
    st.header("🤖 Classification Models")
    st.markdown("We trained a **Random Forest** and a **Deep Neural Network** to classify new customers into the clusters.")
    
    st.subheader("ROC Curves")
    with st.expander("Why is the AUC exactly 1.00?"):
        st.info("The labels used for training these classifiers were generated by a K-Means algorithm on the autoencoder's latent space. Because both Random Forest and the DNN are highly expressive models, they can perfectly learn the distinct decision boundaries established by K-Means. This results in near-perfect classification (AUC = 1.00) on the test set.")
    
    col1, col2 = st.columns(2)
    with col1:
        img_roc_rf = load_image(os.path.join(base_dir, 'figures', 'roc_curve_rf.png'))
        if img_roc_rf:
            st.image(img_roc_rf, caption="Random Forest ROC Curve", use_container_width=True)
    with col2:
        img_roc_dnn = load_image(os.path.join(base_dir, 'figures', 'roc_curve_dnn.png'))
        if img_roc_dnn:
            st.image(img_roc_dnn, caption="Deep Neural Network ROC Curve", use_container_width=True)
            
    st.subheader("Confusion Matrices")
    st.markdown("These heatmaps show exactly how many test samples were correctly predicted versus how many were confused with another class.")
    col3, col4 = st.columns(2)
    with col3:
        img_cm_rf = load_image(os.path.join(base_dir, 'figures', 'cm_rf.png'))
        if img_cm_rf:
            st.image(img_cm_rf, caption="Random Forest Confusion Matrix", use_container_width=True)
    with col4:
        img_cm_dnn = load_image(os.path.join(base_dir, 'figures', 'cm_dnn.png'))
        if img_cm_dnn:
            st.image(img_cm_dnn, caption="Deep Neural Network Confusion Matrix", use_container_width=True)

    img_f1 = load_image(os.path.join(base_dir, 'figures', 'model_vs_f1_score.png'))
    if img_f1:
        st.image(img_f1, caption="F1-Score Comparison", use_container_width=True)
            
    st.subheader("Feature Importances")
    with st.expander("Read more about feature importances"):
        st.markdown("Because our primary models are trained on the compressed latent space of an Autoencoder, their direct feature importances are not easily interpretable. To explain which original features drive the customer segmentations, we trained an auxiliary Random Forest model directly on the scaled original features.")
    
    img_feat_imp = load_image(os.path.join(base_dir, 'figures', 'feature_importances.png'))
    if img_feat_imp:
        st.image(img_feat_imp, caption="Feature Importances (via Auxiliary Model)", use_container_width=True)
            
    st.subheader("DNN Training History")
    col5, col6 = st.columns(2)
    with col5:
        img_acc = load_image(os.path.join(base_dir, 'figures', 'epoch_vs_accuracy.png'))
        if img_acc:
            st.image(img_acc, caption="Epoch vs Accuracy", use_container_width=True)
    with col6:
        img_loss = load_image(os.path.join(base_dir, 'figures', 'epoch_vs_loss.png'))
        if img_loss:
            st.image(img_loss, caption="Epoch vs Loss", use_container_width=True)

with tab5:
    st.header("💡 Personalized Recommendation Engine")
    st.markdown("Input customer financial metrics below to classify their profile and receive tailored banking service recommendations in real-time.")
    
    with st.form("recommendation_form"):
        st.subheader("Customer Financial Data")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            balance = st.number_input("Balance ($)", min_value=0.0, value=1500.0)
            purchases = st.number_input("Purchases ($)", min_value=0.0, value=500.0)
            cash_advance = st.number_input("Cash Advance ($)", min_value=0.0, value=0.0)
        with col_b:
            credit_limit = st.number_input("Credit Limit ($)", min_value=0.0, value=5000.0)
            payments = st.number_input("Payments ($)", min_value=0.0, value=1000.0)
            min_payments = st.number_input("Minimum Payments ($)", min_value=0.0, value=200.0)
        with col_c:
            tenure = st.number_input("Tenure (Months)", min_value=1, max_value=12, value=12)
            prc_full_payment = st.number_input("Pct Full Payment (0-1)", min_value=0.0, max_value=1.0, value=0.1)
            
        submitted = st.form_submit_button("Predict Profile & Get Recommendations", use_container_width=True)
        
    if submitted:
        # Build dictionary of inputs
        input_dict = {
            'BALANCE': balance,
            'PURCHASES': purchases,
            'CASH_ADVANCE': cash_advance,
            'CREDIT_LIMIT': credit_limit,
            'PAYMENTS': payments,
            'MINIMUM_PAYMENTS': min_payments,
            'TENURE': tenure,
            'PRC_FULL_PAYMENT': prc_full_payment
        }
        
        with st.spinner("Analyzing customer data through Deep Neural Network..."):
            try:
                cluster_id, recs = predict_and_recommend(input_dict, base_dir)
                
                # Map ID to name
                cluster_names = {
                    0: "Budget Customers",
                    1: "Premium Customers",
                    2: "High-Risk Customers",
                    3: "Regular Customers"
                }
                
                profile_name = cluster_names.get(cluster_id)
                st.success(f"### 🎉 Predicted Profile: {profile_name}")
                st.markdown(f"Based on the provided metrics, this customer belongs to **Cluster {cluster_id}**.")
                
                st.markdown("### 💡 Recommended Services & Actions")
                for r in recs:
                    if "Personalized" in r:
                        st.success(r)
                    elif "Product" in r:
                        st.info(r)
                    elif "Action" in r or "Feature" in r:
                        st.warning(r)
                    else:
                        st.info(r)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}. Please ensure the models have been trained by running `main.py` first.")
