# 🏦 Bank Customer Classification & Segmentation

![Python](https://img.shields.io/badge/Python-3.1%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)

## 📖 Introduction
Understanding customer behavior is critical for modern banking and financial services. This project implements an end-to-end Machine Learning pipeline designed to automatically segment credit card customers into distinct business profiles (e.g., Premium, High-Risk, Budget, Regular) based on their transaction history and balances.

To achieve robust segmentation on high-dimensional financial data, this project combines **Deep Learning Autoencoders** for feature compression with **K-Means Clustering**. Once the customer profiles are discovered, we train a **Random Forest** and a **Deep Neural Network (DNN)** classifier to instantly predict the segment of any new incoming customer. 

Finally, all models, insights, and a personalized recommendation engine are served through an interactive **Streamlit Dashboard**.

## ✨ Features
- **Exploratory Data Analysis (EDA):** Automated generation of feature distributions, boxplots, and correlation heatmaps.
- **Deep Latent Clustering:** Utilizes a custom Autoencoder to compress sparse financial features before applying K-Means clustering.
- **Dynamic Profiling:** Automatically assigns semantic labels to clusters based on their statistical profiles (e.g., Highest Purchases = "Premium").
- **Classification:** Trains Random Forest and DNN models to predict cluster boundaries with extremely high F1/ROC-AUC scores.
- **Interactive Dashboard:** A highly polished web interface built with Streamlit for exploring the data and testing the recommendation engine.

---

## 🚀 Step-by-Step Instructions

Follow these steps to run the project locally on your machine.

### Step 1: Clone the Repository
Open your terminal and clone this repository to your local machine:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Step 2: Create a Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to isolate the project's dependencies.
```bash
# For Windows:
python -m venv .venv
.\.venv\Scripts\activate

# For Mac/Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
Install all the required Python libraries using `pip`:
```bash
pip install -r requirements.txt
```
> *Note: Ensure your raw dataset is placed at `data/raw_data/creditcardDataset.csv`.*

### Step 4: Run the Machine Learning Pipeline
Execute the main script. This will automatically run data preprocessing, train the Autoencoder, execute K-Means clustering, train the Random Forest & DNN classifiers, and save all the resulting models and metrics locally.
```bash
python main.py
```

### Step 5: Launch the Interactive Dashboard
Once the pipeline finishes and the models are saved, you can spin up the Streamlit UI to visualize your results!
```bash
streamlit run app.py
```
This will open the interactive dashboard in your default web browser (typically at `http://localhost:8501`).

---

## 📂 Project Structure

```text
├── app.py                     # Streamlit dashboard application
├── main.py                    # Main orchestrator pipeline for training models
├── profile_clusters.py        # Script for profiling cluster statistics
├── evaluate.py                # Script for evaluating model metrics
├── requirements.txt           # Project dependencies
├── src/                       # Core source code modules
│   ├── clustering.py          # K-Means and Autoencoder logic
│   ├── models.py              # Random Forest & DNN architectures
│   ├── preprocessing.py       # Data cleaning, scaling, and imputation
│   └── recommendation.py      # Personalized banking recommendation logic
├── tests/                     # Unit tests for preprocessing & clustering
├── data/                      # Directory for raw and processed datasets
├── figures/                   # Auto-generated plots (EDA, ROC curves, etc.)
└── outputs/                   # Saved models (.h5, .pkl), histories, and CSV metrics
```

## 📊 Business Outcomes
By leveraging this pipeline, the bank successfully categorizes users into:
- **Premium Customers**: High limit, high purchase volume (Recommend Premium Travel Cards).
- **High-Risk Customers**: Extremely high cash-advance debt (Recommend Balance Transfer/Refinancing).
- **Budget Customers**: Very low spending activity (Recommend No-Fee starter cards).
- **Regular Customers**: Transactors who pay off their balances quickly (Recommend Cashback cards).
