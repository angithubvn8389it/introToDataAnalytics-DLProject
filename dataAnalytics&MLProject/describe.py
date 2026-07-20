import pandas as pd
import os

os.makedirs('outputs', exist_ok=True)

df = pd.read_csv("data/raw_data/creditcardDataset.csv")

# 1. Basic Statistical Description
print("=== Basic Statistics ===")
desc = df.describe()
print(desc)
desc.to_csv('outputs/data_description.csv')
print("[+] Saved basic statistics to outputs/data_description.csv\n")

# 2. Missing Values Info
print("=== Missing Values ===")
missing_vals = df.isnull().sum()
print(missing_vals)
missing_vals.to_csv('outputs/missing_values.csv')
print("[+] Saved missing values to outputs/missing_values.csv\n")

# 3. Data Types Info
print("=== Data Types ===")
dtypes_df = df.dtypes
print(dtypes_df)
dtypes_df.to_csv('outputs/data_types.csv')
print("[+] Saved data types to outputs/data_types.csv")