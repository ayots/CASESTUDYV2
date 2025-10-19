# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

# Preview the raw dataset
print("Initial preview of dataset:")
print(df.head())

# Check dataset shape
print("Dataset shape:", df.shape)

# Display column names
print("Column names:")
print(df.columns.tolist())

# ------------------ Remove Duplicates ------------------

# Check for duplicates
print("Number of duplicate rows:", df.duplicated().sum())

# Drop duplicate rows
df = df.drop_duplicates()
print("Dataset shape after removing duplicates:", df.shape)

# ------------------ Visualise Missing Values ------------------

# Bar plot of missing values per column
missing_counts = df.isnull().sum()
plt.figure(figsize=(12, 6))
missing_counts.plot(kind='bar', color='salmon')
plt.title('Missing Values per Column')
plt.ylabel('Count of Missing Values')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Heatmap of missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.show()

# ------------------ Drop Rows with Missing Target ------------------

df = df[df['num'].notnull()].copy()

# Create binary target column: 0 = no disease, 1 = disease
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# ------------------ Impute Specific Columns ------------------

# Fill missing 'fbs' (fasting blood sugar) with 'FALSE' (most common and medically typical)
if 'fbs' in df.columns:
    df['fbs'] = df['fbs'].fillna('FALSE')

# Fill missing 'thal' with 'normal' (most frequent and clinically neutral)
if 'thal' in df.columns:
    df['thal'] = df['thal'].fillna('normal')

# ------------------ General Cleaning ------------------

# Identify categorical columns (including bools)
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

# Convert all categorical values to strings to avoid mixed types
for col in categorical_cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].fillna(df[col].mode()[0])  # Fill remaining missing with mode

# Fill missing numerical values with median
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# ------------------ Encode Categorical Variables ------------------

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ------------------ Normalise Numerical Features ------------------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# ------------------ Final Preview ------------------

print("Preview of cleaned dataset:")
print(df.head())

# ------------------ Export Cleaned Dataset ------------------

df.to_csv('heart_disease_cleaned.csv', index=False)
print("Cleaned dataset exported as 'heart_disease_cleaned.csv'")
