# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv('heart_disease_cleaned.csv')

# Split features and target
X = df.drop('target', axis=1)
y = df['target'].astype(int)

# Train-test split (80/20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Machine Learning Models ------------------

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

ml_models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# ------------------ Deep Learning Model ------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Build simple feedforward neural network
dl_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

dl_model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Train deep learning model
dl_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# ------------------ Evaluation ------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

results = []

# Evaluate ML models
for name, model in ml_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

# Evaluate Deep Learning model
y_dl_prob = dl_model.predict(X_test).flatten()
y_dl_pred = (y_dl_prob > 0.5).astype(int)

results.append({
    'Model': 'Deep Learning (Keras)',
    'Accuracy': accuracy_score(y_test, y_dl_pred),
    'Precision': precision_score(y_test, y_dl_pred),
    'Recall': recall_score(y_test, y_dl_pred),
    'F1-score': f1_score(y_test, y_dl_pred),
    'ROC-AUC': roc_auc_score(y_test, y_dl_prob)
})

# ------------------ Tabular Results ------------------

results_df = pd.DataFrame(results)
print("Model Evaluation Results:")
print(results_df)

# ------------------ Bar Chart Comparison ------------------

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):
    plt.bar([x + i*0.15 for x in range(len(results_df))],
            results_df[metric], width=0.15, label=metric)

plt.xticks([x + 0.3 for x in range(len(results_df))], results_df['Model'], rotation=45)
plt.ylabel('Score')
plt.title('Comparison of Model Performance Metrics')
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison_all.png')
plt.show()

# ------------------ Sample Predictions ------------------

sample = X_test.iloc[:5]
print("\nSample Predictions:")

for name, model in ml_models.items():
    preds = model.predict(sample)
    print(f"{name} predictions: {preds.tolist()}")

dl_preds = (dl_model.predict(sample) > 0.5).astype(int).flatten()
print(f"Deep Learning predictions: {dl_preds.tolist()}")
