
# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

from preprocessing import preprocessor, numeric_features, categorical_features

# Load data
data = pd.read_excel('/content/Gallstone.xlsx')

# Reverse the label so that:
# 1 → Has Gallstone, 0 → No Gallstone (more intuitive)
#data['Gallstone Status'] = data['Gallstone Status'].map({0: 1, 1: 0})

# Features and target
X = data[numeric_features + categorical_features]
y = data['Gallstone Status']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Full pipeline with model
full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", CatBoostClassifier(verbose=0, random_state=42))
])

# Train
full_pipeline.fit(X_train, y_train)

# Save the pipeline
os.makedirs('/content/mediscan_ai/models', exist_ok=True)
joblib.dump(full_pipeline, '/content/mediscan_ai/models/gallstone_model.pkl')

print("Model pipeline saved successfully.")
