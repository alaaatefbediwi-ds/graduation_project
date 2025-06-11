
# predictor.py

import joblib
import pandas as pd

# Load the full pipeline (preprocessing + model)
model = joblib.load('/content/mediscan_ai/models/gallstone_model.pkl')

def predict_gallstone(lab_results: dict):
    """
    Predict gallstone status based on input lab results.

    Parameters:
        lab_results (dict): Dictionary of feature_name: value

    Returns:
        prediction_label (str), probability (float)
    """
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([lab_results])  # Single row

    # Predict using the full pipeline
    #prediction = model.predict(input_df)[0]
    #probability = model.predict_proba(input_df)[0][0]  # Probability of "gallstone" class

    #label = "Gallstone Detected" if prediction == 0 else "No Gallstone Detected"
    prediction = model.predict(input_df)[0]
    probas = model.predict_proba(input_df)[0]

    if prediction == 0:
        label = "Gallstone Detected"
        probability = probas[0]  # Probability of class 0 (gallstone)
    else:
        label = "No Gallstone Detected"
        probability = probas[1]  # Probability of class 1 (no gallstone)

    return label, round(probability * 100, 2)  # Return percentage
    #return label, probability
