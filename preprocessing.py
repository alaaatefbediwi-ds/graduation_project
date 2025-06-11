
# preprocessing.py

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Feature lists
numeric_features = [
    'Age', 'Height', 'Weight', 'Body Mass Index (BMI)', 'Total Body Water (TBW)',
    'Extracellular Water (ECW)', 'Intracellular Water (ICW)',
    'Extracellular Fluid/Total Body Water (ECF/TBW)',
    'Total Body Fat Ratio (TBFR) (%)', 'Lean Mass (LM) (%)',
    'Body Protein Content (Protein) (%)', 'Visceral Fat Rating (VFR)',
    'Bone Mass (BM)', 'Muscle Mass (MM)', 'Obesity (%)',
    'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)',
    'Visceral Muscle Area (VMA) (Kg)', 'Hepatic Fat Accumulation (HFA)',
    'Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)',
    'High Density Lipoprotein (HDL)', 'Triglyceride',
    'Aspartat Aminotransferaz (AST)', 'Alanin Aminotransferaz (ALT)',
    'Alkaline Phosphatase (ALP)', 'Creatinine', 'Glomerular Filtration Rate (GFR)',
    'C-Reactive Protein (CRP)', 'Hemoglobin (HGB)', 'Vitamin D'
]

# Split categorical features based on fill strategy
fill_zero_features = ['Comorbidity', 'Diabetes Mellitus (DM)']
fill_most_frequent_features = ['Gender']
categorical_features = fill_zero_features + fill_most_frequent_features

# Pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Gender or other categorical features filled with most frequent and then one-hot encoded
cat_most_freq_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore')) #, sparse=False))
])

# Comorbidity & DM: filled with zero and then one-hot encoded
cat_zero_fill_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("encoder", OneHotEncoder(handle_unknown='ignore')) #, sparse=False))
])

# Combined preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat_mostfreq", cat_most_freq_transformer, fill_most_frequent_features),
    ("cat_zerofill", cat_zero_fill_transformer, fill_zero_features)
])

#categorical_features = fill_zero_features + fill_most_frequent_features
