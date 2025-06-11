
import re
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image

# ----------------------------
# Constants (put at the top)
# ----------------------------
MODEL_FEATURES = [
    'Age', 'Gender', 'Comorbidity', 'Diabetes Mellitus (DM)', 'Height (cm)', 'Weight (kg)',
    'Body Mass Index (BMI)', 'Total Body Water (TBW)', 'Extracellular Water (ECW)',
    'Intracellular Water (ICW)', 'Extracellular Fluid/Total Body Water (ECF/TBW)',
    'Total Body Fat Ratio (TBFR) (%)', 'Lean Mass (%)', 'Body Protein Content (%)',
    'Visceral Fat Rating (VFR)', 'Bone Mass (BM)', 'Muscle Mass (MM)', 'Obesity (%)',
    'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)', 'Visceral Muscle Area (VMA)',
    'Hepatic Fat Accumulation (HFA)', 'Glucose', 'Total Cholesterol (TC)',
    'Low Density Lipoprotein (LDL)', 'High Density Lipoprotein (HDL)', 'Triglyceride',
    'Aspartat Aminotransferaz (AST)', 'Alanin Aminotransferaz (ALT)',
    'Alkaline Phosphatase (ALP)', 'Creatinine', 'Glomerular Filtration Rate (GFR)',
    'C-Reactive Protein (CRP)', 'Hemoglobin (HGB)', 'Vitamin D'
]

gender_map = {"Male": 1, "Female": 0}
yes_no_map = {"Yes": 1, "No": 0}

# -------------------------------------
# Function to prepare input for model
# -------------------------------------
def prepare_input_for_model(extracted_data: dict):
    gender = gender_map.get(extracted_data.get("Gender", ""), 0)
    comorbidity = yes_no_map.get(extracted_data.get("Comorbidity", ""), 0)
    diabetes = yes_no_map.get(extracted_data.get("Diabetes Mellitus (DM)", ""), 0)

    model_input = {
        'Age': float(extracted_data.get('Age', 0)),
        'Gender': gender,
        'Comorbidity': comorbidity,
        'Diabetes Mellitus (DM)': diabetes,
        'Height (cm)': float(extracted_data.get('Height (cm)', 0)),
        'Weight (kg)': float(extracted_data.get('Weight (kg)', 0)),
        'Body Mass Index (BMI)': float(extracted_data.get('Body Mass Index (BMI)', 0)),
        'Total Body Water (TBW)': float(extracted_data.get('Total Body Water (TBW)', 0)),
        'Extracellular Water (ECW)': float(extracted_data.get('Extracellular Water (ECW)', 0)),
        'Intracellular Water (ICW)': float(extracted_data.get('Intracellular Water (ICW)', 0)),
        'Extracellular Fluid/Total Body Water (ECF/TBW)': float(extracted_data.get('Extracellular Fluid/Total Body Water (ECF/TBW)', 0)),
        'Total Body Fat Ratio (TBFR) (%)': float(extracted_data.get('Total Body Fat Ratio (TBFR) (%)', 0)),
        'Lean Mass (%)': float(extracted_data.get('Lean Mass (%)', 0)),
        'Body Protein Content (%)': float(extracted_data.get('Body Protein Content (%)', 0)),
        'Visceral Fat Rating (VFR)': float(extracted_data.get('Visceral Fat Rating (VFR)', 0)),
        'Bone Mass (BM)': float(extracted_data.get('Bone Mass (BM)', 0)),
        'Muscle Mass (MM)': float(extracted_data.get('Muscle Mass (MM)', 0)),
        'Obesity (%)': float(extracted_data.get('Obesity (%)', 0)),
        'Total Fat Content (TFC)': float(extracted_data.get('Total Fat Content (TFC)', 0)),
        'Visceral Fat Area (VFA)': float(extracted_data.get('Visceral Fat Area (VFA)', 0)),
        'Visceral Muscle Area (VMA)': float(extracted_data.get('Visceral Muscle Area (VMA)', 0)),
        'Hepatic Fat Accumulation (HFA)': float(extracted_data.get('Hepatic Fat Accumulation (HFA)', 0)),
        'Glucose': float(extracted_data.get('Glucose', 0)),
        'Total Cholesterol (TC)': float(extracted_data.get('Total Cholesterol (TC)', 0)),
        'Low Density Lipoprotein (LDL)': float(extracted_data.get('Low Density Lipoprotein (LDL)', 0)),
        'High Density Lipoprotein (HDL)': float(extracted_data.get('High Density Lipoprotein (HDL)', 0)),
        'Triglyceride': float(extracted_data.get('Triglyceride', 0)),
        'Aspartat Aminotransferaz (AST)': float(extracted_data.get('Aspartat Aminotransferaz (AST)', 0)),
        'Alanin Aminotransferaz (ALT)': float(extracted_data.get('Alanin Aminotransferaz (ALT)', 0)),
        'Alkaline Phosphatase (ALP)': float(extracted_data.get('Alkaline Phosphatase (ALP)', 0)),
        'Creatinine': float(extracted_data.get('Creatinine', 0)),
        'Glomerular Filtration Rate (GFR)': float(extracted_data.get('Glomerular Filtration Rate (GFR)', 0)),
        'C-Reactive Protein (CRP)': float(extracted_data.get('C-Reactive Protein (CRP)', 0)),
        'Hemoglobin (HGB)': float(extracted_data.get('Hemoglobin (HGB)', 0)),
        'Vitamin D': float(extracted_data.get('Vitamin D', 0)),
    }

    return pd.DataFrame([model_input])

# --------------------------
# OCR and Extraction Logic
# --------------------------
def pdf_to_text_via_ocr(pdf_path, dpi=300):
    images = convert_from_path(pdf_path, dpi=dpi)
    full_text = ""
    for img in images:
        text = pytesseract.image_to_string(img)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII
        full_text += text + "\n"
    return full_text

def extract_lab_report_data(pdf_path):
    # Step 1: OCR + Cleaning
    text = pdf_to_text_via_ocr(pdf_path)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'(?<=\w): ([^:\n]+?)(?= \w+:)', r': \1\n', text)

    # Step 2: Regex patterns (truncated for brevity; insert full dictionary)
    patterns = {
        'Hospital Name': r'Hospital\s*:\s*(.+)',
        'Lab Name': r'Lab Name\s*:\s*(.+)',
        'Patient Name': r'Name\s*:\s*(.+)',
        'Gender': r'Gender\s*:\s*(Male|Female)',
        'Age': r'Age\s*:\s*(\d+)',
        'Lab Date': r'Date\s*:\s*(\w+ \d{2}, \d{4})',
        'Comorbidity': r'Comorbidity\s*:\s*(\w+)',
        'Diabetes Mellitus (DM)': r'Diabetes Mellitus \(DM\)\s*:\s*(\w+)',
        'Body Mass Index (BMI)': r'Body Mass Index \(BMI\)\s*:\s*([\d.]+)',
        'Height (cm)': r'Height\s*:\s*(\d+)',
        'Weight (kg)': r'Weight\s*:\s*([\d.]+)',
        'Total Body Water (TBW)': r'Total Body Water \(TBW\)\s*:\s*([\d.]+)',
        'Extracellular Water (ECW)': r'Extracellular Water \(ECW\)\s*:\s*([\d.]+)',
        'Intracellular Water (ICW)': r'Intracellular Water \(ICW\)\s*:\s*([\d.]+)',
        'Extracellular Fluid/Total Body Water (ECF/TBW)': r'Extracellular Fluid/Total Body Water \(ECF/TBW\)\s*:\s*([\d.]+)',
        'Total Body Fat Ratio (TBFR) (%)': r'Total Body Fat Ratio \(TBFR\)\s*:\s*([\d.]+)%',
        'Lean Mass (%)': r'Lean Mass \(LM\)\s*:\s*([\d.]+)%',
        'Body Protein Content (%)': r'Body Protein Content\s*:\s*([\d.]+)%',
        'Visceral Fat Rating (VFR)': r'Visceral Fat Rating \(VFR\)\s*:\s*([\d.]+)',
        'Bone Mass (BM)': r'Bone Mass\s*:\s*([\d.]+)\s*kg',
        'Muscle Mass (MM)': r'Muscle Mass \(MM\)\s*:\s*([\d.]+)\s*kg',
        'Obesity (%)': r'Obesity\s*:\s*([\d.]+)%',
        'Total Fat Content (TFC)': r'Total Fat Content \(TFC\)\s*:\s*([\d.]+)',
        'Visceral Fat Area (VFA)': r'Visceral Fat Area \(VFA\)\s*:\s*([\d.]+)',
        'Visceral Muscle Area (VMA)': r'Visceral Muscle Area \(VMA\)\s*:\s*([\d.]+)',
        'Hepatic Fat Accumulation (HFA)': r'Hepatic Fat Accumulation \(HFA\)\s*:\s*([\d.]+)',
        'Glucose': r'Glucose\s*:\s*([\d.]+)',
        'Total Cholesterol (TC)': r'Total Cholesterol \(TC\)\s*:\s*([\d.]+)',
        'Low Density Lipoprotein (LDL)': r'Low Density Lipoprotein \(LDL\)\s*:\s*([\d.]+)',
        'High Density Lipoprotein (HDL)': r'High Density Lipoprotein \(HDL\)\s*:\s*([\d.]+)',
        'Triglyceride': r'Triglyceride\s*:\s*([\d.]+)',
        'Aspartat Aminotransferaz (AST)': r'AST\s*:\s*([\d.]+)',
        'Alanin Aminotransferaz (ALT)': r'ALT\s*:\s*([\d.]+)',
        'Alkaline Phosphatase (ALP)': r'ALP\s*:\s*([\d.]+)',
        'Creatinine': r'Creatinine\s*:\s*([\d.]+)',
        'Glomerular Filtration Rate (GFR)': r'GFR\s*:\s*([\d.]+)',
        'C-Reactive Protein (CRP)': r'C-Reactive Protein \(CRP\)\s*:\s*([\d.]+)',
        'Hemoglobin (HGB)': r'Hemoglobin \(HGB\)\s*:\s*([\d.]+)',
        'Vitamin D': r'Vitamin D\s*:\s*([\d.]+)',
    }

    extracted_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            extracted_data[key] = match.group(1).strip()

    return extracted_data
