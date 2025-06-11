
from fpdf import FPDF

def generate_pdf_report(label, probability, patient_info, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Gallstone Disease Prediction Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=10)
    for key, value in patient_info.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
    pdf.cell(200, 10, txt=f"Probability: {probability:.2f}%", ln=True)

    pdf.output(output_path)
