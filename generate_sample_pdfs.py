"""
Generates sample PDF lab reports / screening documents for the demo.
Run once: python3 generate_sample_pdfs.py
"""
import csv
import random
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

random.seed(42)

LAB_PANELS = {
    "Hormone panel - LH/FSH/Estradiol": [
        ("LH", "6.8", "mIU/mL", "2.0 - 12.0"),
        ("FSH", "7.2", "mIU/mL", "3.0 - 10.0"),
        ("Estradiol", "145", "pg/mL", "30 - 400"),
        ("Progesterone", "0.9", "ng/mL", "0.1 - 25.0"),
    ],
    "Thyroid function panel TSH/T3/T4": [
        ("TSH", "3.8", "mIU/L", "0.4 - 4.0"),
        ("Free T3", "2.9", "pg/mL", "2.3 - 4.2"),
        ("Free T4", "1.1", "ng/dL", "0.8 - 1.8"),
        ("TPO Antibodies", "12", "IU/mL", "< 35"),
    ],
    "Complete hormone and glucose panel": [
        ("Fasting Glucose", "102", "mg/dL", "70 - 99"),
        ("Insulin", "18.4", "uIU/mL", "2.6 - 24.9"),
        ("Testosterone (total)", "68", "ng/dL", "8 - 60"),
        ("DHEA-S", "310", "ug/dL", "65 - 380"),
    ],
    "Menstrual cycle hormone tracking": [
        ("LH", "9.1", "mIU/mL", "2.0 - 12.0"),
        ("FSH", "5.4", "mIU/mL", "3.0 - 10.0"),
        ("Estradiol", "210", "pg/mL", "30 - 400"),
        ("AMH", "3.2", "ng/mL", "1.0 - 4.0"),
    ],
    "Menopause hormone panel": [
        ("FSH", "58", "mIU/mL", "> 40 (post-menopausal)"),
        ("Estradiol", "18", "pg/mL", "< 30 (post-menopausal)"),
        ("LH", "34", "mIU/mL", "> 20 (post-menopausal)"),
    ],
    "PCOS diagnostic hormone panel": [
        ("LH", "14.2", "mIU/mL", "2.0 - 12.0"),
        ("FSH", "4.8", "mIU/mL", "3.0 - 10.0"),
        ("LH:FSH Ratio", "2.9", "ratio", "< 2.0"),
        ("Testosterone (total)", "72", "ng/dL", "8 - 60"),
        ("AMH", "6.1", "ng/mL", "1.0 - 4.0"),
    ],
}

SCREENING_NOTES = {
    "Pelvic ultrasound screening": "Transvaginal ultrasound performed. Ovarian volume within normal limits. No dominant follicles noted. Endometrial thickness 8mm, consistent with cycle phase.",
    "Annual reproductive health screening": "General reproductive health screening completed. Pap smear normal. Breast exam unremarkable. Patient reports mild fatigue and occasional headaches.",
    "Bone density and cardiovascular screening": "DEXA scan shows T-score of -1.4 at lumbar spine (osteopenia range). Blood pressure 128/82. Lipid panel shows mildly elevated LDL. Recommend follow-up in 12 months.",
    "Pelvic ultrasound follow-up": "Follow-up transvaginal ultrasound. Bilateral ovaries show multiple small peripheral follicles (>12 per ovary), consistent with polycystic morphology. Correlate clinically.",
}

styles = getSampleStyleSheet()


def build_lab_pdf(path, hospital, patient_name, national_id, record_date, panel_name, doctor_name):
    doc = SimpleDocTemplate(path, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story = []

    story.append(Paragraph(hospital, styles["Title"]))
    story.append(Paragraph("Laboratory Report", styles["Heading2"]))
    story.append(Spacer(1, 12))

    info = (
        f"<b>Patient Name:</b> {patient_name}<br/>"
        f"<b>National ID:</b> {national_id}<br/>"
        f"<b>Report Date:</b> {record_date}<br/>"
        f"<b>Ordering Physician:</b> {doctor_name}<br/>"
        f"<b>Panel:</b> {panel_name}"
    )
    story.append(Paragraph(info, styles["Normal"]))
    story.append(Spacer(1, 16))

    rows = [["Test", "Result", "Unit", "Reference Range"]]
    for row in LAB_PANELS.get(panel_name, []):
        rows.append(list(row))

    table = Table(rows, colWidths=[2.0 * inch, 1.2 * inch, 1.0 * inch, 2.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f4c5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f6")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "This report is for clinical use only. Please discuss results with your treating physician.",
        styles["Italic"]
    ))

    doc.build(story)


def build_screening_pdf(path, hospital, patient_name, national_id, record_date, record_type_desc, doctor_name):
    doc = SimpleDocTemplate(path, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story = []

    story.append(Paragraph(hospital, styles["Title"]))
    story.append(Paragraph("Screening Report", styles["Heading2"]))
    story.append(Spacer(1, 12))

    info = (
        f"<b>Patient Name:</b> {patient_name}<br/>"
        f"<b>National ID:</b> {national_id}<br/>"
        f"<b>Report Date:</b> {record_date}<br/>"
        f"<b>Attending Physician:</b> {doctor_name}<br/>"
        f"<b>Screening Type:</b> {record_type_desc}"
    )
    story.append(Paragraph(info, styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>Findings:</b>", styles["Heading3"]))
    note = SCREENING_NOTES.get(record_type_desc, "No additional notes recorded.")
    story.append(Paragraph(note, styles["Normal"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "This report is for clinical use only. Please discuss results with your treating physician.",
        styles["Italic"]
    ))

    doc.build(story)


def main():
    doctor_names = {"D101": "Dr. Lisa Tran", "D102": "Dr. Robert Kim", "D103": "Dr. Nadia Hassan"}

    with open("data/records/records.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        out_path = f"data/records/files/{row['filename']}"
        doctor_name = doctor_names.get(row["uploaded_by"], row["uploaded_by"])
        if row["record_type"] == "Lab Report":
            build_lab_pdf(
                out_path, row["hospital"], row["patient_name"], row["national_id"],
                row["record_date"], row["description"], doctor_name,
            )
        else:
            build_screening_pdf(
                out_path, row["hospital"], row["patient_name"], row["national_id"],
                row["record_date"], row["description"], doctor_name,
            )
        print(f"Created {out_path}")


if __name__ == "__main__":
    main()
