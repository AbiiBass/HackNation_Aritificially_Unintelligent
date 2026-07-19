"""
pdf_to_json.py

Extracts an uploaded lab/screening PDF into the JSON shape api.py's
PatientInput / predict.py's predict() expect.

USAGE
    python3 pdf_to_json.py sample_patient_workup.pdf
    python3 pdf_to_json.py sample_patient_workup.pdf --out patient_123.json

Fully offline, no network calls: pdfplumber extracts text/tables, then
regexes match the table rows and checklist lines produced by
generate_sample_pdfs.py / generate_sample_lab_pdf.py (plus common
synonyms) onto the PatientInput schema.

Unmatched fields stay null over the template; predict.py's _has_keys()
treats null as "abstain", so a partial extraction is always safe to use.

Limitation: only recognizes label wording it has a pattern for. For other
formats, add a pattern to NUMERIC_TEST_PATTERNS / YES_NO_FIELD_PATTERNS.

Also prints which of the 32 fields came back non-null, grouped by which
downstream model needs them.
"""
import argparse
import json
import re

import pdfplumber

# Canonical schema; keep in sync with api.py's PatientInput.
TEMPLATE = {
    "age": None,
    "bmi": None,
    "cycle_regularity": None,
    "cycle_length_days": None,
    "weight_gain": None,
    "hirsutism": None,
    "skin_darkening": None,
    "hair_loss": None,
    "acne": None,
    "fast_food": None,
    "regular_exercise": None,
    "bp_systolic": None,
    "bp_diastolic": None,
    "fsh": None,
    "lh": None,
    "fsh_lh_ratio": None,
    "amh": None,
    "prl": None,
    "vit_d3": None,
    "progesterone": None,
    "random_blood_sugar": None,
    "follicle_count_left": None,
    "follicle_count_right": None,
    "avg_follicle_size_left": None,
    "avg_follicle_size_right": None,
    "endometrium_mm": None,
    "tsh": None,
    "sex": None,
    "pregnant": None,
    "sick": None,
    "t3": None,
    "t4": None,
    "t4_uptake": None,
    "free_thyroxine_index": None,
}

FIELDS_BY_MODEL = {
    # Model 2 excludes tsh (see predict.py's FIELDS_MODEL2); Model 2b includes it.
    "Model 2 (women-only, PCOS-cohort, thyroid_dysfunction)": [
        "age", "bmi", "cycle_regularity", "cycle_length_days", "weight_gain",
        "hirsutism", "skin_darkening", "hair_loss", "acne", "fast_food",
        "regular_exercise", "bp_systolic", "bp_diastolic", "fsh", "lh",
        "fsh_lh_ratio", "amh", "prl", "vit_d3", "progesterone",
        "random_blood_sugar", "follicle_count_left", "follicle_count_right",
        "avg_follicle_size_left", "avg_follicle_size_right", "endometrium_mm",
    ],
    "Model 2b (women-only, PCOS-cohort, pcos_diagnosis)": [
        "age", "bmi", "cycle_regularity", "cycle_length_days", "weight_gain",
        "hirsutism", "skin_darkening", "hair_loss", "acne", "fast_food",
        "regular_exercise", "bp_systolic", "bp_diastolic", "fsh", "lh",
        "fsh_lh_ratio", "amh", "prl", "vit_d3", "progesterone",
        "random_blood_sugar", "follicle_count_left", "follicle_count_right",
        "avg_follicle_size_left", "avg_follicle_size_right", "endometrium_mm", "tsh",
    ],
    "Model 3 (Rotterdam clinical criteria)": [
        "cycle_regularity", "hirsutism", "lh", "fsh",
        "follicle_count_left", "follicle_count_right", "tsh",
    ],
    "Model 1 (full clinical thyroid panel)": [
        "age", "sex", "pregnant", "sick", "tsh", "t3", "t4",
        "t4_uptake", "free_thyroxine_index",
    ],
}

# ---------------------------------------------------------------------------
# Step 1: get text + tables out of the PDF
# ---------------------------------------------------------------------------
def extract_pdf_content(pdf_path):
    text_parts = []
    table_rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            for table in page.extract_tables():
                table_rows.extend(table)
    return "\n".join(text_parts), table_rows


# ---------------------------------------------------------------------------
# Step 2: regex / table-based extraction (no API, no network)
# ---------------------------------------------------------------------------
NUMERIC_TEST_PATTERNS = {
    "fsh": r"\bFSH\b(?!.*ratio)",
    "lh": r"^\s*LH\s*$",
    "fsh_lh_ratio": r"LH[:\s]*FSH\s*Ratio",
    "amh": r"\bAMH\b",
    "prl": r"Prolactin|\bPRL\b",
    "vit_d3": r"Vitamin\s*D3?",
    "progesterone": r"Progesterone",
    "random_blood_sugar": r"(Random\s*)?Blood\s*Sugar|Fasting\s*Glucose",
    "tsh": r"\bTSH\b",
    "t3": r"Free\s*T3|\bT3\b",
    "t4": r"Free\s*T4|\bT4\b(?!\s*Uptake)",
    "t4_uptake": r"T4\s*Uptake",
    "free_thyroxine_index": r"Free\s*Thyroxine\s*Index|\bFTI\b",
}

YES_NO_FIELD_PATTERNS = {
    "weight_gain": r"weight\s*gain",
    "hirsutism": r"hirsutism",
    "skin_darkening": r"skin\s*darkening",
    "hair_loss": r"hair\s*(thinning|loss)",
    "acne": r"\bacne\b",
    "fast_food": r"fast\s*food",
    "regular_exercise": r"regular\s*exercise",
    "pregnant": r"currently\s*pregnant",
    "sick": r"acutely\s*ill|currently\s*sick",
}


def _to_number(s):
    if s is None:
        return None
    m = re.search(r"-?\d+\.?\d*", s.replace(",", ""))
    return float(m.group()) if m else None


def extract_with_regex(full_text, table_rows):
    result = dict(TEMPLATE)

    # --- numeric lab values, from table rows: [Test, Result, Unit, Range] ---
    for row in table_rows:
        if not row or len(row) < 2 or row[0] is None:
            continue
        label, value = row[0].strip(), row[1]
        for field, pattern in NUMERIC_TEST_PATTERNS.items():
            if result[field] is None and re.search(pattern, label, re.IGNORECASE):
                num = _to_number(value)
                if num is not None:
                    result[field] = num
                break

    # --- Yes/No checklist items, from table rows: [Item, Response] ---
    for row in table_rows:
        if not row or len(row) < 2 or row[0] is None or row[1] is None:
            continue
        label, value = row[0].strip(), row[1].strip()
        for field, pattern in YES_NO_FIELD_PATTERNS.items():
            if result[field] is None and re.search(pattern, label, re.IGNORECASE):
                if re.search(r"\byes\b", value, re.IGNORECASE):
                    result[field] = 1
                elif re.search(r"\bno\b", value, re.IGNORECASE):
                    result[field] = 0
                break

    # --- cycle regularity / length (checklist or free text) ---
    m = re.search(r"cycle\s*regularity\D*?\b(irregular|regular)\b", full_text, re.IGNORECASE)
    if m:
        result["cycle_regularity"] = 4 if m.group(1).lower() == "irregular" else 2
    m = re.search(r"cycle\s*length\D*?(\d+(\.\d+)?)\s*days", full_text, re.IGNORECASE)
    if m:
        result["cycle_length_days"] = float(m.group(1))

    # --- pregnant / sick (often stated in free text header, not a table) ---
    m = re.search(r"Currently\s*Pregnant:?\s*(Yes|No)", full_text, re.IGNORECASE)
    if m:
        result["pregnant"] = 1 if m.group(1).lower() == "yes" else 0
    m = re.search(r"(Currently\s*)?Acutely\s*Ill:?\s*(Yes|No)", full_text, re.IGNORECASE)
    if m:
        result["sick"] = 1 if m.group(2).lower() == "yes" else 0

    # --- age / sex ---
    m = re.search(r"\bAge:?\s*(\d+)", full_text, re.IGNORECASE)
    if m:
        result["age"] = float(m.group(1))
    m = re.search(r"\bSex:?\s*(Male|Female)", full_text, re.IGNORECASE)
    if m:
        result["sex"] = m.group(1).lower()

    # --- BMI / blood pressure ---
    m = re.search(r"\bBMI\D*?(\d+(\.\d+)?)", full_text, re.IGNORECASE)
    if m:
        result["bmi"] = float(m.group(1))
    m = re.search(r"Blood\s*Pressure\D*?(\d+)\s*/\s*(\d+)", full_text, re.IGNORECASE)
    if m:
        result["bp_systolic"] = float(m.group(1))
        result["bp_diastolic"] = float(m.group(2))

    # --- ultrasound narrative fields ---
    m = re.search(r"Left ovary:.*?follicle count\D*?(\d+).*?follicle size\D*?(\d+(\.\d+)?)\s*mm",
                  full_text, re.IGNORECASE | re.DOTALL)
    if m:
        result["follicle_count_left"] = float(m.group(1))
        result["avg_follicle_size_left"] = float(m.group(2))
    m = re.search(r"Right ovary:.*?follicle count\D*?(\d+).*?follicle size\D*?(\d+(\.\d+)?)\s*mm",
                  full_text, re.IGNORECASE | re.DOTALL)
    if m:
        result["follicle_count_right"] = float(m.group(1))
        result["avg_follicle_size_right"] = float(m.group(2))
    m = re.search(r"Endometrial\s*thickness\D*?(\d+(\.\d+)?)\s*mm", full_text, re.IGNORECASE)
    if m:
        result["endometrium_mm"] = float(m.group(1))

    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def pdf_to_patient_json(pdf_path):
    full_text, table_rows = extract_pdf_content(pdf_path)
    extracted = extract_with_regex(full_text, table_rows)

    result = dict(TEMPLATE)
    result.update({k: v for k, v in extracted.items() if k in TEMPLATE})
    return result


def summarize_missing(patient_json):
    print("\nField availability by downstream model:")
    for model_name, fields in FIELDS_BY_MODEL.items():
        missing = [f for f in fields if patient_json.get(f) is None]
        status = "READY" if not missing else f"MISSING {len(missing)}/{len(fields)}"
        print(f"  [{status}] {model_name}")
        if missing:
            print(f"           needs: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description="Extract PatientInput JSON from a lab/screening PDF.")
    parser.add_argument("pdf_path", help="Path to the uploaded PDF")
    parser.add_argument("--out", help="Where to write the JSON (default: stdout only)")
    args = parser.parse_args()

    patient_json = pdf_to_patient_json(args.pdf_path)

    print(json.dumps(patient_json, indent=2))
    summarize_missing(patient_json)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(patient_json, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
