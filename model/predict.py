"""
STUB MODEL — to be replaced by Member 1's real trained model.

Contract (do not change without telling Member 2 / updating the frontend):

    extract_text_from_pdf(pdf_path) -> str
    predict(text: str) -> dict:
        {
            "diagnoses": [
                {"condition": "PCOS", "confidence": 0.72},
                {"condition": "Thyroid Dysfunction", "confidence": 0.15},
                {"condition": "Unknown", "confidence": 0.13}
            ],
            "summary": "Explanation string justifying the confidence values."
        }

Member 1: replace the body of predict() with the real model call.
Keep the function signature and the return shape identical so the
Study Analysis page keeps working without any frontend changes.
"""
import random

CONDITIONS = ["PCOS", "Thyroid Dysfunction", "Unknown"]


def extract_text_from_pdf(pdf_path):
    """Extracts raw text from an uploaded PDF using pdfplumber."""
    try:
        import pdfplumber
        text_chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)
        return "\n".join(text_chunks)
    except Exception as e:
        return f"[Could not extract text: {e}]"


def predict(text):
    """
    FAKE prediction logic — placeholder only.
    Generates plausible-looking but random confidence values so the
    UI/table can be built and tested before the real model is ready.
    """
    raw = [random.uniform(0.1, 1.0) for _ in CONDITIONS]
    total = sum(raw)
    normalized = [round(v / total, 2) for v in raw]

    # make sure rounding still sums close to 1.0
    diagnoses = [
        {"condition": cond, "confidence": conf}
        for cond, conf in zip(CONDITIONS, normalized)
    ]
    diagnoses.sort(key=lambda d: d["confidence"], reverse=True)

    top = diagnoses[0]
    summary = (
        f"[PLACEHOLDER MODEL OUTPUT] Based on the uploaded document, "
        f"the strongest signal points toward '{top['condition']}' "
        f"with an estimated confidence of {top['confidence'] * 100:.0f}%. "
        f"This is a randomly generated placeholder response — replace "
        f"model/predict.py with the trained model to produce real, "
        f"evidence-based confidence scores and justification text. "
        f"Extracted document text length: {len(text)} characters."
    )

    return {
        "diagnoses": diagnoses,
        "summary": summary,
    }
