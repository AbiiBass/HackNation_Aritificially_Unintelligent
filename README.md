# Women's Hormonal Health Platform — Demo App

A rough Flask + Bootstrap build for the Hack-Nation Women's Hormonal Health challenge.
CSV files act as the database; PDFs are stored on disk.

## Setup

```bash
pip install -r requirements.txt
python3 generate_sample_pdfs.py   # only needed once, already run — regenerates sample PDFs
```

### AI Chatbot setup (free Gemini API)

The chatbot uses Google's Gemini API, which has a free tier through Google AI Studio
(no credit card required).

1. Get a free API key: https://aistudio.google.com/apikey
2. Copy `.env.example` to `.env` in the project root
3. Paste your key in: `GEMINI_API_KEY=your-actual-key`
4. Run the app — it loads the key automatically

If you skip this step, the chatbot page still works — it just replies with a
setup reminder instead of a real answer, so the rest of the app is unaffected.

**Free tier note:** the app uses `gemini-2.5-flash`, which is on Google's free
tier (roughly 10 requests/min, 250/day as of mid-2026 — check your live limits
at https://aistudio.google.com). If you hit quota errors, switch `MODEL_NAME`
in `chatbot/gemini_client.py` to `"gemini-2.5-flash-lite"` for a higher-throughput
free option.

```bash
python3 app.py
```

Open http://127.0.0.1:5000

## Test Logins

**Patients** (log in with National ID, leave "I am a Doctor" unchecked):

| National ID | Password | Name              |
|-------------|----------|-------------------|
| P001        | pass123  | Sarah Johnson     |
| P002        | pass123  | Amina Rodriguez   |
| P003        | pass123  | Emily Chen        |
| P004        | pass123  | Grace Okafor      |
| P005        | pass123  | Maria Silva       |

**Doctors** (log in with Employee ID, check "I am a Doctor / Researcher"):

| Employee ID | Password | Name              |
|-------------|----------|-------------------|
| D101        | doc123   | Dr. Lisa Tran     |
| D102        | doc123   | Dr. Robert Kim    |
| D103        | doc123   | Dr. Nadia Hassan  |

## Folder Structure

```
whh-app/
├── app.py                     # all Flask routes
├── generate_sample_pdfs.py    # one-time script that created the sample lab/screening PDFs
├── requirements.txt
├── .env.example                # copy to .env and add your free Gemini API key
├── model/
│   └── predict.py             # ⭐ MEMBER 1 replaces this with the real model
├── chatbot/
│   ├── gemini_client.py       # wraps the free Gemini API + system prompt (health tips, no diagnosis)
│   └── safety.py              # keyword-based safety net for emergency / self-harm messages
├── utils/
│   └── data_helpers.py        # CSV read/write layer (the "database")
├── data/
│   ├── users.csv              # login credentials + role
│   ├── patients.csv           # patient demographics
│   ├── questions.csv          # patient questions -> doctor Answer Help queue
│   ├── tmp_uploads/           # scratch space for Study Analysis uploads (auto-cleared)
│   └── records/
│       ├── records.csv        # metadata for every health document
│       └── files/             # the actual PDF files (sample + uploaded)
├── templates/                 # Jinja2 HTML templates
└── static/
    ├── css/style.css
    └── js/main.js
```

## Handoff point for Member 1 (the model)

Everything the app needs from the model lives in **`model/predict.py`**. It currently
returns random placeholder values. Replace the body of `predict(text)` with the real
model call, but keep the return shape exactly the same:

```python
{
    "diagnoses": [
        {"condition": "PCOS", "confidence": 0.72},
        {"condition": "Thyroid Dysfunction", "confidence": 0.15},
        {"condition": "Unknown", "confidence": 0.13}
    ],
    "summary": "Explanation string justifying the confidence values."
}
```

`extract_text_from_pdf(pdf_path)` already handles converting the doctor's uploaded PDF
to text using `pdfplumber` — pass that text straight into your model.

## Notes / things to know

- Passwords are stored in plain text in `users.csv` for demo speed only — never do this in production.
- File uploads are restricted to `.pdf`.
- Patients can only download their own records; doctors can view/download/upload for any patient.
- No real database — CSVs are read and rewritten on every request. Fine for a demo, not for concurrent production use.

### How the chatbot's safety design works

Two independent layers, so a single point of failure can't slip through:

1. **System prompt** (`chatbot/gemini_client.py`) — instructs Gemini to give only
   general health tips, never diagnose (even under pressure), and clearly tell
   the user to see a doctor for anything that sounds serious or urgent.
2. **Keyword safety net** (`chatbot/safety.py`) — a small, fast backup check that
   runs on every message server-side, independent of the model. If it detects
   clear emergency or self-harm language, it prepends a strong, unmissable
   safety message (crisis line / emergency services) — guaranteed, even if the
   model response is slow, errors out, or doesn't handle it perfectly.

Conversation history is stored in the Flask session (capped at the last 12
turns) so context carries across messages without needing a real database.

The "Ask a Doctor" form on the chatbot page is separate from the AI chat —
it feeds directly into the doctor's **Answer Help** queue for cases the
chatbot can't (or shouldn't) handle.
