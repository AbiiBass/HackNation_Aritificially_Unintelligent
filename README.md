# Women's Hormonal Health Platform — Demo App

A Flask + Bootstrap app for the Hack-Nation Women's Hormonal Health challenge.
CSV files act as the database; PDFs are stored on disk.

Diagnosis predictions come from a real ensemble model (not a placeholder) — see **How the Model Works** below.

---

## Setup

```bash
pip install -r requirements.txt
python3 generate_sample_pdfs.py   # only needed once, already run — regenerates sample PDFs
```

> **Important**
>
> `requirements.txt` intentionally pins:
>
> - `scikit-learn>=1.8.0`
> - `xgboost>=3.3.0`
>
> The trained `.joblib` artifacts inside `model/models/` were saved with these
> versions. Older versions of scikit-learn cannot load the meta-learner and
> will crash during startup.

---

### AI Chatbot Setup (Free Gemini API)

The chatbot uses Google's Gemini API, which has a free tier through Google AI Studio
(no credit card required).

1. Get a free API key:
   https://aistudio.google.com/apikey

2. Copy `.env.example` to `.env` in the project root.

3. Paste your key:

```text
GEMINI_API_KEY=your-actual-key
```

4. Run the app — it loads the key automatically.

If you skip this step, the chatbot page still works — it simply returns a setup reminder instead of an AI-generated response, so the rest of the application remains fully functional.

> **Free tier note**
>
> The app uses `gemini-flash-latest`.
>
> Google has retired `gemini-2.5-flash` and `gemini-2.5-flash-lite` for newly created API keys, so this alias always points to Google's current Flash model.
>
> Free-tier limits may change over time. Check your current quota at:
>
> https://aistudio.google.com

---

```bash
python3 app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## Test Logins

### Patients

Log in with **National ID** and leave **"I am a Doctor"** unchecked.

| National ID | Password | Name |
|-------------|----------|------------------|
| P001 | pass123 | Sarah Johnson |
| P002 | pass123 | Amina Rodriguez |
| P003 | pass123 | Emily Chen |
| P004 | pass123 | Grace Okafor |
| P005 | pass123 | Maria Silva |

### Doctors

Log in with **Employee ID** and check **"I am a Doctor / Researcher"**.

| Employee ID | Password | Name |
|-------------|----------|----------------|
| D101 | doc123 | Dr. Lisa Tran |
| D102 | doc123 | Dr. Robert Kim |
| D103 | doc123 | Dr. Nadia Hassan |

---

## Folder Structure

```text
whh-app/
├── app.py                     # all Flask routes
├── generate_sample_pdfs.py    # one-time script that created the sample lab/screening PDFs
├── requirements.txt
├── .env.example               # copy to .env and add your free Gemini API key
├── model/
│   ├── predict.py             # single entry point: predict(patient_dict) -> ensemble result
│   ├── model_adapter.py       # PDF -> predict() -> doctor-facing {diagnoses, summary, warnings}
│   ├── pdf_to_json.py         # parses uploaded PDFs into a PatientInput dictionary
│   ├── model3.py              # rule-based Rotterdam + TSH criteria scorer
│   ├── meta_learner.py        # trains the 4-way joint (PCOS × thyroid) meta-learner
│   ├── train_model1.py        # generic mixed-sex thyroid model
│   ├── train_model2.py        # women-only thyroid model
│   ├── train_model2b_pcos.py  # women-only PCOS model
│   ├── data_prep.py
│   ├── label_engineering.py
│   ├── split.py
│   ├── pipeline_common.py     # training pipeline utilities
│   ├── robustness_checks.py
│   ├── label_noise_sensitivity.py
│   │                           # validation, ablations, bootstrap CIs and sensitivity tests
│   ├── api.py                  # optional standalone FastAPI wrapper
│   ├── models/                 # committed trained model artifacts (.joblib + metrics)
│   └── tests/                  # pytest suite
├── chatbot/
│   ├── gemini_client.py       # Gemini wrapper + safety system prompt
│   └── safety.py              # keyword-based emergency safety net
├── utils/
│   └── data_helpers.py        # CSV read/write layer ("database")
├── data/
│   ├── users.csv
│   ├── patients.csv
│   ├── questions.csv
│   ├── tmp_uploads/
│   └── records/
│       ├── records.csv
│       └── files/
├── templates/
└── static/
    ├── css/style.css
    └── js/main.js
```

---

# How the Model Works

The prediction pipeline is centered around **`model/predict.py`**, which combines four complementary components to generate a final clinical prediction.

### Model 1 — Generic Thyroid Model

- Trained on a mixed-sex thyroid cohort.
- Requires a complete thyroid laboratory panel:
  - TSH
  - T3
  - T4
  - T4U
  - FTI

---

### Model 2 — Women's Thyroid Model

- Trained only on women's clinical data.
- Predicts thyroid dysfunction from:
  - menstrual cycle history
  - symptoms
  - hormone measurements
  - ultrasound findings

No thyroid laboratory panel is required.

---

### Model 2b — Women's PCOS Model

Predicts PCOS using:

- menstrual history
- symptoms
- reproductive hormones
- ultrasound features

---

### Model 3 — Rule-Based Clinical Scorer

A deterministic (non-trained) model implementing:

- Rotterdam PCOS criteria
- TSH threshold rules for thyroid dysfunction

This provides an interpretable clinical baseline.

---

### Meta-Learner

A logistic regression meta-model combines outputs from:

- Model 2
- Model 2b
- Model 3

to produce a calibrated joint prediction across four possible outcomes:

- Neither condition
- Thyroid dysfunction only
- PCOS only
- Both PCOS and thyroid dysfunction

---

### Intelligent Field Gating

Each model **abstains rather than guessing** if the uploaded document is missing the information it requires.

`model_adapter.py` merges whichever predictions are available into the doctor-facing report shown on the **Study Analysis** page using plain clinical language—without exposing model names or raw statistical outputs.

---

### Running Model Tests

Run the complete model validation suite:

```bash
python3 -m pytest model/tests/
```

The tests verify:

- Rule engine correctness
- Field-gating logic
- Doctor-facing summaries
- Compatibility with committed model artifacts

---

## Notes / Things to Know

- Passwords are stored in plain text inside `users.csv` **for demo purposes only**. Never do this in production.
- File uploads are restricted to `.pdf`.
- Patients can only download their own records.
- Doctors can upload, view, and download records for any patient.
- CSV files are used instead of a database. Data is read and rewritten on every request, which is suitable for demonstrations but not production.
- `.env.example` should **always** contain placeholder values only—never commit a real API key.

---

# How the Chatbot's Safety Design Works

The chatbot uses **two independent safety layers**, preventing a single point of failure.

### 1. System Prompt

Located in:

```
chatbot/gemini_client.py
```

The prompt instructs Gemini to:

- provide general health information only
- never diagnose medical conditions
- refuse requests for medical diagnoses
- encourage users with potentially serious symptoms to seek professional medical care

---

### 2. Keyword Safety Net

Located in:

```
chatbot/safety.py
```

Every user message is scanned server-side before being returned.

If emergency or self-harm language is detected, the application automatically prepends a prominent crisis message directing the user to emergency services or an appropriate helpline.

Because this layer is independent of Gemini, it still works if:

- the API fails
- the model is slow
- the model produces an unsafe response

---

Conversation history is stored in the Flask session (limited to the most recent 12 turns), allowing contextual conversations without requiring a database.

The **Ask a Doctor** form is completely separate from the AI chatbot. Instead of using Gemini, it submits questions directly to the doctor's **Answer Help** queue for cases where human medical review is more appropriate.
