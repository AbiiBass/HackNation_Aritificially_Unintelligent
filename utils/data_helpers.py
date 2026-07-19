"""
Simple CSV-backed data access layer.
Acts as the "database" for this demo app. Every function reads/writes
directly to the CSV files under /data so no real DB setup is needed.
"""
import csv
import os
import uuid
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

USERS_CSV = os.path.join(DATA_DIR, "users.csv")
PATIENTS_CSV = os.path.join(DATA_DIR, "patients.csv")
RECORDS_CSV = os.path.join(DATA_DIR, "records", "records.csv")
RECORDS_FILES_DIR = os.path.join(DATA_DIR, "records", "files")
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions.csv")


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path, fieldnames, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------- USERS ----------

def get_user(username):
    for row in _read_csv(USERS_CSV):
        if row["username"] == username:
            return row
    return None


def verify_login(username, password, expected_role):
    user = get_user(username)
    if user and user["password"] == password and user["role"] == expected_role:
        return user
    return None


# ---------- PATIENTS ----------

def get_patient(national_id):
    for row in _read_csv(PATIENTS_CSV):
        if row["national_id"] == national_id:
            return row
    return None


# ---------- RECORDS ----------

def get_records_for_patient(national_id):
    return [r for r in _read_csv(RECORDS_CSV) if r["national_id"] == national_id]


def get_record_file_path(filename):
    return os.path.join(RECORDS_FILES_DIR, filename)


def add_record(national_id, patient_name, hospital, record_type, record_date,
                filename, uploaded_by, description):
    fieldnames = ["record_id", "national_id", "patient_name", "hospital",
                  "record_type", "record_date", "filename", "uploaded_by", "description"]
    record_id = f"R{uuid.uuid4().hex[:6].upper()}"
    row = {
        "record_id": record_id,
        "national_id": national_id,
        "patient_name": patient_name,
        "hospital": hospital,
        "record_type": record_type,
        "record_date": record_date,
        "filename": filename,
        "uploaded_by": uploaded_by,
        "description": description,
    }
    _append_csv(RECORDS_CSV, fieldnames, row)
    return record_id


# ---------- QUESTIONS (Ask a Doctor / Answer Help) ----------

def get_all_questions():
    rows = _read_csv(QUESTIONS_CSV)
    # Pending first, most recent first
    rows.sort(key=lambda r: (r["status"] != "Pending", r["date_submitted"]), reverse=False)
    return rows


def get_questions_for_patient(national_id):
    return [r for r in _read_csv(QUESTIONS_CSV) if r["national_id"] == national_id]


def add_question(national_id, patient_name, question):
    fieldnames = ["question_id", "national_id", "patient_name", "question",
                  "date_submitted", "status", "answer", "answered_by", "date_answered"]
    question_id = f"Q{uuid.uuid4().hex[:6].upper()}"
    row = {
        "question_id": question_id,
        "national_id": national_id,
        "patient_name": patient_name,
        "question": question,
        "date_submitted": datetime.now().strftime("%Y-%m-%d"),
        "status": "Pending",
        "answer": "",
        "answered_by": "",
        "date_answered": "",
    }
    _append_csv(QUESTIONS_CSV, fieldnames, row)
    return question_id


def answer_question(question_id, answer_text, answered_by):
    rows = _read_csv(QUESTIONS_CSV)
    fieldnames = ["question_id", "national_id", "patient_name", "question",
                  "date_submitted", "status", "answer", "answered_by", "date_answered"]
    for row in rows:
        if row["question_id"] == question_id:
            row["status"] = "Answered"
            row["answer"] = answer_text
            row["answered_by"] = answered_by
            row["date_answered"] = datetime.now().strftime("%Y-%m-%d")
    _write_csv(QUESTIONS_CSV, fieldnames, rows)
