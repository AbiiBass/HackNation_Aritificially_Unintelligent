"""
Women's Hormonal Health Platform — Flask application (Member 2's build)

Run with:
    python3 app.py
Then open http://127.0.0.1:5000
"""
import os
import uuid
from functools import wraps

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_from_directory, flash, abort, jsonify
)
from werkzeug.utils import secure_filename

from utils import data_helpers as db
from model import model_adapter
from chatbot import gemini_client, safety

load_dotenv()  # loads GEMINI_API_KEY from a .env file if present

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_TMP_DIR = os.path.join(BASE_DIR, "data", "tmp_uploads")
os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)

MAX_HISTORY_TURNS = 12  # cap stored turns so the session cookie doesn't grow unbounded

app = Flask(__name__)
app.secret_key = "hackathon-demo-secret-key-change-me"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def role_required(role):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "username" not in session:
                return redirect(url_for("login"))
            if session.get("role") != role:
                abort(403)
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        is_employee = request.form.get("is_employee") == "on"
        expected_role = "doctor" if is_employee else "patient"

        user = db.verify_login(username, password, expected_role)
        if user:
            session["username"] = username
            session["role"] = user["role"]
            session["name"] = user["name"]
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials, or account type does not match the selected login.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

@app.route("/home")
@login_required
def home():
    return render_template("home.html")


# ---------------------------------------------------------------------------
# PATIENT ROUTES
# ---------------------------------------------------------------------------

@app.route("/records")
@role_required("patient")
def patient_records():
    national_id = session["username"]
    records = db.get_records_for_patient(national_id)
    return render_template("patient_records.html", records=records)


@app.route("/download/<record_id>/<filename>")
@login_required
def download_record(record_id, filename):
    # permission check: patients may only download their own records
    if session["role"] == "patient":
        records = db.get_records_for_patient(session["username"])
        if not any(r["record_id"] == record_id and r["filename"] == filename for r in records):
            abort(403)
    return send_from_directory(db.RECORDS_FILES_DIR, filename, as_attachment=True)


@app.route("/chatbot")
@role_required("patient")
def chatbot():
    my_questions = db.get_questions_for_patient(session["username"])
    chat_history = session.get("chat_history", [])
    return render_template(
        "chatbot.html",
        my_questions=my_questions,
        chat_history=chat_history,
        chatbot_configured=gemini_client.is_configured(),
    )


@app.route("/chatbot/message", methods=["POST"])
@role_required("patient")
def chatbot_message():
    user_message = (request.json or {}).get("message", "").strip() if request.is_json \
        else request.form.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    history = session.get("chat_history", [])

    # Ask the model for a reply, using only prior turns as context.
    model_history = [{"role": h["role"], "text": h["text"]} for h in history]
    reply = gemini_client.get_reply(model_history, user_message)

    # Safety net: guarantee a strong safety message on clear emergency /
    # self-harm signals, regardless of what the model said.
    safety_flag = safety.check_safety_flags(user_message)
    if safety_flag:
        banner = safety.safety_banner_for(safety_flag)
        reply = f"{banner}\n\n{reply}"

    history.append({"role": "user", "text": user_message})
    history.append({"role": "model", "text": reply})
    session["chat_history"] = history[-(MAX_HISTORY_TURNS * 2):]

    return jsonify({
        "reply": reply,
        "safety_flag": safety_flag,
        "offer_summary": bool(safety_flag),
    })


@app.route("/chatbot/reset", methods=["POST"])
@role_required("patient")
def chatbot_reset():
    session["chat_history"] = []
    return jsonify({"status": "ok"})


@app.route("/chatbot/summarize-for-doctor", methods=["POST"])
@role_required("patient")
def chatbot_summarize_for_doctor():
    history = session.get("chat_history", [])
    model_history = [{"role": h["role"], "text": h["text"]} for h in history]

    if not model_history:
        return jsonify({"error": "There's no conversation yet to summarize."}), 400

    summary = gemini_client.summarize_for_doctor(model_history)
    if not summary:
        return jsonify({"error": "Couldn't generate a summary right now. Please write your question manually."}), 502

    return jsonify({"summary": summary})


@app.route("/ask-doctor", methods=["POST"])
@role_required("patient")
def ask_doctor():
    question_text = request.form.get("question", "").strip()
    if question_text:
        db.add_question(session["username"], session["name"], question_text)
        flash("Your question has been sent to a doctor. You'll see the answer here once it's addressed.", "success")
    return redirect(url_for("chatbot"))


# ---------------------------------------------------------------------------
# DOCTOR ROUTES
# ---------------------------------------------------------------------------

@app.route("/doctor/patient-records", methods=["GET", "POST"])
@role_required("doctor")
def doctor_patient_records():
    patient = None
    records = []
    searched_id = ""

    if request.method == "POST":
        searched_id = request.form.get("national_id", "").strip()
    else:
        searched_id = request.args.get("national_id", "").strip()

    if searched_id:
        patient = db.get_patient(searched_id)
        if patient:
            records = db.get_records_for_patient(searched_id)
        else:
            flash(f"No patient found with National ID '{searched_id}'.", "warning")

    return render_template(
        "doctor_records.html", patient=patient, records=records, searched_id=searched_id
    )


@app.route("/doctor/upload-record", methods=["POST"])
@role_required("doctor")
def upload_record():
    national_id = request.form.get("national_id", "").strip()
    hospital = request.form.get("hospital", "").strip()
    record_type = request.form.get("record_type", "").strip()
    record_date = request.form.get("record_date", "").strip()
    description = request.form.get("description", "").strip()
    file = request.files.get("document")

    patient = db.get_patient(national_id)
    if not patient:
        flash("Cannot upload: patient not found.", "danger")
        return redirect(url_for("doctor_patient_records"))

    if not file or file.filename == "" or not allowed_file(file.filename):
        flash("Please attach a valid PDF file.", "danger")
        return redirect(url_for("doctor_patient_records", national_id=national_id))

    unique_prefix = uuid.uuid4().hex[:8]
    safe_name = secure_filename(file.filename)
    stored_filename = f"{unique_prefix}_{safe_name}"
    file.save(os.path.join(db.RECORDS_FILES_DIR, stored_filename))

    db.add_record(
        national_id=national_id,
        patient_name=patient["name"],
        hospital=hospital,
        record_type=record_type,
        record_date=record_date,
        filename=stored_filename,
        uploaded_by=session["username"],
        description=description,
    )
    flash("Document uploaded successfully.", "success")

    # re-run the search so the doctor sees the updated record list
    records = db.get_records_for_patient(national_id)
    return render_template(
        "doctor_records.html", patient=patient, records=records, searched_id=national_id
    )


@app.route("/doctor/study-analysis", methods=["GET", "POST"])
@role_required("doctor")
def study_analysis():
    result = None
    searched_id = request.args.get("national_id", "").strip()
    patient_records = db.get_records_for_patient(searched_id) if searched_id else []

    if request.method == "POST":
        source = request.form.get("source", "upload")

        if source == "record":
            record_id = request.form.get("record_id", "").strip()
            record = db.get_record_by_id(record_id)
            searched_id = request.form.get("national_id", "").strip()
            patient_records = db.get_records_for_patient(searched_id) if searched_id else []

            if not record:
                flash("Please select a patient record to analyze.", "danger")
                return render_template(
                    "study_analysis.html", result=result,
                    searched_id=searched_id, patient_records=patient_records,
                )

            pdf_path = db.get_record_file_path(record["filename"])
            if not os.path.exists(pdf_path):
                flash("That record's file could not be found on disk.", "danger")
                return render_template(
                    "study_analysis.html", result=result,
                    searched_id=searched_id, patient_records=patient_records,
                )

            result = model_adapter.analyze_pdf(pdf_path)

        else:
            file = request.files.get("document")
            if not file or file.filename == "" or not allowed_file(file.filename):
                flash("Please upload a valid PDF file for analysis.", "danger")
                return redirect(url_for("study_analysis"))

            temp_path = os.path.join(UPLOAD_TMP_DIR, f"{uuid.uuid4().hex}.pdf")
            file.save(temp_path)

            try:
                result = model_adapter.analyze_pdf(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    return render_template(
        "study_analysis.html", result=result,
        searched_id=searched_id, patient_records=patient_records,
    )


@app.route("/doctor/answer-help", methods=["GET"])
@role_required("doctor")
def answer_help():
    questions = db.get_all_questions()
    return render_template("answer_help.html", questions=questions)


@app.route("/doctor/answer-help/<question_id>", methods=["POST"])
@role_required("doctor")
def submit_answer(question_id):
    answer_text = request.form.get("answer", "").strip()
    if answer_text:
        db.answer_question(question_id, answer_text, session["name"])
        flash("Answer submitted to patient.", "success")
    return redirect(url_for("answer_help"))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
