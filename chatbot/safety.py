"""
Lightweight, keyword-based safety net for the chatbot.

This is a deliberately simple backup layer — it does NOT replace the system
prompt instructions given to the model. Its only job is to guarantee that if
a message contains a clear emergency or self-harm signal, the user sees a
strong, unmissable safety message every single time, even if the model
response is delayed, errors out, or (rarely) doesn't handle it perfectly.

Keep this fast and conservative: it's fine to over-trigger occasionally
(showing a safety banner when it wasn't strictly needed does no harm),
it should never under-trigger on a clear signal.
"""

SELF_HARM_KEYWORDS = [
    "kill myself", "end my life", "suicide", "suicidal", "want to die",
    "don't want to live", "hurting myself", "self harm", "self-harm",
    "harming myself",
]

EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "severe bleeding", "heavy bleeding", "soaking through a pad every hour",
    "soaking a pad every hour", "fainted", "fainting", "passed out",
    "losing consciousness", "severe abdominal pain", "unbearable pain",
    "seizure", "stroke", "can't move", "numbness on one side",
]

SELF_HARM_MESSAGE = (
    "I'm really glad you reached out, and I want you to get support from "
    "someone who can help right now. If you are thinking about harming "
    "yourself or ending your life, please contact a crisis line immediately: "
    "in the US, call or text 988 (Suicide & Crisis Lifeline), or contact your "
    "local emergency number. If you're not in immediate danger but still "
    "struggling, please consider reaching out to a doctor, therapist, or "
    "someone you trust. You deserve support."
)

EMERGENCY_MESSAGE = (
    "What you're describing could be a medical emergency. Please contact "
    "emergency services (e.g. 911 in the US) or go to the nearest emergency "
    "room right away. Don't wait for an online response — this needs "
    "in-person medical attention now."
)


def check_safety_flags(message: str):
    """
    Returns 'self_harm', 'emergency', or None based on simple keyword matching.
    self_harm takes priority if both are present.
    """
    lower = message.lower()

    if any(kw in lower for kw in SELF_HARM_KEYWORDS):
        return "self_harm"

    if any(kw in lower for kw in EMERGENCY_KEYWORDS):
        return "emergency"

    return None


def safety_banner_for(flag: str):
    if flag == "self_harm":
        return SELF_HARM_MESSAGE
    if flag == "emergency":
        return EMERGENCY_MESSAGE
    return None
