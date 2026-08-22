"""
Nizhal Trust — NGO website backend
Flask app that serves the single-page site and powers an NGO-FAQ
chatbot using the Gemini API, grounded on a small local knowledge base
(RAG-lite — same pattern as CloudMate AI), with a rule-based fallback
if the API key is missing or the call fails.
"""

import os
import re
import json
from difflib import SequenceMatcher

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = "gemini-3.6-flash"

_genai_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:  # SDK missing or bad key — fall back gracefully
        print(f"[WARN] Gemini client could not be initialised: {e}")
        _genai_client = None
else:
    print("[WARN] GEMINI_API_KEY not set — chatbot will use local fallback only. "
          "Create a .env file with GEMINI_API_KEY=your_key_here")

# ---------------------------------------------------------------------------
# FAQ knowledge base — used as grounding context for Gemini, and as the
# local fallback matcher (difflib + keywords) when the API is unavailable.
# ---------------------------------------------------------------------------
FAQ_DATA = [
    {
        "id": "donate_how",
        "question": "How can I donate to Nizhal Trust",
        "keywords": ["donate", "donation", "give money", "contribute", "give", "fund"],
        "answer": "You can donate directly on our Donate page — choose an amount, pick a programme (Education, Water or Health), and pay securely. One-time and monthly gifts are both supported.",
        "suggestions": ["Is my donation tax exempt?", "Where does my money go?", "Can I donate monthly?"],
    },
    {
        "id": "tax_exempt",
        "question": "Is my donation tax exempt under 80G",
        "keywords": ["80g", "tax", "tax exemption", "tax benefit", "exempt"],
        "answer": "Yes. Nizhal Trust is a registered charitable trust and donations are eligible for tax benefits under Section 80G of the Income Tax Act. You'll receive a receipt by email for your records.",
        "suggestions": ["How can I donate?", "How do I contact you?"],
    },
    {
        "id": "fund_use",
        "question": "Where does my donation money go",
        "keywords": ["where does money go", "fund usage", "how is money used", "transparency", "spending"],
        "answer": "100% of individual donations go directly to programme work — our founders and board serve without pay. Funds are split across Education (Kalvi Kudil), Clean Water (Neer Payanam) and Health (Nalam Kaakum), and you can direct your gift to a specific one.",
        "suggestions": ["What programmes do you run?", "How can I volunteer?"],
    },
    {
        "id": "monthly_giving",
        "question": "Can I set up a monthly donation",
        "keywords": ["monthly", "recurring", "subscription", "every month"],
        "answer": "Yes — on the Donate page there's a 'Make this a monthly gift' option. You can change or cancel it anytime by writing to us.",
        "suggestions": ["How can I donate?", "How do I contact you?"],
    },
    {
        "id": "volunteer",
        "question": "How can I volunteer",
        "keywords": ["volunteer", "volunteering", "help out", "join as volunteer"],
        "answer": "We'd love that! Fill the form in the Contact section and select 'Volunteering' as the reason. Most of our field coordinators and tutors started exactly that way — as local youth volunteers.",
        "suggestions": ["What programmes do you run?", "Do you take student interns?"],
    },
    {
        "id": "internship",
        "question": "Do you offer internships for students",
        "keywords": ["internship", "intern", "student volunteer", "college project"],
        "answer": "Yes, we host short-term student internships across programme, communications and field-research work, mainly for college students in Tamil Nadu. Reach out via the Contact form with your college and area of interest.",
        "suggestions": ["How can I contact you?", "What programmes do you run?"],
    },
    {
        "id": "programs",
        "question": "What programmes does Nizhal Trust run",
        "keywords": ["programs", "programmes", "what do you do", "projects", "work"],
        "answer": "Three programmes: Kalvi Kudil (after-school learning centres), Neer Payanam (clean water — borewell recharge & rainwater harvesting), and Nalam Kaakum (health camps & child nutrition). All are run with, not for, the village committees.",
        "suggestions": ["Tell me about education programme", "Tell me about water programme", "Tell me about health programme"],
    },
    {
        "id": "program_education",
        "question": "Tell me about the education programme",
        "keywords": ["education programme", "kalvi kudil", "learning centre", "school", "scholarship"],
        "answer": "Kalvi Kudil runs 24 after-school learning centres across 3 districts — library corners, volunteer tutors and scholarships for first-generation learners, with over 1,400 children currently enrolled.",
        "suggestions": ["Tell me about water programme", "Tell me about health programme", "How can I donate?"],
    },
    {
        "id": "program_water",
        "question": "Tell me about the water programme",
        "keywords": ["water programme", "neer payanam", "borewell", "rainwater", "clean water"],
        "answer": "Neer Payanam has rehabilitated 31 borewells and built 9 rainwater harvesting structures since 2014, with monthly water-testing across all 46 villages we work in.",
        "suggestions": ["Tell me about education programme", "Tell me about health programme", "How can I donate?"],
    },
    {
        "id": "program_health",
        "question": "Tell me about the health programme",
        "keywords": ["health programme", "nalam kaakum", "health camp", "nutrition", "maternal"],
        "answer": "Nalam Kaakum runs monthly health camps with a district hospital partner, maternal care support, and a nutrition programme currently supporting 380 children under six.",
        "suggestions": ["Tell me about education programme", "Tell me about water programme", "How can I donate?"],
    },
    {
        "id": "founded",
        "question": "When was Nizhal Trust founded",
        "keywords": ["founded", "started", "history", "when did you start", "established"],
        "answer": "Nizhal Trust started in 2014 in Thiruvarur, after a group of engineering graduates returned home and rebuilt a dried-up village borewell alongside the villagers themselves. That's still our method today.",
        "suggestions": ["What programmes do you run?", "How many villages do you work in?"],
    },
    {
        "id": "reach",
        "question": "How many villages do you work in",
        "keywords": ["how many villages", "reach", "impact", "districts", "villages covered"],
        "answer": "We currently work across 46 villages in 3 districts of Tamil Nadu, touching over 12,800 lives since 2014.",
        "suggestions": ["When was Nizhal Trust founded?", "What programmes do you run?"],
    },
    {
        "id": "contact",
        "question": "How do I contact Nizhal Trust",
        "keywords": ["contact", "reach you", "phone number", "email address", "address", "location"],
        "answer": "Email hello@nizhaltrust.org or call +91 43762 21100. Our office is at 14 Kaveri Street, Thiruvarur, Tamil Nadu 610001 — open Monday to Saturday, 9:30 AM to 5:30 PM.",
        "suggestions": ["How can I volunteer?", "How can I donate?"],
    },
    {
        "id": "csr",
        "question": "Can companies partner for CSR",
        "keywords": ["csr", "corporate", "partner", "partnership", "company donation"],
        "answer": "Yes, we welcome CSR partnerships under Section 135 — we can co-design a programme scope with your team and provide impact reporting. Write to hello@nizhaltrust.org with 'CSR Partnership' in the subject.",
        "suggestions": ["How do I contact you?", "Is my donation tax exempt?"],
    },
    {
        "id": "in_kind",
        "question": "Can I donate items instead of money",
        "keywords": ["in kind", "donate items", "books donation", "clothes", "material donation"],
        "answer": "Yes — we regularly accept books, learning materials, and health-camp supplies. Message us through the Contact form so we can match your donation to a centre that needs it.",
        "suggestions": ["How can I donate?", "How do I contact you?"],
    },
]

FALLBACK_ANSWER = (
    "I don't have an exact answer for that yet. For anything specific, please "
    "write to hello@nizhaltrust.org or use the Contact form below — our team "
    "replies within 3 working days."
)
FALLBACK_SUGGESTIONS = ["How can I donate?", "What programmes do you run?", "How do I contact you?"]

GREETING_WORDS = {"hi", "hello", "hey", "vanakkam", "good morning", "good evening"}
THANKS_WORDS = {"thank", "thanks", "thank you", "nandri"}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


# ---------------------------------------------------------------------------
# Local fallback matcher (no API needed) — keyword overlap + fuzzy match
# ---------------------------------------------------------------------------
def score_entry(user_text: str, entry: dict) -> float:
    norm_user = normalize(user_text)
    user_tokens = set(norm_user.split())

    keyword_score = 0.0
    for term in entry["keywords"] + [entry["question"]]:
        norm_term = normalize(term)
        if not norm_term:
            continue
        if norm_term in norm_user or norm_user in norm_term:
            keyword_score = max(keyword_score, 0.9)
            continue
        term_tokens = set(norm_term.split())
        if term_tokens and user_tokens:
            overlap = len(term_tokens & user_tokens) / len(term_tokens)
            keyword_score = max(keyword_score, overlap * 0.85)

    fuzzy_score = SequenceMatcher(None, norm_user, normalize(entry["question"])).ratio()
    return max(keyword_score, fuzzy_score)


def local_fallback_reply(message: str) -> dict:
    norm = normalize(message)
    if norm in GREETING_WORDS or any(w in norm for w in GREETING_WORDS):
        return {"reply": "Vanakkam! I'm the Nizhal Trust FAQ assistant. Ask me about donating, volunteering, our programmes, or how to reach us.",
                "suggestions": ["How can I donate?", "How can I volunteer?", "What programmes do you run?"]}
    if any(w in norm for w in THANKS_WORDS):
        return {"reply": "Nandri! Happy to help. Anything else you'd like to know about Nizhal Trust?",
                "suggestions": ["What programmes do you run?", "How can I donate?"]}

    best_entry, best_score = None, 0.0
    for entry in FAQ_DATA:
        s = score_entry(message, entry)
        if s > best_score:
            best_entry, best_score = entry, s

    if best_entry and best_score >= 0.32:
        return {"reply": best_entry["answer"], "suggestions": best_entry["suggestions"]}

    return {"reply": FALLBACK_ANSWER, "suggestions": FALLBACK_SUGGESTIONS}


# ---------------------------------------------------------------------------
# Gemini-powered reply, grounded on the FAQ knowledge base
# ---------------------------------------------------------------------------
def build_knowledge_context() -> str:
    lines = []
    for entry in FAQ_DATA:
        lines.append(f"Q: {entry['question']}\nA: {entry['answer']}")
    return "\n\n".join(lines)


SYSTEM_PROMPT = f"""You are "Ask Nizhal", the FAQ assistant on the Nizhal Trust NGO website.
Nizhal Trust is a community-led charitable trust working in rural Tamil Nadu since 2014,
across three programmes: Kalvi Kudil (education), Neer Payanam (clean water),
and Nalam Kaakum (health & nutrition). It is registered under the Indian Trusts Act
and donations qualify for 80G tax benefits.

Answer ONLY questions about Nizhal Trust — its mission, programmes, donating,
volunteering, impact, history, or how to contact them. Use the knowledge base below
as your source of truth; do not invent numbers or facts not grounded in it.
If a question is unrelated to Nizhal Trust (e.g. general knowledge, coding help,
other organisations), politely redirect the user back to what you can help with.

Knowledge base:
{build_knowledge_context()}

Respond ONLY with a single JSON object, no markdown fences, no preamble, in this
exact shape:
{{"reply": "<your answer, 1-3 sentences, warm and concise>", "suggestions": ["<follow-up question 1>", "<follow-up question 2>", "<follow-up question 3>"]}}
The "suggestions" must be short, natural follow-up questions a visitor might
ask next, relevant to your reply."""


def gemini_reply(message: str):
    if not _genai_client:
        return None
    try:
        response = _genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
                {"role": "model", "parts": [{"text": "Understood. I'll answer only as Ask Nizhal, in that JSON shape."}]},
                {"role": "user", "parts": [{"text": message}]},
            ],
        )
        raw_text = (response.text or "").strip()
        cleaned = re.sub(r"^```json|^```|```$", "", raw_text, flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)

        reply = parsed.get("reply", "").strip()
        suggestions = parsed.get("suggestions", [])
        if not reply:
            return None
        if not isinstance(suggestions, list):
            suggestions = []
        return {"reply": reply, "suggestions": suggestions[:3]}
    except Exception as e:
        print(f"[WARN] Gemini call failed, using local fallback: {e}")
        return None



# ---------------------------------------------------------------------------
# NGO GROWTH / IMPACT DATA
# This data is sent to index.html and used by the Chart.js visualization.
# ---------------------------------------------------------------------------
GROWTH_DATA = {
    "years": [2014, 2022, 2026],
    "villages": [1, 30, 46],
    "lives_touched": [0, 0, 12800],
    "learning_centres": [0, 0, 24],
    "events": [0, 0, 0]
}

@app.route("/")
def home():
    return render_template(
        "index.html",
        growth_data=GROWTH_DATA
    )



@app.route("/api/growth", methods=["GET"])
def growth():
    """Return NGO growth/impact data for the website visualization."""
    return jsonify(GROWTH_DATA)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"reply": "Please type a question so I can help.",
                         "suggestions": FALLBACK_SUGGESTIONS, "source": "none"})

    result = gemini_reply(message)
    source = "gemini"
    if result is None:
        result = local_fallback_reply(message)
        source = "local_fallback"

    return jsonify({"reply": result["reply"], "suggestions": result["suggestions"], "source": source})


@app.route("/api/donate", methods=["POST"])
def donate():
    """Demo endpoint — validates and echoes back, no real payment processing."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    amount = data.get("amount")

    if not name or not email or not amount:
        return jsonify({"ok": False, "error": "Name, email and amount are required."}), 400

    print(f"[DEMO DONATION] {name} <{email}> pledged Rs.{amount} to "
          f"{data.get('program', 'Wherever needed most')}")

    return jsonify({
        "ok": True,
        "message": f"Thank you, {name}! Your Rs.{amount} gift has been recorded (demo mode — no real payment taken).",
    })


@app.route("/api/contact", methods=["POST"])
def contact():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    msg = (data.get("message") or "").strip()

    if not name or not email or not msg:
        return jsonify({"ok": False, "error": "Name, email and message are required."}), 400

    print(f"[DEMO CONTACT] {name} <{email}> ({data.get('reason', 'General')}): {msg}")

    return jsonify({
        "ok": True,
        "message": "Message sent! We typically reply within 3 working days.",
    })


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
