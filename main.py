from dotenv import load_dotenv
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# --------------------------------------------------
# HEALTHMATE.AI - FASTAPI BACKEND
# --------------------------------------------------

app = FastAPI(
    title="HealthMate.AI API",
    description=(
        "AI-assisted health information, symptom guidance, "
        "and medical document summaries. Not a diagnostic system."
    ),
    version="2.0.0",
)

# Hackathon development configuration.
# Restrict allow_origins to your actual frontend URL before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

MEDICAL_DISCLAIMER = (
    "HealthBuddy.AI provides informational support, not a diagnosis "
    "or a substitute for a qualified healthcare professional. "
    "AI may make mistakes. Verify medical information with a clinician."
)

SYSTEM_INSTRUCTION = """
You are HealthBuddy, a friendly health-information assistant for ordinary people.

Use simple English, short sentences, and easy-to-understand bullet points.

For health symptoms, organize your answer into these sections:

1. Possible reasons
- List a few common possible reasons for the symptoms.
- Explain that these are possibilities, not a confirmed diagnosis.

2. Safe home care
- Suggest simple, low-risk steps that may help.
- Give home-care advice only when appropriate.
- Never suggest dangerous remedies or unverified treatments.
- Never tell users to start, stop, or change prescribed medicines.

3. When to consult a doctor
- Recommend seeing a doctor if symptoms persist, worsen, or return.
- Explain when medical attention is urgent.
- If emergency warning signs are present, advise immediate emergency care before discussing home remedies.

For medical reports and prescriptions:
- Explain the findings in simple language using only information visible in the document.
- Do not invent missing results or prescribe treatment.
- Recommend discussing important findings with a qualified clinician.

Always:
- Be kind, calm, and easy to understand.
- Do not diagnose diseases or guarantee treatments.
- Explain medical words in everyday language.
- Never delay emergency care with routine advice.
- Remind users that HealthBuddy provides information, not a medical diagnosis.
"""

# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class SymptomRequest(BaseModel):
    symptoms: str = Field(min_length=2, max_length=2000)
    age: int = Field(ge=0, le=120)
    duration: str = Field(default="Not specified", max_length=100)


class ReportRequest(BaseModel):
    report_text: str = Field(min_length=2, max_length=10000)
    question: str = Field(
        default="Summarize this report in plain language.",
        max_length=1000,
    )


class MedicineRequest(BaseModel):
    medicine_name: str = Field(min_length=1, max_length=200)
    dosage: str = Field(default="Not provided", max_length=100)
    frequency: str = Field(default="Not provided", max_length=100)
    instructions: str = Field(default="Not provided", max_length=1000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=4000)


# --------------------------------------------------
# AI HELPER
# --------------------------------------------------

def generate_ai_response(contents, system_instruction=SYSTEM_INSTRUCTION):
    """Send text or multimodal content to Gemini without saving it locally."""

    if ai_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI is not configured. Set GEMINI_API_KEY in your "
                "environment and restart the backend."
            ),
        )

    try:
        response = ai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=1800,
            ),
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response. Please retry.",
            )

        return response.text

    except HTTPException:
        raise
    except Exception as exc:
        print("Gemini API error:", repr(exc))
        raise HTTPException(
            status_code=502,
            detail="AI request failed. Check the VS Code terminal."
        )


# --------------------------------------------------
# BASIC ROUTES
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "app": "HealthMate.AI",
        "status": "running",
        "ai_configured": ai_client is not None,
        "message": "Welcome to the HealthMate.AI backend.",
        "documentation": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ai_configured": ai_client is not None,
    }


# --------------------------------------------------
# SYMPTOM GUIDANCE
# --------------------------------------------------

EMERGENCY_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "can't breathe",
    "cannot breathe",
    "unconscious",
    "fainting",
    "sudden confusion",
    "severe bleeding",
    "face drooping",
    "sudden weakness",
    "swelling of tongue",
    "swelling of throat",
    "severe allergic reaction",
]


@app.post("/analyze-symptoms")
def analyze_symptoms(request: SymptomRequest):
    symptoms_lower = request.symptoms.lower()

    # Basic keyword screening only; it cannot rule out emergencies.
    matched_signs = [
        phrase for phrase in EMERGENCY_KEYWORDS
        if phrase in symptoms_lower
    ]

    if matched_signs:
        return {
            "urgency": "URGENT",
            "warning_signs_detected": matched_signs,
            "message": (
                "These symptoms may indicate a medical emergency. "
                "Seek emergency medical care now. In India, call 112 "
                "if emergency assistance is needed. Do not rely on "
                "this keyword check to assess your safety."
            ),
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    if ai_client is None:
        return {
            "urgency": "REVIEW_NEEDED",
            "message": (
                "Automated keyword screening did not identify one of "
                "its configured warning phrases. This does not rule "
                "out a serious condition. Seek medical care if symptoms "
                "are severe, worsening, or concerning."
            ),
            "ai_analysis": None,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    prompt = f"""
    A person is requesting general guidance about symptoms.

    Age provided: {request.age}
    Duration provided: {request.duration}
    Symptoms provided: {request.symptoms}

    Give general educational information only.
    Do not diagnose or prescribe.
    Mention relevant warning signs and when to seek urgent care.
    Suggest reasonable questions to ask a clinician.
    If information is insufficient, say so.
    """

    result = generate_ai_response(prompt)

    return {
        "urgency": "REVIEW_NEEDED",
        "warning_signs_detected": [],
        "ai_analysis": result,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


# --------------------------------------------------
# TEXT-BASED REPORT ANALYSIS
# Supports pasted lab, diagnostic, and prescription text.
# --------------------------------------------------

@app.post("/analyze-report")
def analyze_report(request: ReportRequest):
    if ai_client is None:
        # Keep a useful non-AI fallback for existing frontend tests.
        lines = [
            line.strip()
            for line in request.report_text.splitlines()
            if line.strip()
        ]

        return {
            "organized_text": "\n".join(lines),
            "ai_summary": None,
            "message": (
                "Report text organized. Configure GEMINI_API_KEY "
                "to enable AI-assisted summarization."
            ),
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    prompt = f"""
    The user supplied this medical document text and asks:
    {request.question}

    Document text:
    ---
    {request.report_text}
    ---

    Identify the apparent document type if possible.
    Summarize the information in plain language.
    If laboratory values and reference ranges are present, repeat
    them accurately and distinguish the reported values from any
    general explanation. Do not invent missing information.
    Explain uncertainty and recommend clinician review.
    For prescriptions, transcribe the instructions rather than
    recommending new medicines or changing doses.
    """

    result = generate_ai_response(prompt)

    return {
        "organized_text": request.report_text,
        "ai_summary": result,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


# --------------------------------------------------
# PDF / IMAGE DOCUMENT ANALYSIS
# Accepts laboratory reports, diagnostic reports,
# scan reports, and prescription images/PDFs.
# Files are processed in memory, not saved to disk.
# --------------------------------------------------

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
}

DOCUMENT_GUIDANCE = {
    "lab_report": (
        "Focus on laboratory tests, values, units, and reference ranges."
    ),
    "diagnostic_report": (
        "Focus on the findings and impression stated in the report."
    ),
    "prescription": (
        "Transcribe medicine names and written instructions carefully. "
        "Mark anything unclear. Do not invent doses or advise changes."
    ),
    "medical_document": (
        "Identify the document type and summarize its visible content."
    ),
}


@app.post("/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    document_type: str = Form("medical_document"),
    question: str = Form(
        "Explain this document in simple language."
    ),
):
    if document_type not in DOCUMENT_GUIDANCE:
        raise HTTPException(
            status_code=400,
            detail=(
                "document_type must be lab_report, diagnostic_report, "
                "prescription, or medical_document."
            ),
        )

    mime_type = (file.content_type or "").lower().split(";")[0].strip()

    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a PDF, JPG, PNG, or WEBP file.",
        )

    data = await file.read(MAX_FILE_SIZE + 1)

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The file is too large. Maximum allowed size is 10 MB.",
        )

    # Check basic file signatures rather than trusting the MIME type alone.
    if mime_type == "application/pdf" and not data.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The file does not appear to be a valid PDF.",
        )

    if mime_type == "image/jpeg" and not data.startswith(b"\xff\xd8\xff"):
        raise HTTPException(
            status_code=400,
            detail="The file does not appear to be a valid JPEG image.",
        )

    if mime_type == "image/png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(
            status_code=400,
            detail="The file does not appear to be a valid PNG image.",
        )

    if mime_type == "image/webp" and not (
        data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    ):
        raise HTTPException(
            status_code=400,
            detail="The file does not appear to be a valid WEBP image.",
        )

    prompt = f"""
    Analyze the attached medical document as an information assistant.

    Requested document category: {document_type}
    Guidance: {DOCUMENT_GUIDANCE[document_type]}
    User question: {question[:1000]}

    Please:
    1. Describe the document type if identifiable.
    2. Summarize what is visibly written.
    3. Explain medical terms in plain language.
    4. For lab values, preserve exact values and units; mention
       reference ranges only if they are visible.
    5. For diagnostic reports, distinguish the report's own findings
       from general explanation.
    6. For prescriptions, transcribe only what is legible and flag
       uncertain names or instructions for pharmacist/clinician review.
    7. Do not diagnose, prescribe, calculate a new dose, or recommend
       stopping or changing prescribed medication.
    8. Clearly state if parts of the document cannot be read.
    """

    # The bytes are sent to the AI provider for processing.
    # This endpoint does not write the uploaded document to local disk.
    document_part = types.Part.from_bytes(
        data=data,
        mime_type=mime_type,
    )

    result = generate_ai_response([prompt, document_part])

    return {
        "filename": file.filename,
        "document_type": document_type,
        "summary": result,
        "saved_to_server": False,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


# --------------------------------------------------
# MEDICINE INFORMATION ORGANIZER
# Formats information supplied by the user; does not prescribe.
# --------------------------------------------------

@app.post("/organize-medicine")
def organize_medicine(request: MedicineRequest):
    return {
        "medicine_name": request.medicine_name,
        "prescribed_dosage_as_entered": request.dosage,
        "frequency_as_entered": request.frequency,
        "instructions_as_entered": request.instructions,
        "safety_note": (
            "This endpoint only organizes information you entered. "
            "Confirm unclear instructions with your pharmacist or "
            "prescribing clinician. Do not change a dose based on AI."
        ),
        "disclaimer": MEDICAL_DISCLAIMER,
    }


# --------------------------------------------------
# AI HEALTH ASSISTANT / CHAT
# --------------------------------------------------


@app.post("/chat")
def health_chat(request: ChatRequest):
    
    prompt = f"""
    User's question and instructions:
    {request.message}

    LANGUAGE PRIORITY:
    Follow any explicit language requirement included in the user's
    message. If a selected response language is specified, it takes
    priority over the language used to write the question.

    Answer entirely in the selected language specified in the
user's message. Supported languages include English, Hindi,
Telugu, Bengali, Marathi, Kannada, Tamil, Malayalam, Gujarati,
Punjabi, Odia, Assamese, and Urdu as instructed.
    Use simple, natural language that is easy to understand.
    Preserve medicine names, dosages, units, and medical values exactly.

    Answer in plain language and provide general health education.
    Do not diagnose, prescribe, or make unsupported claims.
    If the question describes possible emergency warning signs,
    advise immediate emergency assessment.
    """

    result = generate_ai_response(prompt)

    return {
        "reply": result,
        "disclaimer": MEDICAL_DISCLAIMER,
    }

