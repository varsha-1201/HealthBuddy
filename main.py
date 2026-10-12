
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="HealthMate.AI",
    description="AI-powered personal health copilot",
    version="1.0.0"
)

# Development configuration for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "app": "HealthMate.AI",
        "status": "running",
        "message": "Your personal health copilot backend is ready"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}




from pydantic import BaseModel, Field

class SymptomRequest(BaseModel):
    symptoms: str = Field(
        min_length=2,
        max_length=2000,
        description="Symptoms described by the user"
    )
    age: int = Field(ge=0, le=120)
    duration: str = Field(
        default="Not specified",
        max_length=100
    )

@app.post("/analyze-symptoms")
def analyze_symptoms(data: SymptomRequest):
    symptoms = data.symptoms.strip()
    symptom_text = symptoms.lower()


    emergency_signs = [
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
        "severe allergic reaction",
        "swelling of tongue",
        "swelling of throat",
        "suicidal thoughts"
    ]

    detected_signs = [
        sign for sign in emergency_signs
        if sign in symptom_text
    ]

    if detected_signs:
        urgency = "URGENT"
        guidance = (
            "Your symptoms may indicate an emergency. "
            "Seek emergency medical care now. "
            "In India, call 112 if appropriate. "
            "Do not wait for this app to assess you."
        )
    else:
        urgency = "REVIEW_NEEDED"
        guidance = (
            "This app cannot determine the cause of your symptoms. "
            "Consult a qualified healthcare professional if symptoms "
            "are concerning, persistent, or getting worse."
        )

    return {
        "success": True,
        "symptoms_received": symptoms,
        "age": data.age,
        "duration": data.duration,
        "urgency": urgency,
        "warning_signs_detected": detected_signs,
        "guidance": guidance,
        "next_steps": [
            "Record when the symptoms started.",
            "Note any changes in symptom severity.",
            "Prepare questions for your healthcare professional."
        ],
        "disclaimer": (
            "This is not a medical diagnosis. The warning-sign "
            "check is limited and cannot rule out an emergency."
        )
    }




class MedicalReportRequest(BaseModel):
    report_text: str = Field(
        min_length=2,
        max_length=10000,
        description="Text copied from a medical or lab report"
    )


@app.post("/analyze-report")
def analyze_report(data: MedicalReportRequest):
    report_text = data.report_text.strip()

    # Extract simple lines for organization.
    lines = [
        line.strip()
        for line in report_text.splitlines()
        if line.strip()
    ]

    return {
        "success": True,
        "report_received": True,
        "report_summary": lines,
        "message": (
            "Your report text has been organized. "
            "This prototype does not yet interpret medical results."
        ),
        "next_steps": [
            "Compare values with the reference ranges printed on the report.",
            "Discuss abnormal or concerning results with a qualified clinician.",
            "Keep the original report available for verification."
        ],
        "disclaimer": (
            "This tool does not diagnose conditions or determine "
            "whether a result is safe."
        )
    }




class MedicineRequest(BaseModel):
    medicine_name: str = Field(min_length=1, max_length=200)
    dosage: str = Field(min_length=1, max_length=100)
    frequency: str = Field(min_length=1, max_length=100)
    instructions: str = Field(
        default="Not specified",
        max_length=500
    )


@app.post("/organize-medicine")
def organize_medicine(data: MedicineRequest):
    return {
        "success": True,
        "medicine": {
            "name": data.medicine_name.strip(),
            "prescribed_dosage": data.dosage.strip(),
            "prescribed_frequency": data.frequency.strip(),
            "instructions": data.instructions.strip()
        },
        "message": (
            "Medicine information organized successfully. "
            "Verify all details against the original prescription."
        ),
        "safety_notice": (
            "This tool does not prescribe medicines or recommend "
            "changes to dosage or frequency. Confirm unclear "
            "instructions with a qualified healthcare professional "
            "or pharmacist."
        )
    }
