"""
GenMed AI — FastAPI Backend

HOW TO RUN (always from the project root folder):
    cd genmed_project
    uvicorn backend.main:app --reload --port 8000

API Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json, joblib, numpy as np, os, sys
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

# ── Ensure project root is on sys.path so imports always resolve ───────────────
THIS_FILE = Path(__file__).resolve()           # .../genmed_project/backend/main.py
PROJECT_ROOT = THIS_FILE.parent.parent         # .../genmed_project/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GenMed AI API",
    description="Healthcare Analytics & Disease Prediction Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = PROJECT_ROOT

# ── Load artefacts ─────────────────────────────────────────────────────────────
try:
    model    = joblib.load(BASE / "models" / "model.pkl")
    scaler   = joblib.load(BASE / "models" / "scaler.pkl")
    encoders = joblib.load(BASE / "models" / "encoders.pkl")
    with open(BASE / "models" / "feature_info.json") as f:
        feature_info = json.load(f)
    with open(BASE / "data" / "patients.json") as f:
        ALL_PATIENTS = json.load(f)
    print(f"✅ Loaded {len(ALL_PATIENTS)} patients, model ready.")
except Exception as e:
    print(f"❌ Startup error: {e}")
    model = scaler = encoders = feature_info = None
    ALL_PATIENTS = []

# ── Pydantic models ────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    name: str = Field(..., example="Arjun Kumar")
    age: int = Field(..., ge=1, le=120, example=52)
    gender: str = Field(..., example="Male")
    blood_type: str = Field(..., example="O+")
    admission_type: str = Field(..., example="Emergency")
    test_results: str = Field(..., example="Abnormal")
    billing_amount: float = Field(0.0, example=45000.0)
    symptom_score: float = Field(50.0, ge=0.0, le=100.0, example=75.0,
                                  description="Clinical symptom severity score 0-100")
    bmi: float = Field(25.0, ge=10.0, le=60.0, example=28.5,
                       description="Body Mass Index")
    medication: Optional[str] = Field(None, example="Metformin")
    hospital: Optional[str] = Field(None, example="Apollo Hospitals Chennai")
    doctor: Optional[str] = Field(None, example="Dr. Ravi Kumar")

class PredictResponse(BaseModel):
    predicted_condition: str
    confidence: float
    all_probabilities: dict
    risk_level: str
    recommendation: str
    feature_importances: dict

# ── Helpers ────────────────────────────────────────────────────────────────────
RISK_THRESHOLDS = {"Cancer": 0.4, "Heart Disease": 0.4, "Kidney Disease": 0.35,
                   "Liver Disease": 0.35, "Diabetes": 0.4}

RECOMMENDATIONS = {
    "Diabetes":       "Monitor blood glucose regularly. Maintain low-carb diet. Regular HbA1c checks advised.",
    "Hypertension":   "Reduce sodium intake, regular BP monitoring. Lifestyle changes + medication review.",
    "Asthma":         "Keep rescue inhaler accessible. Avoid triggers. Pulmonary function test recommended.",
    "Cancer":         "Immediate oncology consultation required. Staging workup and biopsy recommended.",
    "Arthritis":      "Physiotherapy and anti-inflammatory regimen. Joint mobility exercises advised.",
    "Heart Disease":  "Cardiology referral urgent. ECG, echo, and lipid panel required immediately.",
    "Obesity":        "Structured diet and exercise plan. BMI monitoring. Endocrinology consult if needed.",
    "Kidney Disease": "Nephrology referral. Renal function panel (eGFR, creatinine). Hydration management.",
    "Liver Disease":  "Hepatology consultation. LFT monitoring. Alcohol abstinence and dietary changes.",
    "Anemia":         "CBC with differential. Iron/B12/folate panel. Dietary supplementation plan."
}

def safe_encode(encoder, value, col_name):
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        raise HTTPException(400, f"Invalid value '{value}' for field '{col_name}'. "
                            f"Valid options: {list(encoder.classes_)}")

def build_features(p: PatientInput, los: int = 3):
    """
    Build the full 19-feature vector matching train_model.py FEATURE_COLS:
    age, age_group, gender_enc, blood_type_enc,
    admission_type_enc, is_emergency,
    test_results_enc, is_abnormal, is_inconclusive,
    billing_amount, billing_per_day, high_billing,
    los, long_stay,
    symptom_score, high_symptom,
    bmi, obese,
    age_x_severity
    """
    enc = encoders
    gender_enc  = safe_encode(enc["gender"],        p.gender,          "gender")
    blood_enc   = safe_encode(enc["blood_type"],    p.blood_type,      "blood_type")
    adm_enc     = safe_encode(enc["admission_type"],p.admission_type,  "admission_type")
    test_enc    = safe_encode(enc["test_results"],  p.test_results,    "test_results")

    los             = max(1, los)
    billing_per_day = p.billing_amount / los

    age_group = (0 if p.age < 18 else
                 1 if p.age < 35 else
                 2 if p.age < 50 else
                 3 if p.age < 65 else 4)

    is_emergency    = 1 if p.admission_type == "Emergency"    else 0
    is_abnormal     = 1 if p.test_results   == "Abnormal"     else 0
    is_inconclusive = 1 if p.test_results   == "Inconclusive" else 0
    high_billing    = 1 if p.billing_amount > 30000           else 0
    long_stay       = 1 if los              > 7               else 0
    high_symptom    = 1 if p.symptom_score  > 60              else 0
    obese           = 1 if p.bmi            > 30              else 0
    age_x_severity  = p.age * p.symptom_score / 100.0

    X = np.array([[
        p.age,            age_group,
        gender_enc,       blood_enc,
        adm_enc,          is_emergency,
        test_enc,         is_abnormal,      is_inconclusive,
        p.billing_amount, billing_per_day,  high_billing,
        los,              long_stay,
        p.symptom_score,  high_symptom,
        p.bmi,            obese,
        age_x_severity
    ]], dtype=float)
    return scaler.transform(X)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "GenMed AI API", "version": "1.0.0"}

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "patients_loaded": len(ALL_PATIENTS),
        "model_type": feature_info.get("model_name") if feature_info else None,
        "model_accuracy": feature_info.get("accuracy") if feature_info else None
    }

# ── Patients endpoints ─────────────────────────────────────────────────────────

@app.get("/patients", tags=["Patients"])
def get_patients(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    condition: Optional[str] = None,
    gender: Optional[str] = None,
    admission_type: Optional[str] = None,
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
):
    """Get paginated patient list with optional filters."""
    data = ALL_PATIENTS
    if condition:
        data = [p for p in data if p["medical_condition"].lower() == condition.lower()]
    if gender:
        data = [p for p in data if p["gender"].lower() == gender.lower()]
    if admission_type:
        data = [p for p in data if p["admission_type"].lower() == admission_type.lower()]
    if min_age is not None:
        data = [p for p in data if p["age"] >= min_age]
    if max_age is not None:
        data = [p for p in data if p["age"] <= max_age]

    total = len(data)
    start = (page - 1) * limit
    end   = start + limit
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit,
        "patients": data[start:end]
    }

@app.get("/patients/{patient_id}", tags=["Patients"])
def get_patient(patient_id: int):
    """Get a single patient by ID."""
    patient = next((p for p in ALL_PATIENTS if p["id"] == patient_id), None)
    if not patient:
        raise HTTPException(404, f"Patient with id={patient_id} not found")
    return patient

@app.get("/patients/search/{name}", tags=["Patients"])
def search_patients(name: str):
    """Search patients by name (case-insensitive partial match)."""
    results = [p for p in ALL_PATIENTS if name.lower() in p["name"].lower()]
    return {"total": len(results), "patients": results[:20]}

# ── Predict endpoint ───────────────────────────────────────────────────────────

@app.post("/predict", tags=["Prediction"], response_model=PredictResponse)
def predict(patient: PatientInput):
    """Predict disease condition for a patient using the trained ML model."""
    if model is None:
        raise HTTPException(503, "Model not loaded. Run models/train_model.py first.")

    X = build_features(patient)
    proba = model.predict_proba(X)[0]
    classes = encoders["medical_condition"].classes_

    pred_idx   = int(np.argmax(proba))
    pred_class = classes[pred_idx]
    confidence = float(proba[pred_idx])

    all_probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    # Risk level
    risk_threshold = RISK_THRESHOLDS.get(pred_class, 0.45)
    if confidence >= 0.7:
        risk_level = "High"
    elif confidence >= risk_threshold:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # Feature importances for this patient
    fi_names   = feature_info["feature_names"]
    fi_values  = feature_info["importances"]
    feat_imp   = {n: round(v, 4) for n, v in zip(fi_names, fi_values)}

    return PredictResponse(
        predicted_condition=pred_class,
        confidence=round(confidence, 4),
        all_probabilities=all_probs,
        risk_level=risk_level,
        recommendation=RECOMMENDATIONS.get(pred_class, "Please consult a physician."),
        feature_importances=feat_imp
    )

# ── Analytics endpoints ────────────────────────────────────────────────────────

@app.get("/analytics/summary", tags=["Analytics"])
def analytics_summary():
    """High-level dataset summary statistics."""
    ages        = [p["age"] for p in ALL_PATIENTS]
    billings    = [p["billing_amount"] for p in ALL_PATIENTS]
    conditions  = {}
    genders     = {}
    blood_types = {}
    admissions  = {}
    test_res    = {}

    for p in ALL_PATIENTS:
        conditions[p["medical_condition"]]  = conditions.get(p["medical_condition"], 0) + 1
        genders[p["gender"]]                = genders.get(p["gender"], 0) + 1
        blood_types[p["blood_type"]]        = blood_types.get(p["blood_type"], 0) + 1
        admissions[p["admission_type"]]     = admissions.get(p["admission_type"], 0) + 1
        test_res[p["test_results"]]         = test_res.get(p["test_results"], 0) + 1

    return {
        "total_patients": len(ALL_PATIENTS),
        "avg_age": round(sum(ages) / len(ages), 1),
        "min_age": min(ages),
        "max_age": max(ages),
        "avg_billing": round(sum(billings) / len(billings), 2),
        "total_revenue": round(sum(billings), 2),
        "conditions": dict(sorted(conditions.items(), key=lambda x: -x[1])),
        "genders": genders,
        "blood_types": blood_types,
        "admission_types": admissions,
        "test_results": test_res
    }

@app.get("/analytics/conditions", tags=["Analytics"])
def analytics_conditions():
    """Condition-wise breakdown with avg age and avg billing."""
    from collections import defaultdict
    cond_data = defaultdict(lambda: {"count": 0, "ages": [], "billings": []})
    for p in ALL_PATIENTS:
        c = p["medical_condition"]
        cond_data[c]["count"] += 1
        cond_data[c]["ages"].append(p["age"])
        cond_data[c]["billings"].append(p["billing_amount"])

    result = {}
    for cond, d in cond_data.items():
        result[cond] = {
            "count": d["count"],
            "avg_age": round(sum(d["ages"]) / len(d["ages"]), 1),
            "avg_billing": round(sum(d["billings"]) / len(d["billings"]), 2),
            "percentage": round(d["count"] / len(ALL_PATIENTS) * 100, 1)
        }
    return result

@app.get("/analytics/age-distribution", tags=["Analytics"])
def analytics_age_distribution():
    """Age distribution in bins of 10 years."""
    bins = {f"{i}-{i+9}": 0 for i in range(0, 100, 10)}
    for p in ALL_PATIENTS:
        for b in bins:
            lo, hi = map(int, b.split("-"))
            if lo <= p["age"] <= hi:
                bins[b] += 1
                break
    return {"bins": bins}

@app.get("/analytics/monthly-admissions", tags=["Analytics"])
def analytics_monthly():
    """Monthly admission counts (all years combined by month name)."""
    months = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
    counts = {m: 0 for m in months}
    billings = {m: 0.0 for m in months}
    for p in ALL_PATIENTS:
        try:
            month_idx = int(p["date_of_admission"].split("-")[1]) - 1
            counts[months[month_idx]]   += 1
            billings[months[month_idx]] += p["billing_amount"]
        except Exception:
            pass
    return {
        "months": months,
        "admissions": [counts[m] for m in months],
        "revenue": [round(billings[m], 2) for m in months]
    }

@app.get("/analytics/top-hospitals", tags=["Analytics"])
def analytics_hospitals():
    """Top hospitals by patient count."""
    hosp = {}
    for p in ALL_PATIENTS:
        h = p.get("hospital", "Unknown")
        hosp[h] = hosp.get(h, 0) + 1
    return dict(sorted(hosp.items(), key=lambda x: -x[1]))

@app.get("/analytics/model-info", tags=["Analytics"])
def model_info():
    """Return trained model metadata and feature importances."""
    if feature_info is None:
        raise HTTPException(503, "Model info not available.")
    return feature_info

# ── Meta endpoints ─────────────────────────────────────────────────────────────

@app.get("/meta/options", tags=["Meta"])
def meta_options():
    """Return valid input options for all categorical fields."""
    return {
        "genders":         list(encoders["gender"].classes_),
        "blood_types":     list(encoders["blood_type"].classes_),
        "admission_types": list(encoders["admission_type"].classes_),
        "test_results":    list(encoders["test_results"].classes_),
        "conditions":      list(encoders["medical_condition"].classes_)
    }

# ── Report Summarization endpoint ──────────────────────────────────────────────

DETAILED_RECOMMENDATIONS = {
    "Diabetes": {
        "immediate": "Monitor fasting blood glucose and HbA1c levels immediately.",
        "medication": "Continue or initiate Metformin as first-line therapy. Review insulin requirements.",
        "lifestyle": "Low glycemic index diet. 150 min/week moderate exercise. Weight management.",
        "followup": "Endocrinology review in 2 weeks. Monthly glucose monitoring. Annual eye and foot exam.",
        "warning_signs": "Hypoglycemia (dizziness, sweating), hyperglycemia (excessive thirst, frequent urination)."
    },
    "Hypertension": {
        "immediate": "Continuous BP monitoring every 4 hours. Target <140/90 mmHg.",
        "medication": "ACE inhibitor (Lisinopril) or Calcium Channel Blocker (Amlodipine) as first-line.",
        "lifestyle": "Low sodium diet (<2g/day). DASH diet recommended. Reduce alcohol and caffeine.",
        "followup": "Cardiology consult within 1 week. 24-hour ambulatory BP monitoring recommended.",
        "warning_signs": "Severe headache, vision changes, chest pain — seek emergency care immediately."
    },
    "Asthma": {
        "immediate": "Ensure rescue inhaler (Salbutamol/Albuterol) is accessible at all times.",
        "medication": "Inhaled corticosteroids (Fluticasone) for maintenance. SABA for acute relief.",
        "lifestyle": "Identify and avoid triggers (dust, pollen, smoke). Use air purifier indoors.",
        "followup": "Pulmonology review in 2 weeks. Spirometry and peak flow monitoring.",
        "warning_signs": "Severe shortness of breath, blue lips, inability to speak — call emergency services."
    },
    "Cancer": {
        "immediate": "Urgent oncology consultation required. Do not delay staging workup.",
        "medication": "Treatment protocol depends on cancer type and stage — await oncology assessment.",
        "lifestyle": "Maintain nutrition. Psychological support and palliative care team involvement recommended.",
        "followup": "Oncology review within 48 hours. CT/MRI staging, biopsy, tumor marker panel.",
        "warning_signs": "Unexplained weight loss, severe pain, fever — report to oncology team immediately."
    },
    "Arthritis": {
        "immediate": "Pain assessment and initiation of anti-inflammatory therapy.",
        "medication": "NSAIDs (Ibuprofen) for mild-moderate pain. DMARDs (Methotrexate) for RA.",
        "lifestyle": "Low-impact exercise (swimming, cycling). Physical therapy. Joint protection techniques.",
        "followup": "Rheumatology consult in 1 week. X-ray of affected joints. ESR/CRP and RF panel.",
        "warning_signs": "Sudden joint swelling, fever with joint pain — may indicate septic arthritis."
    },
    "Heart Disease": {
        "immediate": "Urgent cardiology referral. 12-lead ECG and cardiac enzyme panel immediately.",
        "medication": "Aspirin, statin therapy (Atorvastatin), beta-blocker (Carvedilol) as indicated.",
        "lifestyle": "Cardiac rehabilitation program. Low-fat, low-sodium diet. No smoking. Limit alcohol.",
        "followup": "Echocardiogram, stress test. Cardiology review within 24–48 hours.",
        "warning_signs": "Chest pain, left arm/jaw pain, shortness of breath at rest — call emergency immediately."
    },
    "Obesity": {
        "immediate": "BMI and body composition assessment. Rule out metabolic syndrome.",
        "medication": "Orlistat if BMI >30 with comorbidities. Bariatric surgery evaluation if BMI >40.",
        "lifestyle": "Structured calorie-deficit diet (500 kcal/day deficit). 300 min/week moderate exercise.",
        "followup": "Dietician referral. Endocrinology consult for hormonal evaluation. Monthly weight tracking.",
        "warning_signs": "Chest pain on exertion, severe sleep apnea, rapid weight loss without effort."
    },
    "Kidney Disease": {
        "immediate": "Renal function panel: eGFR, serum creatinine, BUN, electrolytes, urinalysis.",
        "medication": "BP control with ACE inhibitor. Erythropoietin for anemia of CKD. Phosphate binders.",
        "lifestyle": "Low protein diet (<0.8g/kg/day). Restrict potassium and phosphorus. Adequate hydration.",
        "followup": "Nephrology consult within 48 hours. Renal ultrasound. Dialysis planning if eGFR <15.",
        "warning_signs": "Decreased urine output, severe swelling, confusion — seek emergency care."
    },
    "Liver Disease": {
        "immediate": "Liver function tests (LFT), PT/INR, albumin, bilirubin panel.",
        "medication": "Lactulose for hepatic encephalopathy. Rifaximin. Furosemide for ascites.",
        "lifestyle": "Strict alcohol abstinence. Low sodium diet. Adequate protein unless encephalopathy.",
        "followup": "Hepatology consult within 48 hours. Liver ultrasound and fibroscan. HBV/HCV serology.",
        "warning_signs": "Yellowing of eyes/skin, black stools, confusion, severe abdominal swelling."
    },
    "Anemia": {
        "immediate": "Complete blood count with differential. Peripheral blood smear. Reticulocyte count.",
        "medication": "Iron supplementation (Ferrous Sulfate) for IDA. B12 injections for megaloblastic anemia.",
        "lifestyle": "Iron-rich diet (red meat, leafy greens, legumes). Vitamin C with iron supplements.",
        "followup": "Hematology review in 2 weeks. Bone marrow biopsy if aplastic anemia suspected.",
        "warning_signs": "Severe breathlessness at rest, fainting, chest pain — seek emergency care."
    }
}

def calculate_los(admission_date: str, discharge_date: str = None) -> int:
    from datetime import datetime
    try:
        adm = datetime.strptime(admission_date, "%Y-%m-%d")
        dis = datetime.strptime(discharge_date, "%Y-%m-%d") if discharge_date else datetime.now()
        return max(1, (dis - adm).days)
    except Exception:
        return 3

def generate_report_text(patient_data: dict, prediction: dict) -> dict:
    """
    Generate a structured medical report from patient data + ML prediction.
    Pure Python template engine — zero external APIs.
    """
    cond   = prediction["predicted_condition"]
    conf   = prediction["confidence"]
    risk   = prediction["risk_level"]
    detail = DETAILED_RECOMMENDATIONS.get(cond, {})
    los    = calculate_los(
        patient_data.get("date_of_admission", "2024-01-01"),
        patient_data.get("date_of_discharge")
    )

    risk_note = {
        "High":     "URGENT — This patient requires immediate clinical attention.",
        "Moderate": "ELEVATED — Close monitoring and prompt clinical review advised.",
        "Low":      "ROUTINE — Standard follow-up and outpatient management appropriate."
    }.get(risk, "")

    report = {
        "report_id": f"RPT-{patient_data.get('id', 0):04d}",
        "generated_at": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sections": {
            "patient_overview": (
                f"Patient {patient_data.get('name', 'Unknown')} is a "
                f"{patient_data.get('age', '—')}-year-old {patient_data.get('gender', '—')} "
                f"with blood type {patient_data.get('blood_type', '—')}. "
                f"Admitted to {patient_data.get('hospital', 'the hospital')} under the care of "
                f"{patient_data.get('doctor', 'the attending physician')} "
                f"(Room {patient_data.get('room_number', '—')})."
            ),
            "admission_details": (
                f"Admission type: {patient_data.get('admission_type', '—')}. "
                f"Date of admission: {patient_data.get('date_of_admission', '—')}. "
                f"Date of discharge: {patient_data.get('date_of_discharge', 'Ongoing')}. "
                f"Length of stay: {los} day(s). "
                f"Insurance: {patient_data.get('insurance_provider', '—')}."
            ),
            "clinical_findings": (
                f"Laboratory/diagnostic test results reported as: {patient_data.get('test_results', '—')}. "
                f"Current prescribed medication: {patient_data.get('medication', 'None documented')}. "
                f"Total billed amount: INR {patient_data.get('billing_amount', 0):,.2f}."
            ),
            "ai_diagnosis": (
                f"Based on the AI-assisted analysis of patient demographics, admission type, "
                f"test results, and billing data, the most probable primary diagnosis is "
                f"<strong>{cond}</strong> with a model confidence of {conf*100:.1f}%. "
                f"This prediction is generated by a Random Forest classifier trained on "
                f"historical patient data and should be validated by a qualified clinician."
            ),
            "risk_assessment": (
                f"Risk Level: <strong>{risk}</strong>. {risk_note} "
                f"The model evaluated {len(prediction.get('all_probabilities', {}))} disease "
                f"categories. The top contributing features to this prediction are: "
                + ", ".join(
                    f"{k} ({v*100:.1f}%)"
                    for k, v in sorted(
                        prediction.get("feature_importances", {}).items(),
                        key=lambda x: -x[1]
                    )[:3]
                ) + "."
            ),
            "treatment_recommendation": (
                f"<strong>Immediate actions:</strong> {detail.get('immediate', 'Consult physician.')} "
                f"<strong>Medication:</strong> {detail.get('medication', 'As prescribed.')} "
                f"<strong>Lifestyle modifications:</strong> {detail.get('lifestyle', 'Healthy diet and exercise.')} "
                f"<strong>Follow-up:</strong> {detail.get('followup', 'Schedule outpatient review.')}"
            ),
            "warning_signs": (
                f"<strong>Alert signs requiring immediate attention:</strong> "
                f"{detail.get('warning_signs', 'Any sudden worsening of symptoms — seek emergency care.')}"
            ),
            "discharge_summary": (
                f"Patient has been under observation for {los} day(s). "
                f"Billing amount of INR {patient_data.get('billing_amount', 0):,.2f} has been recorded. "
                f"Ensure all discharge medications are dispensed and patient education on "
                f"{cond} management has been completed before discharge. "
                f"This report is auto-generated for clinical reference and must be reviewed "
                f"and countersigned by the attending physician before filing."
            )
        },
        "disclaimer": (
            "This report is generated by an AI-assisted system for educational and "
            "clinical support purposes only. It does NOT constitute a certified medical diagnosis. "
            "All findings must be validated by a licensed healthcare professional."
        )
    }
    return report


class ReportRequest(BaseModel):
    patient_id: Optional[int] = Field(None, description="Existing patient ID from patients.json")
    patient_data: Optional[dict] = Field(None, description="Custom patient data dict")

@app.post("/report/generate", tags=["Report"])
def generate_report(req: ReportRequest):
    """
    Generate a full structured medical report for a patient.
    Combines patient data with ML prediction into 7 report sections.
    No external API — pure template-based generation.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded. Run models/train_model.py first.")

    # Resolve patient data
    if req.patient_id:
        patient = next((p for p in ALL_PATIENTS if p["id"] == req.patient_id), None)
        if not patient:
            raise HTTPException(404, f"Patient {req.patient_id} not found.")
    elif req.patient_data:
        patient = req.patient_data
    else:
        raise HTTPException(400, "Provide either patient_id or patient_data.")

    # Run prediction
    try:
        pi = PatientInput(
            name           = patient.get("name", "Unknown"),
            age            = int(patient.get("age", 30)),
            gender         = patient.get("gender", "Male"),
            blood_type     = patient.get("blood_type", "O+"),
            admission_type = patient.get("admission_type", "Elective"),
            test_results   = patient.get("test_results", "Normal"),
            billing_amount = float(patient.get("billing_amount", 0)),
            medication     = patient.get("medication"),
            hospital       = patient.get("hospital"),
        )
    except Exception as e:
        raise HTTPException(422, f"Patient data validation error: {e}")

    adm = patient.get("date_of_admission", "2024-01-01")
    dis = patient.get("date_of_discharge")
    los = calculate_los(adm, dis)

    X    = build_features(pi, los=los)
    proba = model.predict_proba(X)[0]
    classes = encoders["medical_condition"].classes_
    pred_idx   = int(__import__('numpy').argmax(proba))
    pred_class = classes[pred_idx]
    confidence = float(proba[pred_idx])

    all_probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    risk_threshold = RISK_THRESHOLDS.get(pred_class, 0.45)
    risk_level = "High" if confidence >= 0.7 else ("Moderate" if confidence >= risk_threshold else "Low")

    fi_names  = feature_info["feature_names"]
    fi_values = feature_info["importances"]
    feat_imp  = {n: round(v, 4) for n, v in zip(fi_names, fi_values)}

    prediction = {
        "predicted_condition": pred_class,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
        "risk_level": risk_level,
        "feature_importances": feat_imp
    }

    report = generate_report_text(patient, prediction)
    def _s(v, default="—"): return str(v) if v is not None else default
    report["patient_snapshot"] = {
        "id":                patient.get("id", 0),
        "name":              _s(patient.get("name")),
        "age":               _s(patient.get("age")),
        "gender":            _s(patient.get("gender")),
        "blood_type":        _s(patient.get("blood_type")),
        "medical_condition": _s(patient.get("medical_condition", pred_class)),
        "hospital":          _s(patient.get("hospital")),
        "doctor":            _s(patient.get("doctor")),
        "room_number":       _s(patient.get("room_number")),
        "insurance_provider":_s(patient.get("insurance_provider")),
    }
    report["prediction"] = prediction
    return report


@app.get("/report/generate/{patient_id}", tags=["Report"])
def generate_report_by_id(patient_id: int):
    """Shortcut: generate full report for an existing patient by ID."""
    return generate_report(ReportRequest(patient_id=patient_id))
