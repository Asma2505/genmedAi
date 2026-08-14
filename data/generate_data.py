"""
Script to generate synthetic healthcare dataset (patients.json)
Run: python data/generate_data.py
"""
import json
import random
from datetime import datetime, timedelta

random.seed(42)

CONDITIONS = ["Diabetes", "Hypertension", "Asthma", "Cancer", "Arthritis",
              "Heart Disease", "Obesity", "Kidney Disease", "Liver Disease", "Anemia"]

MEDICATIONS = {
    "Diabetes": ["Metformin", "Insulin", "Glipizide"],
    "Hypertension": ["Lisinopril", "Amlodipine", "Atenolol"],
    "Asthma": ["Albuterol", "Fluticasone", "Montelukast"],
    "Cancer": ["Paclitaxel", "Carboplatin", "Doxorubicin"],
    "Arthritis": ["Ibuprofen", "Methotrexate", "Prednisone"],
    "Heart Disease": ["Aspirin", "Atorvastatin", "Carvedilol"],
    "Obesity": ["Orlistat", "Phentermine", "Bupropion"],
    "Kidney Disease": ["Amlodipine", "Erythropoietin", "Sodium Bicarbonate"],
    "Liver Disease": ["Lactulose", "Rifaximin", "Furosemide"],
    "Anemia": ["Ferrous Sulfate", "Folic Acid", "Erythropoietin"]
}

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
GENDERS = ["Male", "Female"]
ADMISSION_TYPES = ["Emergency", "Elective", "Urgent"]
TEST_RESULTS = ["Normal", "Abnormal", "Inconclusive"]
HOSPITALS = [
    "Apollo Hospitals Chennai", "Fortis Healthcare", "AIIMS Delhi",
    "Manipal Hospital", "Narayana Health", "Max Super Speciality Hospital"
]
DOCTORS = [
    "Dr. Ravi Kumar", "Dr. Priya Sharma", "Dr. Anand Mehta",
    "Dr. Sunita Rao", "Dr. Vikram Singh", "Dr. Deepa Nair",
    "Dr. Arun Patel", "Dr. Kavitha Reddy", "Dr. Sanjay Gupta"
]

def random_date(start_year=2020, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    d = start + timedelta(days=random_days)
    return d.strftime("%Y-%m-%d")

def generate_patient(pid):
    gender = random.choice(GENDERS)
    condition = random.choice(CONDITIONS)
    age = random.randint(18, 85)
    admission = random_date()
    discharge = (datetime.strptime(admission, "%Y-%m-%d") +
                 timedelta(days=random.randint(1, 14))).strftime("%Y-%m-%d")
    billing = round(random.uniform(5000, 85000), 2)

    # Age-biased conditions
    if age > 60:
        condition = random.choices(
            CONDITIONS,
            weights=[15, 20, 5, 15, 15, 20, 5, 10, 8, 7]
        )[0]

    first_names_m = ["Arjun", "Rajesh", "Vikram", "Suresh", "Karthik",
                     "Mohan", "Arun", "Siva", "Ravi", "Anand"]
    first_names_f = ["Priya", "Sunita", "Kavitha", "Deepa", "Anitha",
                     "Meena", "Lakshmi", "Pooja", "Nisha", "Swathi"]
    last_names = ["Kumar", "Sharma", "Reddy", "Nair", "Patel",
                  "Singh", "Gupta", "Rao", "Iyer", "Mehta"]

    fn = random.choice(first_names_m if gender == "Male" else first_names_f)
    ln = random.choice(last_names)

    return {
        "id": pid,
        "name": f"{fn} {ln}",
        "age": age,
        "gender": gender,
        "blood_type": random.choice(BLOOD_TYPES),
        "medical_condition": condition,
        "date_of_admission": admission,
        "date_of_discharge": discharge,
        "admission_type": random.choice(ADMISSION_TYPES),
        "medication": random.choice(MEDICATIONS[condition]),
        "test_results": random.choices(
            TEST_RESULTS,
            weights=[50, 35, 15]
        )[0],
        "billing_amount": billing,
        "hospital": random.choice(HOSPITALS),
        "doctor": random.choice(DOCTORS),
        "room_number": random.randint(100, 500),
        "insurance_provider": random.choice([
            "Star Health", "HDFC Ergo", "New India Assurance",
            "United India Insurance", "Bajaj Allianz", "Self-Pay"
        ])
    }

if __name__ == "__main__":
    patients = [generate_patient(i + 1) for i in range(1000)]
    with open("data/patients.json", "w") as f:
        json.dump(patients, f, indent=2)
    print(f"Generated {len(patients)} patient records → data/patients.json")
