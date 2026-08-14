"""
GenMed AI — Model Training Script
Run from project root: python models/train_model.py
"""
import json, joblib, warnings, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)

print("=" * 60)
print("  GenMed AI — Model Training")
print("=" * 60)

with open(os.path.join(BASE, "data", "patients.json")) as f:
    patients = json.load(f)

df = pd.DataFrame(patients)
print(f"\n  Loaded {len(df)} records | {df['medical_condition'].nunique()} classes")

# Feature engineering
df["admission_dt"]    = pd.to_datetime(df["date_of_admission"])
df["discharge_dt"]    = pd.to_datetime(df["date_of_discharge"])
df["los"]             = (df["discharge_dt"] - df["admission_dt"]).dt.days.clip(lower=1)
df["billing_per_day"] = df["billing_amount"] / df["los"]
df["age_group"]       = pd.cut(df["age"], bins=[0,18,35,50,65,120], labels=[0,1,2,3,4]).astype(int)
df["is_emergency"]    = (df["admission_type"] == "Emergency").astype(int)
df["is_abnormal"]     = (df["test_results"]   == "Abnormal").astype(int)
df["is_inconclusive"] = (df["test_results"]   == "Inconclusive").astype(int)
df["high_billing"]    = (df["billing_amount"] > 30000).astype(int)
df["long_stay"]       = (df["los"]            > 7).astype(int)
df["symptom_score"]   = df.get("symptom_score", pd.Series([50.0]*len(df)))
df["bmi"]             = df.get("bmi",           pd.Series([25.0]*len(df)))
df["high_symptom"]    = (df["symptom_score"]  > 60).astype(int)
df["obese"]           = (df["bmi"]            > 30).astype(int)
df["age_x_severity"]  = df["age"] * df["symptom_score"] / 100.0

# Encode
encoders = {}
for col in ["gender", "blood_type", "admission_type", "test_results"]:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

le_target = LabelEncoder()
y = le_target.fit_transform(df["medical_condition"])
encoders["medical_condition"] = le_target

FEATURE_COLS = [
    "age", "age_group",
    "gender_enc", "blood_type_enc",
    "admission_type_enc", "is_emergency",
    "test_results_enc", "is_abnormal", "is_inconclusive",
    "billing_amount", "billing_per_day", "high_billing",
    "los", "long_stay",
    "symptom_score", "high_symptom",
    "bmi", "obese",
    "age_x_severity"
]
FEATURE_NAMES = [
    "Age", "Age Group",
    "Gender", "Blood Type",
    "Admission Type", "Is Emergency",
    "Test Results", "Is Abnormal", "Is Inconclusive",
    "Billing Amount", "Billing Per Day", "High Billing",
    "Length of Stay", "Long Stay",
    "Symptom Score", "High Symptom",
    "BMI", "Obese",
    "Age x Severity"
]

X = df[FEATURE_COLS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# Train
print("\n  Training Random Forest (500 trees)...")
rf = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_leaf=1,
    max_features="sqrt", class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"  RandomForest:     {rf_acc*100:.1f}%")

print("  Training Gradient Boosting (300 trees)...")
gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, min_samples_split=4, random_state=42
)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"  GradientBoosting: {gb_acc*100:.1f}%")

print("  Training Voting Ensemble...")
ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")
ensemble.fit(X_train, y_train)
ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
print(f"  Voting Ensemble:  {ens_acc*100:.1f}%")

scores    = {"RandomForest":(rf,rf_acc), "GradientBoosting":(gb,gb_acc), "Ensemble":(ensemble,ens_acc)}
best_name, (best_model, best_acc) = max(scores.items(), key=lambda x: x[1][1])
print(f"\n  Best model: {best_name} — {best_acc*100:.1f}%")

cv_scores = cross_val_score(best_model, X_scaled, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=42),
                             scoring="accuracy", n_jobs=-1)
print(f"  5-fold CV: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print("\n" + classification_report(y_test, best_model.predict(X_test), target_names=le_target.classes_))

joblib.dump(best_model, os.path.join(BASE, "models", "model.pkl"))
joblib.dump(scaler,     os.path.join(BASE, "models", "scaler.pkl"))
joblib.dump(encoders,   os.path.join(BASE, "models", "encoders.pkl"))

fi_vals = (ensemble.estimators_[0].feature_importances_.tolist()
           if best_name == "Ensemble"
           else best_model.feature_importances_.tolist())

with open(os.path.join(BASE, "models", "feature_info.json"), "w") as f:
    json.dump({
        "feature_names": FEATURE_NAMES,
        "feature_cols":  FEATURE_COLS,
        "importances":   fi_vals,
        "model_name":    best_name,
        "accuracy":      round(best_acc, 4),
        "cv_mean":       round(float(cv_scores.mean()), 4),
        "cv_std":        round(float(cv_scores.std()), 4),
        "classes":       le_target.classes_.tolist()
    }, f, indent=2)

print(f"  Models saved. Final accuracy: {best_acc*100:.1f}%")
print("=" * 60)
