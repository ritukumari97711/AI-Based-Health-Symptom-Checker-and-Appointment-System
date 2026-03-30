import sqlite3
from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "disease_model.pkl"
SYMPTOMS_PATH = BASE_DIR / "symptoms.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
DB_PATH = BASE_DIR / "database.db"

model = joblib.load(MODEL_PATH)
symptoms_list = joblib.load(SYMPTOMS_PATH)
encoder = joblib.load(ENCODER_PATH)


def predict_disease(selected_symptoms):
    if not isinstance(selected_symptoms, list):
        raise ValueError("symptoms must be a list")
    if len(selected_symptoms) < 4:
        raise ValueError("Please provide at least 4 symptoms")

    unknown = [s for s in selected_symptoms if s not in symptoms_list]
    if unknown:
        raise ValueError(f"Unknown symptom(s): {', '.join(unknown)}")

    input_data = [0] * len(symptoms_list)
    for symptom in selected_symptoms:
        input_data[symptoms_list.index(symptom)] = 1

    probabilities = model.predict_proba(np.array(input_data).reshape(1, -1))[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    diseases = encoder.inverse_transform(top3_indices)

    predictions = [
        {"disease": str(diseases[i]), "confidence": float(probabilities[top3_indices[i]])}
        for i in range(3)
    ]
    return predictions


def book_appointment(data):
    required = ["patient_name", "age", "gender", "disease", "doctor", "appointment_date"]
    missing = [field for field in required if field not in data or data[field] in (None, "")]
    if missing:
        raise ValueError(f"Missing field(s): {', '.join(missing)}")

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO appointments
            (patient_name, age, gender, disease, doctor, appointment_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            data["patient_name"], data["age"], data["gender"],
            data["disease"], data["doctor"], data["appointment_date"]
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_appointments():
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, patient_name, age, gender, disease, doctor, appointment_date
            FROM appointments ORDER BY id DESC
        """)
        rows = cursor.fetchall()
        return [
            {
                "id": r[0], "patient_name": r[1], "age": r[2], "gender": r[3],
                "disease": r[4], "doctor": r[5], "appointment_date": r[6]
            }
            for r in rows
        ]
    finally:
        conn.close()
