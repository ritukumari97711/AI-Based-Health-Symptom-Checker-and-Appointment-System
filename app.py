import streamlit as st
import joblib
import numpy as np
import sqlite3
import requests
import pandas as pd

# PAGE CONFIG
st.set_page_config(
    page_title="AI Health Symptom Checker",
    page_icon="🩺",
    layout="wide"
)

# LOAD MODEL FILES

model = joblib.load("disease_model.pkl")
symptoms_list = joblib.load("symptoms.pkl")
encoder = joblib.load("label_encoder.pkl")

API_URL = "http://127.0.0.1:5000/api"

# SESSION STATE

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "top3_diseases" not in st.session_state:
    st.session_state.top3_diseases = []

if "top3_probs" not in st.session_state:
    st.session_state.top3_probs = []

# SIDEBAR

st.sidebar.title("🩺 Health Assistant")

page = st.sidebar.radio(
    "Navigation",
    [
        "Disease Prediction",
        "Appointment History"
    ]
)

st.sidebar.info(
    """
    AI-powered disease prediction system.

    Select symptoms and get possible disease predictions.
    """
)

st.sidebar.success("System Status: Active ✅")

# DISEASE PREDICTION PAGE

if page == "Disease Prediction":

    st.title("🩺 AI Based Health Symptom Checker")

    st.markdown(
        """
        Welcome to the AI-powered healthcare assistant.

        Select your symptoms below to predict possible diseases.
        """
    )

    # SYMPTOM SELECTION

    selected_symptoms = st.multiselect(
        "Choose Symptoms",
        symptoms_list
    )

    # PREDICT BUTTON

    if st.button("Predict Disease"):

        # Check symptom count
        if len(selected_symptoms) < 4:

            st.warning(
                "⚠ Please select at least 4 symptoms for better prediction accuracy."
            )

            st.session_state.prediction_done = False

        else:

            # API INTEGRATION: Streamlit sends symptoms to Flask REST API
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"symptoms": selected_symptoms},
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()

                if not result.get("success"):
                    raise ValueError(result.get("error", "Prediction failed"))

                predictions = result["predictions"]
                top3_diseases = np.array([item["disease"] for item in predictions])
                top3_probs = np.array([item["confidence"] for item in predictions])

            except requests.RequestException:
                st.error("API server is not running. Start api_server.py first.")
                st.stop()
            except ValueError as e:
                st.error(str(e))
                st.stop()

            # Save in session state
            st.session_state.prediction_done = True

            st.session_state.top3_diseases = top3_diseases

            st.session_state.top3_probs = top3_probs

    # SHOW RESULTS AFTER PREDICTION

    if st.session_state.prediction_done:

        st.subheader("🩺 Prediction Result")

        st.success(
            f"Most Probable Disease: {st.session_state.top3_diseases[0]}"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{st.session_state.top3_probs[0]*100:.2f}%"
        )
        
        # TOP 3 PREDICTIONS

        st.subheader("Top 3 Possible Diseases")

        for i in range(3):

            st.write(
                f"{i+1}. "
                f"{st.session_state.top3_diseases[i]} "
                f"→ "
                f"{st.session_state.top3_probs[i]*100:.2f}%"
            )

        # CONFIDENCE WARNING

        if st.session_state.top3_probs[0] < 0.30:

            st.warning(
                "⚠ Prediction confidence is low. "
                "Please provide more accurate symptoms."
            )

        # APPOINTMENT BOOKING

        st.subheader("📅 Book Appointment")

        # Doctor mapping
        doctor_mapping = {
            "Allergy": "Dermatologist",
            "GERD": "Gastroenterologist",
            "Heart attack": "Cardiologist",
            "Bronchial Asthma": "Pulmonologist"
        }

        suggested_doctor = doctor_mapping.get(
            st.session_state.top3_diseases[0],
            "General Physician"
        )

        # APPOINTMENT FORM

        with st.form("appointment_form"):

            patient_name = st.text_input(
                "Enter Patient Name"
            )

            age = st.number_input(
                "Enter Age",
                min_value=1,
                max_value=120,
                step=1
            )

            gender = st.selectbox(
                "Select Gender",
                ["Male", "Female", "Other"]
            )

            doctor = st.text_input(
                "Suggested Doctor",
                value=suggested_doctor
            )

            appointment_date = st.date_input(
                "Select Appointment Date"
            )

            submit_appointment = st.form_submit_button(
                "Book Appointment"
            )

            # SAVE TO DATABASE

            if submit_appointment:

                try:
                    response = requests.post(
                        f"{API_URL}/book",
                        json={
                            "patient_name": patient_name,
                            "age": int(age),
                            "gender": gender,
                            "disease": str(st.session_state.top3_diseases[0]),
                            "doctor": doctor,
                            "appointment_date": str(appointment_date)
                        },
                        timeout=10
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.success("✅ Appointment Booked Successfully!")

                except requests.RequestException:
                    st.error("API server is not running. Start api_server.py first.")
                except Exception as e:
                    st.error(f"API Error: {e}")

# APPOINTMENT HISTORY PAGE

elif page == "Appointment History":

    st.title("📋 Appointment History")

    try:

        response = requests.get(f"{API_URL}/appointments", timeout=10)
        response.raise_for_status()
        result = response.json()
        data = [
            (item["patient_name"], item["age"], item["gender"],
             item["disease"], item["doctor"], item["appointment_date"])
            for item in result.get("appointments", [])
        ]

        if len(data) == 0:

            st.warning(
                "No appointments booked yet."
            )

        else:

            df = pd.DataFrame(
                data,
                columns=[
                    "Patient Name",
                    "Age",
                    "Gender",
                    "Disease",
                    "Doctor",
                    "Appointment Date"
                ]
            )

            st.dataframe(
                df,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Database Error: {e}"
        )

# DISCLAIMER

st.markdown("---")

st.caption(
    "⚠ This system provides AI-based predictions "
    "and should not replace professional medical consultation."
)