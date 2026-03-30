from flask import Blueprint, jsonify, request
from .services import predict_disease, book_appointment, get_appointments, symptoms_list

api_bp = Blueprint("api", __name__)


@api_bp.get("/health")
def health():
    return jsonify({"status": "active", "message": "Health Symptom Checker API is running"})


@api_bp.get("/symptoms")
def symptoms():
    return jsonify({"symptoms": list(symptoms_list)})


@api_bp.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True) or {}
        predictions = predict_disease(data.get("symptoms", []))
        return jsonify({
            "success": True,
            "predictions": predictions,
            "disease": predictions[0]["disease"],
            "confidence": predictions[0]["confidence"]
        }), 200
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "Prediction failed", "details": str(e)}), 500


@api_bp.post("/book")
def book():
    try:
        data = request.get_json(silent=True) or {}
        appointment_id = book_appointment(data)
        return jsonify({
            "success": True,
            "message": "Appointment booked successfully",
            "appointment_id": appointment_id
        }), 201
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "Database error", "details": str(e)}), 500


@api_bp.get("/appointments")
def appointments():
    try:
        return jsonify({"success": True, "appointments": get_appointments()}), 200
    except Exception as e:
        return jsonify({"success": False, "error": "Database error", "details": str(e)}), 500
