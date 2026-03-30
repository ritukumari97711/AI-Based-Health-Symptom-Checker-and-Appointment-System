# API Development & Integration Module

This project uses Flask REST APIs between the Streamlit UI and the machine-learning/database layers.

## Run

Terminal 1:
```bash
python api_server.py
```

Terminal 2:
```bash
streamlit run app.py
```

## Endpoints

- `GET /api/health` — API health check
- `GET /api/symptoms` — available symptoms
- `POST /api/predict` — disease prediction from selected symptoms
- `POST /api/book` — save an appointment
- `GET /api/appointments` — retrieve appointment history

## Example prediction request
```json
{
  "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"]
}
```
