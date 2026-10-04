from pathlib import Path
import sys
import csv
import io
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
# Locate the project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import the existing fraud risk engine
from risk_engine import FraudRiskEngine

# Import our SQLite database functions
from database.db import initialize_database, get_connection


app = FastAPI(
    title="AI-Powered Fraud Detection & Risk Analytics Platform",
    version="1.0.0"
)

# Allow the local HTML dashboard to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null",
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the database table
initialize_database()

# Load the trained model
try:
    engine = FraudRiskEngine()
    print("Fraud detection model loaded successfully.")
except Exception as error:
    engine = None
    print(f"Model loading error: {error}")


class TransactionInput(BaseModel):
    Time: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": engine is not None
    }


@app.post("/predict")
def predict(transaction: TransactionInput):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Fraud detection model is not available."
        )

    try:
        # Generate the prediction using the trained model
        result = engine.predict(transaction.model_dump())

        # Support either dictionary or object-style results
        if not isinstance(result, dict):
            result = vars(result)

        prediction = str(result["prediction"])
        risk_score = float(result["risk_score"])
        risk_level = str(result["risk_level"])

        # Save the prediction in SQLite
        connection = get_connection()

        try:
            connection.execute(
                """
                INSERT INTO transaction_history
                (transaction_time, amount, prediction, risk_score, risk_level)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    transaction.Time,
                    transaction.Amount,
                    prediction,
                    risk_score,
                    risk_level
                )
            )
            connection.commit()
        finally:
            connection.close()

        # Return the prediction to the dashboard
        return {
            "prediction": prediction,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "saved_to_database": True
        }

    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction or database error: {error}"
        )

@app.get("/history")
def get_transaction_history():
    """Return the 100 most recent saved predictions."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                transaction_time,
                amount,
                prediction,
                risk_score,
                risk_level,
                created_at
            FROM transaction_history
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()

        return {
            "total_returned": len(rows),
            "transactions": [dict(row) for row in rows]
        }

    finally:
        connection.close()

@app.get("/history/export")
def export_history():
    connection = get_connection()

    try:
        rows = connection.execute("""
            SELECT
                id,
                transaction_time,
                amount,
                prediction,
                risk_score,
                risk_level,
                created_at
            FROM transaction_history
            ORDER BY id DESC
        """).fetchall()

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "ID",
            "Transaction Time",
            "Amount",
            "Prediction",
            "Risk Score",
            "Risk Level",
            "Created At"
        ])

        for row in rows:
            writer.writerow([
                row["id"],
                row["transaction_time"],
                row["amount"],
                row["prediction"],
                row["risk_score"],
                row["risk_level"],
                row["created_at"]
            ])

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    "attachment; filename=fraud_analysis_report.csv"
            }
        )

    finally:
        connection.close()

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_REPORT = PROJECT_DIR / "reports" / "fraud_analysis_report.pdf"


@app.get("/reports/pdf")
def download_pdf_report():
    if not PDF_REPORT.exists():
        return {
            "error": "PDF report not found. Generate it first."
        }

    return FileResponse(
        path=str(PDF_REPORT),
        media_type="application/pdf",
        filename="fraud_analysis_report.pdf"
    )
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_COMPARISON_FILE = PROJECT_DIR / "reports" / "model_comparison.csv"


@app.get("/model-metrics")
def get_model_metrics():
    if not MODEL_COMPARISON_FILE.exists():
        return {"error": "Model comparison file not found. Train the models first."}

    return FileResponse(
        path=str(MODEL_COMPARISON_FILE),
        media_type="text/csv",
        filename="model_comparison.csv"
    )