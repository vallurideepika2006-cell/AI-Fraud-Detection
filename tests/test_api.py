from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_history_endpoint():
    response = client.get("/history")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)
    assert "transactions" in data
    assert isinstance(data["transactions"], list)
    assert "total_returned" in data
    assert data["total_returned"] == len(data["transactions"])


def test_prediction_endpoint():
    transaction = {
        "Time": 0,
        **{f"V{i}": 0 for i in range(1, 29)},
        "Amount": 100
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "risk_score" in result
    assert "risk_level" in result

    assert result["prediction"] in ["Fraud", "Legitimate"]
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in ["Low", "Medium", "High"]


def test_prediction_rejects_missing_features():
    response = client.post(
        "/predict",
        json={"Time": 0, "Amount": 100}
    )

    assert response.status_code == 422


def test_prediction_rejects_non_finite_values():
    from api.main import TransactionInput
    from pydantic import ValidationError
    import pytest

    invalid_values = [
        float("nan"),
        float("inf"),
        float("-inf")
    ]

    for invalid_value in invalid_values:
        transaction = {
            "Time": 0,
            **{f"V{i}": 0 for i in range(1, 29)},
            "Amount": 100
        }
        transaction["V1"] = invalid_value

        with pytest.raises(ValidationError):
            TransactionInput.model_validate(transaction)
