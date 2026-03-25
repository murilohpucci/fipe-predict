from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_valid_prediction():
    response = client.post("/predict", json={
        "anoModelo": 2010,
        "marca": "Ford",
        "modelo": "Fiesta TRAIL 1.0 8V Flex 5p",
        "anoReferencia": 2022,
        "mesReferencia": 2
    })

    assert response.status_code == 200
    data = response.json()

    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)

def test_invalid_year():
    response = client.post("/predict", json={
        "anoModelo": 1800,  # invalid (below 1950)
        "marca": "Ford",
        "modelo": "Fiesta 1.6",
        "anoReferencia": 2022,
        "mesReferencia": 5
    })

    assert response.status_code == 422

def test_invalid_marca():
    response = client.post("/predict", json={
        "anoModelo": 2015,
        "marca": "InvalidBrand",
        "modelo": "Fiesta 1.6",
        "anoReferencia": 2022,
        "mesReferencia": 5
    })

    assert response.status_code == 200
    assert "error" in response.json()

def test_model_not_found():
    response = client.post("/predict", json={
        "anoModelo": 2015,
        "marca": "Ford",
        "modelo": "Unknown Model",
        "anoReferencia": 2022,
        "mesReferencia": 5
    })

    assert response.status_code == 200
    assert "error" in response.json()
