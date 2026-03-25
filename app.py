from fastapi import FastAPI
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pydantic import BaseModel, Field

model = joblib.load("model.pkl")
FEATURE_COLUMNS = [
    "anoModelo",
    "price_variation",
    "idade",
    "marca_mean_price",
    "modelo_mean_price"
]
print("Model loaded successfully!")

class VehicleInput(BaseModel):
    anoModelo: int = Field(..., ge=1950, le=2026)
    marca: str
    modelo: str
    anoReferencia: int
    mesReferencia: int

df = pd.read_csv("data/tabela-fipe-historico-precos.csv")

valid_marcas = set(df["marca"].unique())
valid_modelos = set(df["modelo"].unique())
valid_anos = set(df["anoModelo"].unique())

app = FastAPI()

@app.post("/predict")
def predict(data: VehicleInput):
    print(model.feature_names_in_)
    
    # Validate values
    if data.marca not in valid_marcas:
        return {"error": "Marca inválida"}
    
    if data.modelo not in valid_modelos:
        return {"error": "Modelo não encontrado"}
    
    if data.anoModelo not in valid_anos:
        return {"error": "Ano do modelo inválido"}

    # --- Feature engineering (same as training!) ---
    idade = data.anoReferencia - data.anoModelo

    marca_mean = df.groupby("marca")["valor"].mean().to_dict()
    modelo_mean = df.groupby("modelo")["valor"].mean().to_dict()

    marca_mean_price = marca_mean.get(data.marca, df["valor"].mean())
    modelo_mean_price = modelo_mean.get(data.modelo, df["valor"].mean())

    # Build input
    input_data = pd.DataFrame([{
        "anoModelo": data.anoModelo,
        "price_variation": 0,  # default
        "idade": idade,
        "marca_mean_price": marca_mean_price,
        "modelo_mean_price": modelo_mean_price
    }])

    # Ensure same column order
    input_data = input_data[FEATURE_COLUMNS]

    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": float(prediction)
    }