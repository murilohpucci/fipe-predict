# 🚗 Vehicle Price Prediction API

This project implements a complete Machine Learning pipeline to predict vehicle prices based on historical data from the Brazilian FIPE table. It includes data analysis, model training, evaluation using cross-validation, and a production-ready REST API built with FastAPI.

---

## 📊 Dataset

The dataset used in this project is publicly available on Kaggle:

* https://www.kaggle.com/datasets/franckepeixoto/tabela-fipe

It contains historical vehicle pricing information, including:

* Brand (`marca`)
* Model (`modelo`)
* Model year (`anoModelo`)
* Reference month/year (`mesReferencia`, `anoReferencia`)
* Price (`valor`)

---

## 🧠 Machine Learning Model

Several regression models were trained and compared:

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* HistGradientBoosting Regressor

### 📈 Evaluation

Models were evaluated using **K-Fold Cross Validation (k=5)** to ensure robustness.

Metrics used:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² Score**

### 🏆 Final Model

The best-performing model was selected based on the evaluation metrics and trained on the full dataset.

Feature engineering included:

* Vehicle age (`idade`)
* Mean price per brand (`marca_mean_price`)
* Mean price per model (`modelo_mean_price`)
* Price variation (defaulted for inference)

The final model was serialized using `joblib` for reuse.

---

## 🚀 API (FastAPI)

A REST API was implemented using FastAPI to serve predictions.

### 🔗 Endpoint

```
POST /predict
```

### 📥 Input (JSON)

```json
{
  "anoModelo": 2015,
  "marca": "Ford",
  "modelo": "Fiesta 1.6",
  "anoReferencia": 2022,
  "mesReferencia": 5
}
```

### 📤 Output

```json
{
  "predicted_price": 35000.0
}
```

---

## ✅ Input Validation

Validation is handled using **Pydantic**, ensuring:

* Required fields are present
* Valid ranges for numerical values
* Valid categories for brand and model

---

## 📚 API Documentation

Interactive documentation is automatically generated and available at:

```
/docs
```

(Swagger UI)

---

## 🧪 Automated Tests

Automated tests were implemented using:

* `pytest`
* `httpx`

### ✔ Covered Scenarios

* Valid prediction request
* Invalid input (schema validation)
* Invalid brand/model
* Missing fields

Run tests with:

```bash
pytest -v
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the API

```bash
uvicorn app:app --reload
```

Then access:

```
http://localhost:8000/docs
```

---

## 🧩 Project Structure

```
.
├── app.py
├── model.pkl
├── data/
│   └── tabela-fipe-historico-precos.csv
├── test_api.py
├── requirements.txt
└── README.md
```

---

## 🧠 Key Considerations

* Feature consistency between training and inference was strictly maintained
* Cross-validation was used to ensure generalization
* Computational constraints were considered when selecting models
* The API was designed to be simple, robust, and easily extensible

---

## 📌 Conclusion

This project demonstrates an end-to-end Machine Learning workflow:

* Data analysis and preprocessing
* Model training and evaluation
* Deployment via REST API
* Automated testing

It provides a solid foundation for real-world ML applications involving structured data and prediction services.

---
