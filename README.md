# PROJECT FORESIGHT

FORESIGHT is a demand forecasting and inventory risk analysis project built using Python.

## Features
- SKU-level demand forecasting
- Historical backtesting
- Stockout and overstock risk identification
- Streamlit dashboard
- FastAPI scoring service

## API Note
The API currently serves saved CSV results. Its forecast endpoint returns historical backtest predictions, not live or future forecasts.

## API Endpoints

Run the API locally using:

python -m uvicorn api:app --reload

Open the API documentation at http://127.0.0.1:8000/docs

- GET / — Check whether the service is running.
- GET /risk/{sku} — Get saved inventory risk details for one SKU.
- GET /forecast/{sku} — Get historical backtest forecasts for one SKU.
- GET /score/{sku} — Get both saved risk and historical forecast results.
- POST /score/batch — Get results for multiple SKUs.

The API is currently local and reads saved CSV files. It does not generate new forecasts.

## API Input and Output

Single-SKU example:
GET /score/SKU012

Batch request:
POST /score/batch

Request body:
{
  "skus": ["SKU012", "SKU045"]
}

The response contains saved risk details and historical backtest forecasts for each SKU.

Error handling:
- Unknown SKU: 404 — SKU not found
- Empty batch list: 400 — At least one SKU is required

## Run the Dashboard

Install dependencies:

pip install -r requirements.txt

Start the Streamlit dashboard:

streamlit run dashboard.py

The dashboard displays historical forecasts, inventory risk, and recommended actions using saved project outputs.

## Current Limitations

- The API runs locally and is not yet publicly hosted.
- Forecast responses contain historical backtest predictions, not future forecasts.
- The API reads saved CSV outputs rather than running the trained model for each request.
- Inventory risk results and historical backtest forecasts come from different evaluation periods.