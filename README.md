# PROJECT FORESIGHT

FORESIGHT is a demand forecasting and inventory risk analysis project built using Python.

## Features
- SKU-level demand forecasting
- Historical backtesting
- Stockout and overstock risk identification
- Streamlit dashboard
- FastAPI scoring service

## Live Dashboard

The Streamlit dashboard is publicly hosted at:

https://project-foresight-ti9iulcysfmwtel2zhbayq.streamlit.app/

The dashboard displays SKU-level demand analysis, historical backtest forecasts, inventory risk, and recommended actions.


## API Note
The API currently serves saved CSV results. Its forecast endpoint returns historical backtest predictions, not live or future forecasts.

## API Endpoints

The API is publicly hosted at https://project-foresight-uvpx.onrender.com
API documentation: https://project-foresight-uvpx.onrender.com/docs

The API reads saved CSV files and does not generate new forecasts.

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
- Forecast responses contain historical backtest predictions, not future forecasts.
- The API reads saved CSV outputs rather than running the trained model for each request.
- Inventory risk results and historical backtest forecasts come from different evaluation periods.