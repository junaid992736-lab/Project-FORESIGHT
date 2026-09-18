import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="FORESIGHT Scoring Service")


@app.get("/")
def home():
    return {"message": "FORESIGHT Scoring Service is running"}

@app.get("/risk/{sku}")
def get_risk(sku: str):
    risk_df = pd.read_csv("final_risk_table.csv")

    result = risk_df[
        risk_df["SKU"].str.upper() == sku.upper()
    ]

    if result.empty:
        raise HTTPException(status_code=404, detail="SKU not found")

    row = result.iloc[0]

    return {
        "SKU": row["SKU"],
        "Risk_Status": row["Risk_Status"],
        "Recommended_Action": row["Recommended_Action"],
        "Value_At_Stake_Rs": float(row["Value_At_Stake_Rs"])
    }

@app.get("/forecast/{sku}")
def get_forecast(sku: str):
    forecast_df = pd.read_csv("backtest_predictions.csv")

    result = forecast_df[
        forecast_df["SKU"].str.upper() == sku.upper()
    ]

    if result.empty:
        raise HTTPException(status_code=404, detail="SKU not found")

    return {
        "SKU": result.iloc[0]["SKU"],
        "Forecast_Type": "Historical backtest",
        "Forecasts": result[["Date", "Forecast"]].to_dict(orient="records")
    }

@app.get("/score/{sku}")
def get_score(sku: str):
    return {
        "Risk": get_risk(sku),
        "Forecast": get_forecast(sku)
    }

class BatchRequest(BaseModel):
    skus: list[str]


@app.post("/score/batch")
def score_batch(request: BatchRequest):
    if not request.skus:
        raise HTTPException(
            status_code=400,
            detail="At least one SKU is required"
        )

    return {
        "Results": [
            get_score(sku)
            for sku in request.skus
        ]
    }