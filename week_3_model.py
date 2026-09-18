from sklearn.ensemble import HistGradientBoostingRegressor

import pandas as pd

df = pd.read_csv("integrated_sales_dataset.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)

print("Dataset Shape:", df.shape)
print("Date Range:", df["Date"].min(), "to", df["Date"].max())
print("Number of SKUs:", df["SKU"].nunique())

# ===== FEATURE ENGINEERING =====

df["lag_1"] = (
    df.groupby("SKU")["Units_Sold"]
    .shift(1)
)

df["lag_7"] = (
    df.groupby("SKU")["Units_Sold"]
    .shift(7)
)

df["rolling_7"] = (
    df.groupby("SKU")["Units_Sold"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

df["rolling_30"] = (
    df.groupby("SKU")["Units_Sold"]
    .transform(lambda x: x.shift(1).rolling(30).mean())
)

print("\nFeature Missing Values:")
print(df[["lag_1", "lag_7", "rolling_7", "rolling_30"]].isna().sum())

# ===== PREPARE MODEL DATA =====

model_data = df.dropna(
    subset=["lag_1", "lag_7", "rolling_7", "rolling_30"]
).copy()

print("\n===== MODEL DATA =====")
print("Original Rows:", len(df))
print("Model Rows:", len(model_data))
print("Rows Removed:", len(df) - len(model_data))

print(
    "Model Date Range:",
    model_data["Date"].min(),
    "to",
    model_data["Date"].max()
)

print("Number of SKUs:", model_data["SKU"].nunique())

# ===== SELECT MODEL FEATURES =====

features = [
    "lag_1",
    "lag_7",
    "rolling_7",
    "rolling_30",
    "Promotion",
    "month",
    "week",
    "is_weekend",
    "is_holiday"
]

target = "Units_Sold"

print("Features:", features)
print("Target:", target)

# ===== ROLLING-ORIGIN BACKTEST SETUP =====

backtest_dates = [
    pd.Timestamp("2025-07-17"),
    pd.Timestamp("2025-09-11"),
    pd.Timestamp("2025-11-06")
]

horizon_days = 56

print("\n===== BACKTEST ORIGINS =====")
for date in backtest_dates:
    print(date.date())

    # ===== FULL ROLLING-ORIGIN BACKTEST =====

fold_results = []

for origin in backtest_dates:

    test_end = origin + pd.Timedelta(days=horizon_days - 1)

    train_fold = model_data[
        model_data["Date"] < origin
    ].copy()

    future_fold = df[
        (df["Date"] >= origin) &
        (df["Date"] <= test_end)
    ].copy()

    X_train_fold = train_fold[features]
    y_train_fold = train_fold[target]

    model = HistGradientBoostingRegressor(
        random_state=42
    )

    model.fit(X_train_fold, y_train_fold)

    history = {}

    for sku in df["SKU"].unique():

        sku_history = (
            df[
                (df["SKU"] == sku) &
                (df["Date"] < origin)
            ]
            .sort_values("Date")["Units_Sold"]
            .tolist()
        )

        history[sku] = sku_history

    prediction_records = []

    forecast_dates = pd.date_range(
        start=origin,
        end=test_end,
        freq="D"
    )

    for forecast_date in forecast_dates:

        day_data = future_fold[
            future_fold["Date"] == forecast_date
        ].copy()

        for sku in day_data["SKU"]:

            sku_history = history[sku]

            day_data.loc[
                day_data["SKU"] == sku, "lag_1"
            ] = sku_history[-1]

            day_data.loc[
                day_data["SKU"] == sku, "lag_7"
            ] = sku_history[-7]

            day_data.loc[
                day_data["SKU"] == sku, "rolling_7"
            ] = sum(sku_history[-7:]) / 7

            day_data.loc[
                day_data["SKU"] == sku, "rolling_30"
            ] = sum(sku_history[-30:]) / 30

        day_data["Prediction"] = model.predict(
            day_data[features]
        )

        day_data["Prediction"] = day_data["Prediction"].clip(lower=0)

        prediction_records.append(
            day_data[
                ["Date", "SKU", "Units_Sold", "Prediction"]
            ]
        )

        for sku in day_data["SKU"]:

            prediction = day_data.loc[
                day_data["SKU"] == sku,
                "Prediction"
            ].iloc[0]

            history[sku].append(prediction)

    predictions_df = pd.concat(
        prediction_records,
        ignore_index=True
    )

    fold_wape = (
        (
            predictions_df["Units_Sold"] -
            predictions_df["Prediction"]
        ).abs().sum()
        /
        predictions_df["Units_Sold"].sum()
    ) * 100

    fold_bias = (
        predictions_df["Prediction"].sum()
        - predictions_df["Units_Sold"].sum()
    ) / predictions_df["Units_Sold"].sum() * 100

    fold_results.append({
        "Origin": origin.date(),
        "WAPE": fold_wape,
        "Bias": fold_bias
    })

    print(
        f"Origin {origin.date()} -> "
        f"WAPE: {fold_wape:.2f}% | "
        f"Bias: {fold_bias:.2f}%"
    )

print("\n===== ALL FOLD RESULTS =====")
for result in fold_results:
    print(result)

    # ===== OVERALL BACKTEST WAPE =====

average_wape = sum(
    result["WAPE"] for result in fold_results
) / len(fold_results)

average_bias = sum(
    result["Bias"] for result in fold_results
) / len(fold_results)

print("\n===== OVERALL BACKTEST RESULT =====")
print(f"Average Model WAPE: {average_wape:.2f}%")
print(f"Average Model Bias: {average_bias:.2f}%")

# ===== FAIR BASELINE BACKTEST =====

baseline_fold_results = []

print("\n===== SEASONAL NAIVE BACKTEST =====")

for origin in backtest_dates:

    test_end = origin + pd.Timedelta(days=horizon_days - 1)

    baseline_history = {}

    for sku in df["SKU"].unique():

        baseline_history[sku] = (
            df[
                (df["SKU"] == sku) &
                (df["Date"] < origin)
            ]
            .sort_values("Date")["Units_Sold"]
            .tolist()
        )

    baseline_predictions = []

    forecast_dates = pd.date_range(
        start=origin,
        end=test_end,
        freq="D"
    )

    for forecast_date in forecast_dates:

        day_actual = df[
            df["Date"] == forecast_date
        ]

        for sku in day_actual["SKU"]:

            sku_history = baseline_history[sku]

            prediction = sku_history[-7]

            actual = day_actual.loc[
                day_actual["SKU"] == sku,
                "Units_Sold"
            ].iloc[0]

            baseline_predictions.append({
                "Date": forecast_date,
                "SKU": sku,
                "Actual": actual,
                "Prediction": prediction
            })

            baseline_history[sku].append(prediction)

    baseline_df = pd.DataFrame(baseline_predictions)

    baseline_wape = (
        (
            baseline_df["Actual"] -
            baseline_df["Prediction"]
        ).abs().sum()
        /
        baseline_df["Actual"].sum()
    ) * 100

    baseline_fold_results.append({
        "Origin": origin.date(),
        "WAPE": baseline_wape
    })

    print(
        f"Origin {origin.date()} -> Baseline WAPE: "
        f"{baseline_wape:.2f}%"
    )

# ===== FINAL MODEL VS BASELINE COMPARISON =====

average_baseline_wape = sum(
    result["WAPE"] for result in baseline_fold_results
) / len(baseline_fold_results)

print("\n===== FINAL COMPARISON =====")
print(f"Average Model WAPE: {average_wape:.2f}%")
print(f"Average Baseline WAPE: {average_baseline_wape:.2f}%")
print(
    f"Improvement: "
    f"{average_baseline_wape - average_wape:.2f} percentage points"
)

# ===== INVENTORY DATA CHECK FOR RISK SCORING =====

inventory_cols = [
    "Current_Stock",
    "On_Order",
    "Lead_Time_Days",
    "Safety_Stock",
    "Reorder_Point",
    "Inventory_Value"
]

print("\n===== INVENTORY DATA CHECK =====")

print("Available inventory rows:")
print(df[inventory_cols].notna().all(axis=1).sum())

print("\nMissing values:")
print(df[inventory_cols].isna().sum())

# ===== LATEST INVENTORY SNAPSHOT PER SKU =====

latest_inventory = (
    df.dropna(subset=["Current_Stock"])
    .sort_values(["SKU", "Snapshot_Date"])
    .groupby("SKU")
    .tail(1)
    .copy()
)

print("\n===== LATEST INVENTORY SNAPSHOT =====")
print("Rows:", len(latest_inventory))
print("SKUs:", latest_inventory["SKU"].nunique())

print(
    latest_inventory[
        [
            "SKU",
            "Snapshot_Date",
            "Current_Stock",
            "On_Order",
            "Lead_Time_Days",
            "Safety_Stock",
            "Reorder_Point"
        ]
    ].head(10)
)

# ===== PREPARE RISK DATA =====

risk_data = latest_inventory.copy()

risk_data["Inventory_Position"] = (
    risk_data["Current_Stock"]
    + risk_data["On_Order"]
)

print("\n===== RISK DATA READY =====")
print("Rows:", len(risk_data))
print("SKUs:", risk_data["SKU"].nunique())

# ===== RISK FORECAST PERIOD =====

risk_origin = pd.Timestamp("2025-12-02")
max_lead_time = int(risk_data["Lead_Time_Days"].max())

risk_forecast_end = (
    risk_origin
    + pd.Timedelta(days=max_lead_time - 1)
)

print("\n===== RISK FORECAST PERIOD =====")
print("Forecast Start:", risk_origin.date())
print("Maximum Lead Time:", max_lead_time)
print("Forecast End:", risk_forecast_end.date())

# ===== TRAIN FINAL RISK MODEL =====

risk_train = model_data[
    model_data["Date"] < risk_origin
].copy()

X_risk_train = risk_train[features]
y_risk_train = risk_train[target]

risk_model = HistGradientBoostingRegressor(
    random_state=42
)

risk_model.fit(X_risk_train, y_risk_train)

print("\n===== FINAL RISK MODEL TRAINED =====")
print("Training Rows:", len(risk_train))
print(
    "Training Date Range:",
    risk_train["Date"].min().date(),
    "to",
    risk_train["Date"].max().date()
)
print("Features Used:", len(features))

# ===== 14-DAY RECURSIVE RISK FORECAST =====

risk_history = {}

for sku in df["SKU"].unique():

    risk_history[sku] = (
        df[
            (df["SKU"] == sku) &
            (df["Date"] < risk_origin)
        ]
        .sort_values("Date")["Units_Sold"]
        .tolist()
    )

risk_prediction_records = []

risk_forecast_dates = pd.date_range(
    start=risk_origin,
    end=risk_forecast_end,
    freq="D"
)

for forecast_date in risk_forecast_dates:

    day_data = df[
        df["Date"] == forecast_date
    ].copy()

    for sku in day_data["SKU"]:

        sku_history = risk_history[sku]

        day_data.loc[
            day_data["SKU"] == sku, "lag_1"
        ] = sku_history[-1]

        day_data.loc[
            day_data["SKU"] == sku, "lag_7"
        ] = sku_history[-7]

        day_data.loc[
            day_data["SKU"] == sku, "rolling_7"
        ] = sum(sku_history[-7:]) / 7

        day_data.loc[
            day_data["SKU"] == sku, "rolling_30"
        ] = sum(sku_history[-30:]) / 30

    day_data["Forecast_Demand"] = risk_model.predict(
        day_data[features]
    )

    day_data["Forecast_Demand"] = (
        day_data["Forecast_Demand"].clip(lower=0)
    )

    risk_prediction_records.append(
        day_data[
            ["Date", "SKU", "Forecast_Demand"]
        ]
    )

    for sku in day_data["SKU"]:

        prediction = day_data.loc[
            day_data["SKU"] == sku,
            "Forecast_Demand"
        ].iloc[0]

        risk_history[sku].append(prediction)

risk_forecast_df = pd.concat(
    risk_prediction_records,
    ignore_index=True
)

print("\n===== 14-DAY RISK FORECAST COMPLETE =====")
print("Rows:", len(risk_forecast_df))
print(
    "Date Range:",
    risk_forecast_df["Date"].min().date(),
    "to",
    risk_forecast_df["Date"].max().date()
)
print(risk_forecast_df.head(10))

# ===== FORECAST DEMAND OVER SKU LEAD TIME =====

lead_time_forecasts = []

for _, row in risk_data.iterrows():

    sku = row["SKU"]
    lead_time = int(row["Lead_Time_Days"])

    sku_forecast = (
        risk_forecast_df[
            risk_forecast_df["SKU"] == sku
        ]
        .sort_values("Date")
        .head(lead_time)
    )

    forecast_lead_time_demand = (
        sku_forecast["Forecast_Demand"].sum()
    )

    lead_time_forecasts.append(
        forecast_lead_time_demand
    )

risk_data["Forecast_Lead_Time_Demand"] = (
    lead_time_forecasts
)

print("\n===== FORECAST LEAD-TIME DEMAND =====")

print(
    risk_data[
        [
            "SKU",
            "Lead_Time_Days",
            "Inventory_Position",
            "Forecast_Lead_Time_Demand",
            "Safety_Stock"
        ]
    ].head(10)
)

# ===== FINAL FORECAST-BASED STOCKOUT RISK =====

risk_data["Forecast_Projected_Stock"] = (
    risk_data["Inventory_Position"]
    - risk_data["Forecast_Lead_Time_Demand"]
)

risk_data["Final_Stockout_Risk"] = (
    risk_data["Forecast_Projected_Stock"]
    < risk_data["Safety_Stock"]
)

print("\n===== FINAL FORECAST-BASED STOCKOUT RISK =====")

print(
    risk_data[
        [
            "SKU",
            "Inventory_Position",
            "Forecast_Lead_Time_Demand",
            "Forecast_Projected_Stock",
            "Safety_Stock",
            "Final_Stockout_Risk"
        ]
    ].head(10)
)

print("\nTotal Stockout Risk SKUs:")
print(risk_data["Final_Stockout_Risk"].sum())

# ===== STOCKOUT RISK SKU LIST =====

stockout_risk_skus = risk_data[
    risk_data["Final_Stockout_Risk"] == True
].copy()

print("\n===== STOCKOUT RISK SKUs =====")
print("Count:", len(stockout_risk_skus))

print(
    stockout_risk_skus[
        [
            "SKU",
            "Current_Stock",
            "On_Order",
            "Inventory_Position",
            "Lead_Time_Days",
            "Forecast_Lead_Time_Demand",
            "Forecast_Projected_Stock",
            "Safety_Stock"
        ]
    ]
)

# ===== 14-DAY FORECAST DEMAND PER SKU =====

forecast_14d = (
    risk_forecast_df
    .groupby("SKU")["Forecast_Demand"]
    .sum()
    .reset_index(name="Forecast_Demand_14D")
)

risk_data = risk_data.merge(
    forecast_14d,
    on="SKU",
    how="left"
)

print("\n===== 14-DAY FORECAST DEMAND =====")

print(
    risk_data[
        [
            "SKU",
            "Current_Stock",
            "On_Order",
            "Forecast_Demand_14D",
            "Safety_Stock"
        ]
    ].head(10)
)

# ===== OVERSTOCK RISK =====

risk_data["Projected_On_Hand_After_14D"] = (
    risk_data["Current_Stock"]
    - risk_data["Forecast_Demand_14D"]
)


# ===== STOCK COVERAGE RATIO =====

risk_data["Stock_Coverage_Ratio"] = (
    risk_data["Current_Stock"]
    / risk_data["Forecast_Demand_14D"]
)

print("\n===== STOCK COVERAGE RATIO =====")

print(
    risk_data[
        [
            "SKU",
            "Current_Stock",
            "Forecast_Demand_14D",
            "Stock_Coverage_Ratio"
        ]
    ]
    .sort_values(
        "Stock_Coverage_Ratio",
        ascending=False
    )
    .head(10)
)

# ===== STOCK COVERAGE DISTRIBUTION =====

print("\n===== STOCK COVERAGE DISTRIBUTION =====")

print(
    risk_data["Stock_Coverage_Ratio"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90]
    )
)

# ===== FINAL OVERSTOCK RISK =====

coverage_threshold = risk_data[
    "Stock_Coverage_Ratio"
].quantile(0.90)

risk_data["Final_Overstock_Risk"] = (
    risk_data["Stock_Coverage_Ratio"] > coverage_threshold
)

print("\n===== FINAL OVERSTOCK RISK =====")
print(f"Coverage Threshold: {coverage_threshold:.2f}")

overstock_skus = risk_data[
    risk_data["Final_Overstock_Risk"]
].sort_values(
    "Stock_Coverage_Ratio",
    ascending=False
)

print("Total Overstock Risk SKUs:", len(overstock_skus))

print(
    overstock_skus[
        [
            "SKU",
            "Current_Stock",
            "Forecast_Demand_14D",
            "Stock_Coverage_Ratio",
            "Final_Overstock_Risk"
        ]
    ]
)

# ===== FINAL RISK STATUS =====

def get_risk_status(row):

    if row["Final_Stockout_Risk"] and row["Final_Overstock_Risk"]:
        return "Watch / Volatile"

    elif row["Final_Stockout_Risk"]:
        return "Stockout Risk"

    elif row["Final_Overstock_Risk"]:
        return "Overstock Risk"

    else:
        return "Healthy"


risk_data["Risk_Status"] = risk_data.apply(
    get_risk_status,
    axis=1
)

print("\n===== FINAL RISK STATUS =====")

print(
    risk_data[
        [
            "SKU",
            "Final_Stockout_Risk",
            "Final_Overstock_Risk",
            "Risk_Status"
        ]
    ].head(15)
)

print("\nRisk Status Counts:")
print(risk_data["Risk_Status"].value_counts())

# ===== RECOMMENDED ACTION =====

action_map = {
    "Stockout Risk": "Reorder Now",
    "Overstock Risk": "Markdown / Clear",
    "Watch / Volatile": "Investigate Manually",
    "Healthy": "No Action"
}

risk_data["Recommended_Action"] = (
    risk_data["Risk_Status"].map(action_map)
)

print("\n===== RECOMMENDED ACTIONS =====")

print(
    risk_data[
        [
            "SKU",
            "Risk_Status",
            "Recommended_Action"
        ]
    ].head(15)
)

print("\nAction Counts:")
print(risk_data["Recommended_Action"].value_counts())



# ===== STOCKOUT SALES AT RISK =====

risk_data["Shortage_Units"] = (
    risk_data["Forecast_Lead_Time_Demand"]
    - risk_data["Inventory_Position"]
).clip(lower=0)

risk_data["Sales_At_Risk_Rs"] = (
    risk_data["Shortage_Units"]
    * risk_data["Selling_Price"]
)

print("\n===== STOCKOUT SALES AT RISK =====")

print(
    risk_data[
        [
            "SKU",
            "Final_Stockout_Risk",
            "Shortage_Units",
            "Selling_Price",
            "Sales_At_Risk_Rs"
        ]
    ]
    .sort_values(
        "Sales_At_Risk_Rs",
        ascending=False
    )
    .head(10)
)

# ===== OVERSTOCK CAPITAL LOCKED =====

risk_data["Excess_Stock_Units"] = (
    risk_data["Projected_On_Hand_After_14D"]
    - risk_data["Safety_Stock"]
).clip(lower=0)

risk_data["Capital_Locked_Rs"] = (
    risk_data["Excess_Stock_Units"]
    * risk_data["Cost_Price"]
)

print("\n===== OVERSTOCK CAPITAL LOCKED =====")

print(
    risk_data[
        [
            "SKU",
            "Final_Overstock_Risk",
            "Excess_Stock_Units",
            "Cost_Price",
            "Capital_Locked_Rs"
        ]
    ]
    .sort_values(
        "Capital_Locked_Rs",
        ascending=False
    )
    .head(10)
)

# ===== FINAL VALUE AT STAKE =====

risk_data["Final_Capital_Locked_Rs"] = 0.0

risk_data.loc[
    risk_data["Final_Overstock_Risk"],
    "Final_Capital_Locked_Rs"
] = risk_data.loc[
    risk_data["Final_Overstock_Risk"],
    "Capital_Locked_Rs"
]

print("\n===== FINAL VALUE AT STAKE =====")

print(
    risk_data[
        [
            "SKU",
            "Risk_Status",
            "Recommended_Action",
            "Sales_At_Risk_Rs",
            "Final_Capital_Locked_Rs"
        ]
    ]
    .query("Risk_Status != 'Healthy'")
    .sort_values(
        ["Sales_At_Risk_Rs", "Final_Capital_Locked_Rs"],
        ascending=False
    )
)

# ===== FINAL BUSINESS RISK TABLE =====

risk_data["Value_At_Stake_Rs"] = (
    risk_data["Sales_At_Risk_Rs"]
    + risk_data["Final_Capital_Locked_Rs"]
)

final_risk_table = risk_data[
    [
        "SKU",
        "Risk_Status",
        "Recommended_Action",
        "Value_At_Stake_Rs"
    ]
].copy()

print("\n===== FINAL BUSINESS RISK TABLE =====")

print(
    final_risk_table[
        final_risk_table["Risk_Status"] != "Healthy"
    ]
    .sort_values(
        "Value_At_Stake_Rs",
        ascending=False
    )
)

# ===== DECISIONING GRID SCORES =====

risk_data["Stockout_Score"] = (
    (
        risk_data["Forecast_Lead_Time_Demand"]
        + risk_data["Safety_Stock"]
    )
    / risk_data["Inventory_Position"]
)

risk_data["Overstock_Score"] = (
    risk_data["Stock_Coverage_Ratio"]
    / coverage_threshold
)

print("\n===== DECISIONING GRID SCORES =====")

print(
    risk_data[
        [
            "SKU",
            "Stockout_Score",
            "Overstock_Score",
            "Risk_Status"
        ]
    ].head(10)
)
# ===== DECISIONING GRID PLOT =====

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

for status in risk_data["Risk_Status"].unique():

    subset = risk_data[
        risk_data["Risk_Status"] == status
    ]

    plt.scatter(
        subset["Overstock_Score"],
        subset["Stockout_Score"],
        label=status
    )

plt.axhline(
    y=1,
    linestyle="--"
)

plt.axvline(
    x=1,
    linestyle="--"
)

plt.xlabel("Overstock Score")
plt.ylabel("Stockout Score")
plt.title("SKU Inventory Risk Decisioning Grid")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "decisioning_grid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===== SAVE FINAL RISK OUTPUT =====

final_risk_table.to_csv(
    "final_risk_table.csv",
    index=False
)

# ===== SAVE BACKTEST PREDICTIONS FOR DASHBOARD =====

backtest_dashboard = predictions_df[
    ["Date", "SKU", "Units_Sold", "Prediction"]
].copy()

backtest_dashboard = backtest_dashboard.rename(
    columns={
        "Units_Sold": "Actual",
        "Prediction": "Forecast"
    }
)

backtest_dashboard.to_csv(
    "backtest_predictions.csv",
    index=False
)

risk_data[
    ["SKU", "Category", "Stockout_Score",
     "Overstock_Score", "Risk_Status"]
].to_csv("decisioning_grid_data.csv", index=False)