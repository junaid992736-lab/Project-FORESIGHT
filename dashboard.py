import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Project Foresight Dashboard",
    page_icon="📊",
    layout="wide"
)
# ===== DASHBOARD STYLING =====

st.markdown("""
<style>

/* Main dashboard background */
.stApp {
    background-color: #F5F7FA;
}

/* Main headings */
h1 {
    color: #16324F;
    font-weight: 800;
}

h2, h3 {
    color: #234E70;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E1E7EF;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #52606D;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #16324F;
    font-weight: 700;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #D9E2EC;
}

</style>
""", unsafe_allow_html=True)
st.title("PROJECT FORESIGHT")
st.subheader("Demand Forecasting & Inventory Risk Dashboard")

st.write(
    "Planning dashboard for SKU-level demand forecast, "
    "stockout/overstock risk, and recommended actions."
)

# Load final SKU-level risk results
risk_df = pd.read_csv("final_risk_table.csv")


# ===== COLORED KPI CARDS =====

st.markdown("""
<style>

/* Total SKUs - Blue */
[data-testid="stMetric"]:nth-of-type(1) {
    border-top: 5px solid #2563EB;
}

/* Healthy - Green */
[data-testid="stMetric"]:nth-of-type(2) {
    border-top: 5px solid #16A34A;
}

/* Stockout Risk - Red */
[data-testid="stMetric"]:nth-of-type(3) {
    border-top: 5px solid #DC2626;
}

/* Overstock Risk - Orange */
[data-testid="stMetric"]:nth-of-type(4) {
    border-top: 5px solid #F59E0B;
}

</style>
""", unsafe_allow_html=True)

# Load integrated dataset for filters
sales_df = pd.read_csv("integrated_sales_dataset.csv")

st.divider()
st.header("SKU Filters")

categories = sorted(sales_df["Category"].dropna().unique())

selected_category = st.selectbox(
    "Select Category",
    ["All"] + categories
)

if selected_category == "All":
    sku_options = sorted(sales_df["SKU"].unique())
else:
    sku_options = sorted(
        sales_df.loc[
            sales_df["Category"] == selected_category,
            "SKU"
        ].unique()
    )

selected_sku = st.selectbox(
    "Select SKU",
    sku_options
)

st.divider()

if selected_category == "All":
    overview_df = risk_df.copy()
else:
    category_skus = sales_df.loc[
        sales_df["Category"] == selected_category,
        "SKU"
    ].unique()

    overview_df = risk_df[
        risk_df["SKU"].isin(category_skus)
    ].copy()

st.header("Inventory Risk Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total SKUs", len(overview_df))
col2.metric(
    "Healthy",
    (overview_df["Risk_Status"] == "Healthy").sum()
)
col3.metric(
    "Stockout Risk",
    (overview_df["Risk_Status"] == "Stockout Risk").sum()
)
col4.metric(
    "Overstock Risk",
    (overview_df["Risk_Status"] == "Overstock Risk").sum()
)

st.divider()
st.header("Priority Action List")

priority_df = risk_df[
    risk_df["Risk_Status"] != "Healthy"
].copy()

if selected_category != "All":
    category_skus = sales_df.loc[
        sales_df["Category"] == selected_category,
        "SKU"
    ].unique()

    priority_df = priority_df[
        priority_df["SKU"].isin(category_skus)
    ]

priority_df = priority_df.sort_values(
    "Value_At_Stake_Rs",
    ascending=False
)

if priority_df.empty:
    st.info("No stockout or overstock risks in this category.")
else:
    st.dataframe(
        priority_df,
        use_container_width=True,
        hide_index=True
    )

st.write("Selected SKU:", selected_sku)

# ===== SELECTED SKU RISK DETAILS =====

selected_risk = risk_df[
    risk_df["SKU"] == selected_sku
]

st.subheader("Selected SKU Risk Details")

if not selected_risk.empty:

    risk_row = selected_risk.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Risk Status",
        risk_row["Risk_Status"]
    )

    col2.metric(
        "Recommended Action",
        risk_row["Recommended_Action"]
    )

    col3.metric(
        "Value at Stake",
        f"₹{risk_row['Value_At_Stake_Rs']:,.0f}"
    )

else:
    st.warning("Risk information is not available for this SKU.")

# ===== SELECTED SKU SALES HISTORY =====

st.subheader("Selected SKU Demand History")

sku_history = sales_df[
    sales_df["SKU"] == selected_sku
].copy()

sku_history["Date"] = pd.to_datetime(
    sku_history["Date"]
)

sku_history = sku_history.sort_values("Date")

st.line_chart(
    sku_history.set_index("Date")["Units_Sold"]
)

# ===== FORECAST VS ACTUAL =====

st.divider()
st.subheader("Forecast vs Actual Demand")

forecast_df = pd.read_csv("backtest_predictions.csv")

forecast_df["Date"] = pd.to_datetime(
    forecast_df["Date"]
)

sku_forecast = forecast_df[
    forecast_df["SKU"] == selected_sku
].copy()

sku_forecast = sku_forecast.sort_values("Date")

forecast_chart = sku_forecast[
    ["Date", "Actual", "Forecast"]
].set_index("Date")

if forecast_chart.empty:
    st.info("Forecast data is not available for this SKU.")
else:
    st.line_chart(forecast_chart)

# ===== INVENTORY RISK DECISIONING GRID =====

st.divider()
st.subheader("Inventory Risk Decisioning Grid")

# ===== INTERACTIVE DECISIONING GRID =====

import plotly.express as px

grid_df = pd.read_csv("decisioning_grid_data.csv")


if selected_category != "All":
    grid_df = grid_df[
        grid_df["Category"] == selected_category
    ]

grid_df["Selection"] = grid_df["SKU"].apply(
    lambda sku: "Selected SKU" if sku == selected_sku else "Other SKUs"
)

fig = px.scatter(
    grid_df,
    x="Overstock_Score",
    y="Stockout_Score",
    color="Risk_Status",
    hover_name="SKU",
    hover_data=["Category"],
    title="SKU Inventory Risk Decisioning Grid"
)

# ===== HIGHLIGHT SELECTED SKU =====

selected_point = grid_df[
    grid_df["SKU"] == selected_sku
]

fig.add_scatter(
    x=selected_point["Overstock_Score"],
    y=selected_point["Stockout_Score"],
    mode="markers",
    marker=dict(
        symbol="star",
        size=8,
        color="gold",
        line=dict(color="black", width=2)
    ),
    name=f"Selected: {selected_sku}",
    hovertext=selected_point["SKU"],
    hoverinfo="text"
)


fig.add_hline(y=1, line_dash="dash")
fig.add_vline(x=1, line_dash="dash")

st.plotly_chart(fig, width="stretch")