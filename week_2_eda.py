import pandas as pd

df = pd.read_csv("integrated_sales_dataset.csv")

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\n===== DEMAND DISTRIBUTION =====")

print("\nUnits Sold Statistics:")
print(df["Units_Sold"].describe())

print("\nZero Sales Records:")
print((df["Units_Sold"] == 0).sum())

print("\nTotal Units Sold:")
print(df["Units_Sold"].sum())

print("\nAverage Daily/SKU Demand:")
print(df["Units_Sold"].mean())

print("\nMedian Demand:")
print(df["Units_Sold"].median())

print("\nMinimum Demand:")
print(df["Units_Sold"].min())

print("\nMaximum Demand:")
print(df["Units_Sold"].max())

print("\n===== TOP MOVERS =====")

top_movers = (
    df.groupby("SKU")["Units_Sold"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 SKUs by Total Units Sold:")
print(top_movers.head(10))

print("\nBottom 10 SKUs by Total Units Sold:")
print(top_movers.tail(10))

print("\n===== DEAD STOCK ANALYSIS =====")

sku_demand = (
    df.groupby("SKU")["Units_Sold"]
    .agg(["sum", "mean", "count"])
    .sort_values("sum")
)

print("\nSKUs with zero total sales:")
print(sku_demand[sku_demand["sum"] == 0])

print("\nNumber of SKUs with zero total sales:")
print((sku_demand["sum"] == 0).sum())

print("\nBottom 10 SKUs by total demand:")
print(sku_demand.head(10))

print("\n===== CORRELATION ANALYSIS =====")

numeric_columns = [
    "Units_Sold",
    "Price",
    "Promotion",
    "Current_Stock",
    "On_Order",
    "Lead_Time_Days",
    "Safety_Stock",
    "Reorder_Point",
    "Inventory_Value",
    "Margin_Percentage",
    "Product_Age_Days"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation with Units_Sold:")
print(
    correlation["Units_Sold"]
    .sort_values(ascending=False)
)

# ===== SEASONALITY ANALYSIS =====

print("\n===== MONTHLY DEMAND ANALYSIS =====")

monthly_demand = (
    df.groupby(['year', 'month'])['Units_Sold']
    .sum()
    .reset_index()
)

print(monthly_demand.to_string(index=False))

# ===== SEASONAL DEMAND ANALYSIS =====

print("\n===== SEASONAL DEMAND ANALYSIS =====")

seasonal_demand = (
    df.groupby('season')['Units_Sold']
    .agg(['sum', 'mean', 'count'])
    .sort_values('sum', ascending=False)
)

print(seasonal_demand)

# ===== PROMOTION EFFECT ANALYSIS =====

print("\n===== PROMOTION EFFECT ON DEMAND =====")

promotion_demand = (
    df.groupby('Promotion')['Units_Sold']
    .agg(['sum', 'mean', 'count'])
)

print(promotion_demand)

# ===== CATEGORY DEMAND ANALYSIS =====

print("\n===== CATEGORY DEMAND ANALYSIS =====")

category_demand = (
    df.groupby('Category')['Units_Sold']
    .agg(['sum', 'mean', 'count'])
    .sort_values('sum', ascending=False)
)

print(category_demand)

# ===== WEEKEND EFFECT ANALYSIS =====

print("\n===== WEEKEND VS WEEKDAY DEMAND =====")

weekend_demand = (
    df.groupby('is_weekend')['Units_Sold']
    .agg(['sum', 'mean', 'count'])
)

print(weekend_demand)

# ===== DAY OF WEEK DEMAND ANALYSIS =====

print("\n===== DAY OF WEEK DEMAND =====")

day_demand = (
    df.groupby('day_of_week')['Units_Sold']
    .agg(['sum', 'mean', 'count'])
)

print(day_demand)

# ===== HOLIDAY EFFECT ANALYSIS =====

print("\n===== HOLIDAY VS NON-HOLIDAY DEMAND =====")

holiday_demand = (
    df.groupby('is_holiday')['Units_Sold']
    .agg(['sum', 'mean', 'count'])
)

print(holiday_demand)

# ===== CALENDAR PROMOTION EVENT EFFECT =====

print("\n===== PROMOTION EVENT VS NON-EVENT DEMAND =====")

promo_event_demand = (
    df.groupby('promotion_event')['Units_Sold']
    .agg(['sum', 'mean', 'count'])
)

print(promo_event_demand)

# Lag-1 demand: previous day's sales for the same SKU

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(['SKU', 'Date']).reset_index(drop=True)

df['lag_1'] = (
    df.groupby('SKU')['Units_Sold']
    .shift(1)
)

print("\n===== LAG 1 FEATURE =====")
print(df[['Date', 'SKU', 'Units_Sold', 'lag_1']].head(15))

# ===== LAG 7 FEATURE =====

df['lag_7'] = (
    df.groupby('SKU')['Units_Sold']
    .shift(7)
)

print("\n===== LAG 7 FEATURE =====")
print(df[['Date', 'SKU', 'Units_Sold', 'lag_1', 'lag_7']].head(15))

# ===== 7-DAY ROLLING AVERAGE =====

df['rolling_7'] = (
    df.groupby('SKU')['Units_Sold']
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

print("\n===== 7-DAY ROLLING AVERAGE =====")
print(df[['Date', 'SKU', 'Units_Sold', 'lag_1', 'lag_7', 'rolling_7']].head(15))

# ===== 30-DAY ROLLING AVERAGE =====

df['rolling_30'] = (
    df.groupby('SKU')['Units_Sold']
    .transform(lambda x: x.shift(1).rolling(30).mean())
)

print("\n===== 30-DAY ROLLING AVERAGE =====")
print(df[['Date', 'SKU', 'Units_Sold', 'lag_1', 'lag_7', 'rolling_7', 'rolling_30']].head(35))

# ===== SEASONAL NAIVE BASELINE =====

df = df.sort_values(["SKU", "Date"])

# Previous week's demand (same weekday)
df["seasonal_naive_7"] = df.groupby("SKU")["Units_Sold"].shift(7)

print("\n===== SEASONAL NAIVE BASELINE =====")
print(df[["Date", "SKU", "Units_Sold", "seasonal_naive_7"]].head(15))

# ===== WAPE CALCULATION =====

print("\n===== SEASONAL NAIVE WAPE =====")

baseline_data = df.dropna(subset=['seasonal_naive_7'])

wape = (
    baseline_data['Units_Sold'] - baseline_data['seasonal_naive_7']
).abs().sum() / baseline_data['Units_Sold'].sum()

print("WAPE:", wape)
print("WAPE (%):", wape * 100)

# ===== BASELINE SUMMARY =====

print("\n===== BASELINE SUMMARY =====")
print("Model: Seasonal Naive (7-day)")
print("Metric: WAPE")
print(f"WAPE: {wape * 100:.2f}%")