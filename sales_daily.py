import pandas as pd

sales_daily = pd.read_csv("sales_daily.csv")

print("Shape:", sales_daily.shape)

print("\nColumns:")
print(sales_daily.columns)

print("\nData Types:")
print(sales_daily.dtypes)

print("\nMissing Values:")
print(sales_daily.isnull().sum())

print("\nDuplicate Rows:")
print(sales_daily.duplicated().sum())

print("\nFirst 5 Rows:")
print(sales_daily.head())

sales_daily["Date"] = pd.to_datetime(sales_daily["Date"])

print("Date data type:", sales_daily["Date"].dtype)
print("Minimum Date:", sales_daily["Date"].min())
print("Maximum Date:", sales_daily["Date"].max())

print("Zero Units Sold:", (sales_daily["Units_Sold"] == 0).sum())
print("Negative Units Sold:", (sales_daily["Units_Sold"] < 0).sum())

print("Zero Revenue:", (sales_daily["Revenue"] == 0).sum())
print("Negative Revenue:", (sales_daily["Revenue"] < 0).sum())

print("\nUnits Sold statistics:")
print(sales_daily["Units_Sold"].describe())

print("\nRevenue statistics:")
print(sales_daily["Revenue"].describe())

expected_revenue = sales_daily["Units_Sold"] * sales_daily["Price"]

difference = (sales_daily["Revenue"] - expected_revenue).abs()

print("Revenue mismatches:", (difference > 0.01).sum())
print("Maximum difference:", difference.max())

print("Promotion values:")
print(sales_daily["Promotion"].value_counts(dropna=False))

print("\nInvalid Promotion values:")
print(sales_daily.loc[~sales_daily["Promotion"].isin([0, 1]), "Promotion"].unique())

print("Unique SKUs:", sales_daily["SKU"].nunique())

print("\nMissing SKU values:", sales_daily["SKU"].isnull().sum())

print("\nSample SKU values:")
print(sales_daily["SKU"].unique()[:20])

sku_master = pd.read_csv("sku_master.csv")

print("SKU Master Shape:", sku_master.shape)

print("\nSKU Master Columns:")
print(sku_master.columns)

missing_skus = set(sales_daily["SKU"]) - set(sku_master["SKU"])

print("\nSKUs missing from SKU Master:", len(missing_skus))
print(missing_skus)

print("\nDuplicate Date + SKU combinations:")

duplicate_date_sku = sales_daily.duplicated(
    subset=["Date", "SKU"]
).sum()

print(duplicate_date_sku)

duplicate_date_sku = sales_daily.duplicated(
    subset=["Date", "SKU"]
).sum()

print("Duplicate Date + SKU combinations:", duplicate_date_sku)

calendar = pd.read_csv("calendar.csv")
calendar["date"] = pd.to_datetime(calendar["date"])

missing_dates = set(sales_daily["Date"].dt.date) - set(calendar["date"].dt.date)

print("Sales dates missing from Calendar:", len(missing_dates))
print(missing_dates)


calendar["date"] = pd.to_datetime(calendar["date"])

merged_check = sales_daily.merge(
    calendar[["date", "promotion_event"]],
    left_on="Date",
    right_on="date",
    how="left"
)

print("Promotion = 1 with no Calendar promotion event:",
      ((merged_check["Promotion"] == 1) &
       (merged_check["promotion_event"].isna())).sum())