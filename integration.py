import pandas as pd

sales_daily = pd.read_csv("sales_daily.csv")
sku_master = pd.read_csv("sku_master_cleaned.csv")

sales_daily["Date"] = pd.to_datetime(sales_daily["Date"])
sku_master["Launch_Date"] = pd.to_datetime(sku_master["Launch_Date"])

sales_sku = sales_daily.merge(
    sku_master,
    on="SKU",
    how="left"
)

print("Sales Daily Shape:", sales_daily.shape)
print("SKU Master Shape:", sku_master.shape)
print("After Sales + SKU Master Merge:", sales_sku.shape)

print("\nMissing SKU Master values after merge:")
print(sales_sku["Product_Name"].isnull().sum())

calendar = pd.read_csv("calendar.csv")
calendar["date"] = pd.to_datetime(calendar["date"])

sales_sku_calendar = sales_sku.merge(
    calendar,
    left_on="Date",
    right_on="date",
    how="left"
)

print("After Sales + SKU Master + Calendar Merge:", sales_sku_calendar.shape)

print("\nMissing Calendar values after merge:")
print(sales_sku_calendar["year"].isnull().sum())

inventory = pd.read_csv("inventory_snapshots_cleaned.csv")
inventory["Snapshot_Date"] = pd.to_datetime(inventory["Snapshot_Date"])

sales_final = sales_sku_calendar.merge(
    inventory,
    left_on=["Date", "SKU"],
    right_on=["Snapshot_Date", "SKU"],
    how="left"
)

print("Final Integrated Shape:", sales_final.shape)

print("\nInventory rows matched:")
print(sales_final["Current_Stock"].notnull().sum())

print("\nInventory rows not matched:")
print(sales_final["Current_Stock"].isnull().sum())

print("\nMatched Inventory Date Range:")
print(sales_final.loc[
    sales_final["Current_Stock"].notnull(),
    "Date"
].min())

print(sales_final.loc[
    sales_final["Current_Stock"].notnull(),
    "Date"
].max())

print("\nMatched Inventory SKUs:")
print(sales_final.loc[
    sales_final["Current_Stock"].notnull(),
    "SKU"
].nunique())


print("Final Shape:", sales_final.shape)


print("\nFinal Dataset Columns:")
print(sales_final.columns.tolist())

print("\nTotal Columns:")
print(len(sales_final.columns))

sales_final = sales_final.drop(columns=["date"])

print("\nAfter removing duplicate date column:")
print("Shape:", sales_final.shape)

print("\nColumns:")
print(sales_final.columns.tolist())

sales_final.to_csv(
    "integrated_sales_dataset.csv",
    index=False
)

print("Final Shape:", sales_final.shape)
print(sales_final.head())