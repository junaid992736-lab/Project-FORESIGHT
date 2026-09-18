import pandas as pd

inventory_snapshots = pd.read_csv("inventory_snapshots.csv")



print("Shape:")
print(inventory_snapshots.shape)

print("\nColumns:")
print(inventory_snapshots.columns.tolist())

print("\nFirst 10 rows:")
print(inventory_snapshots.head(10))

print("Missing values:")
print(inventory_snapshots.isnull().sum())

print("\nDuplicate rows:")
print(inventory_snapshots.duplicated().sum())

print("\nData Types:")
print(inventory_snapshots.dtypes)

inventory_snapshots["Snapshot_Date"] = pd.to_datetime(
    inventory_snapshots["Snapshot_Date"]
)

print(inventory_snapshots.dtypes)

print("Negative Current Stock:")
print((inventory_snapshots["Current_Stock"] < 0).sum())

print("\nNegative On Order:")
print((inventory_snapshots["On_Order"] < 0).sum())

print("\nInvalid Lead Time:")
print((inventory_snapshots["Lead_Time_Days"] <= 0).sum())

print("\nNegative Safety Stock:")
print((inventory_snapshots["Safety_Stock"] < 0).sum())

print("\nNegative Reorder Point:")
print((inventory_snapshots["Reorder_Point"] < 0).sum())

print("Duplicate SKU + Snapshot Date combinations:")
print(
    inventory_snapshots.duplicated(
        subset=["Snapshot_Date", "SKU"]
    ).sum()
)

sku_master = pd.read_csv("sku_master_cleaned.csv")

missing_skus = set(inventory_snapshots["SKU"]) - set(sku_master["SKU"])

print("SKUs missing from SKU Master:")
print(len(missing_skus))
print(missing_skus)

print("Unique SKUs in Inventory:")
print(inventory_snapshots["SKU"].nunique())

print("\nUnique SKUs in SKU Master:")
print(sku_master["SKU"].nunique())

print("\nSKU Master range:")
print(sku_master["SKU"].min(),sku_master["SKU"].max())

print("\nInventory SKU range:")
print(inventory_snapshots["SKU"].min(),inventory_snapshots["SKU"].max())

print("\nExtra Inventory SKU Analysis:")

extra_inventory = inventory_snapshots[
    inventory_snapshots["SKU"].isin(missing_skus)
]

print("Rows belonging to extra SKUs:", len(extra_inventory))

print("Date range of extra SKUs:")
print(extra_inventory["Snapshot_Date"].min())
print(extra_inventory["Snapshot_Date"].max())

print("Extra SKU count:")
print(extra_inventory["SKU"].nunique())

# Keep only inventory records whose SKU exists in SKU Master

inventory_cleaned = inventory_snapshots[
    inventory_snapshots["SKU"].isin(sku_master["SKU"])
].copy()

print("\nFiltered Inventory Shape:")
print(inventory_cleaned.shape)

print("\nUnique SKUs after filtering:")
print(inventory_cleaned["SKU"].nunique())

print("\nRemaining missing SKUs:")
print(
    set(inventory_cleaned["SKU"]) -
    set(sku_master["SKU"])
)

inventory_cleaned.to_csv(
    "inventory_snapshots_cleaned.csv",
    index=False
)

print("\nCleaned inventory file saved.")

