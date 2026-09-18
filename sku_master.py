import pandas as pd
sku_master = pd.read_csv("sku_master.csv")

print("shape is")
print(sku_master.shape)

print("ten rows are")
print(sku_master.head(10))

print("Missing values in SKU Master:")
print(sku_master.isnull().sum())

print("\nDuplicate rows:")
print(sku_master.duplicated().sum())

print("\nDuplicate SKUs:")
print(sku_master["SKU"].duplicated().sum())

print("\nData Types:")
print(sku_master.dtypes)

sku_master["Launch_Date"] = pd.to_datetime(sku_master["Launch_Date"])

sku_master["Margin_Percentage"] = (
    sku_master["Gross_Margin_Per_Unit"] / sku_master["Selling_Price"]
) * 100

sku_master["Product_Age_Days"] = (
    pd.Timestamp("2025-12-31") - sku_master["Launch_Date"]
).dt.days

print(sku_master.head(10))

print("Missing values in new columns:")
print(sku_master[["Margin_Percentage", "Product_Age_Days"]].isnull().sum())

print("\nNegative Product Age:")
print((sku_master["Product_Age_Days"] < 0).sum())

print("\nMargin Percentage range:")
print(sku_master["Margin_Percentage"].min(), "to", sku_master["Margin_Percentage"].max())

print("Final SKU Master Shape:")
print(sku_master.shape)

#sku_master.to_csv("sku_master_cleaned.csv", index=False)

print("\nSelling Price <= Cost Price:")
print((sku_master["Selling_Price"] <= sku_master["Cost_Price"]).sum())

calculated_margin = (
    sku_master["Selling_Price"] - sku_master["Cost_Price"]
)

margin_difference = (
    sku_master["Gross_Margin_Per_Unit"] - calculated_margin
).abs()

print("\nGross Margin mismatches:")
print((margin_difference > 0.01).sum())

print("Maximum difference:", margin_difference.max())

sku_master.to_csv("sku_master_cleaned.csv", index=False)