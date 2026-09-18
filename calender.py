import pandas as pd

calendar = pd.read_csv("calendar.csv")

print("Shape:", calendar.shape)
print("\nColumns:")
print(calendar.columns)

print("\nFirst 5 rows:")
print(calendar.head())

calendar["date"] = pd.to_datetime(calendar["date"])

print("\nData Types:")
print(calendar.dtypes)

print("\nMissing values:")
print(calendar.isnull().sum())

print("\nUnique values in is_holiday:")
print(calendar["is_holiday"].unique())

print("\nUnique values in is_weekend:")
print(calendar["is_weekend"].unique())

print("\nDuplicate dates:")
print(calendar["date"].duplicated().sum())

print("\nDate Range:")
print(calendar["date"].min(), "to", calendar["date"].max())

print("\nMissing dates:")
expected_dates = pd.date_range(
    start=calendar["date"].min(),
    end=calendar["date"].max(),
    freq="D"
)

missing_dates = expected_dates.difference(calendar["date"])

print("Number of missing dates:", len(missing_dates))
print(missing_dates)

print("\nYear consistency:")
print((calendar["date"].dt.year == calendar["year"]).all())

print("\nMonth consistency:")
print((calendar["date"].dt.month == calendar["month"]).all())

print("\nWeekend consistency:")
expected_weekend = calendar["date"].dt.dayofweek >= 5
print((expected_weekend == (calendar["is_weekend"] == 1)).all())

print("\nQuarter consistency:")

expected_quarter = ((calendar["month"] - 1) // 3 + 1).map(lambda x: f"Q{x}")

print((expected_quarter == calendar["quarter"]).all())

print("\nDay of week consistency:")

expected_day = calendar["date"].dt.day_name()

print((expected_day == calendar["day_of_week"]).all())

print("\nSeason values:")
print(calendar["season"].unique())

print("\nSeason by month:")
print(calendar.groupby("season")["month"].unique())

print("\nSeason consistency:")

expected_season = calendar["month"].map({
    1: "Winter", 2: "Winter", 12: "Winter",
    3: "Spring", 4: "Spring",
    5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Autumn", 11: "Autumn"
})

print((expected_season == calendar["season"]).all())

print("\nWeek consistency:")

expected_week = calendar["date"].dt.isocalendar().week

print((expected_week == calendar["week"]).all())

print("\nHoliday consistency:")

holiday_check = (
    ((calendar["is_holiday"] == 1) & calendar["holiday"].notna()) |
    ((calendar["is_holiday"] == 0) & calendar["holiday"].isna())
)

print(holiday_check.all())

calendar.to_csv("calendar_cleaned.csv", index=False)

