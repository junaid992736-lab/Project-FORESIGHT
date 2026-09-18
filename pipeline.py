import subprocess
import sys

print("Starting Week 1 pipeline...\n")

scripts = [
    "sales_daily.py",
    "sku_master.py",
    "calender.py",
    "inventory_snapshots.py",
    "integration.py"
]

for script in scripts:
    print(f"\nRunning {script}...")
    
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(f"\nERROR: {script} failed.")
        print(result.stderr)
        sys.exit(1)

print("\nWeek 1 pipeline completed successfully.")
print("Final file: integrated_sales_dataset.csv")