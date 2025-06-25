# Test script to verify year filtering works with your CSV
import pandas as pd
import numpy as np

# Load your CSV (replace with actual path)
df = pd.read_csv('/Users/shijuprakash/Downloads/sales_2024.csv')

print("Original data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Sample data:")
print(df.head())

print("\nUnique years:", sorted(df['year'].unique()))
print("Unique months:", sorted(df['month'].unique()))

# Test year filtering
year_to_filter = 2024
filtered_df = df[df['year'] == year_to_filter]

print(f"\nAfter filtering for year {year_to_filter}:")
print("Filtered data shape:", filtered_df.shape)

# Calculate total sales for 2024
if 'sales' in df.columns:
    total_sales_2024 = filtered_df['sales'].sum()
    print(f"Total sales for 2024: {total_sales_2024:,}")

    # Also show by month
    monthly_sales = filtered_df.groupby('month')['sales'].sum()
    print(f"\nMonthly sales for 2024:")
    for month, sales in monthly_sales.items():
        print(f"Month {month}: {sales:,}")

if 'profit' in df.columns:
    total_profit_2024 = filtered_df['profit'].sum()
    print(f"Total profit for 2024: {total_profit_2024:,}")

# Test the regex pattern that should match "Total sum of 2024"
import re

query = "Total sum of 2024"
year_patterns = [
    r'\b(20\d{2})\b',  # 2024, 2023, etc.
    r'\byear\s+(20\d{2})\b',  # "year 2024"
    r'\bin\s+(20\d{2})\b',  # "in 2024"
    r'\bof\s+(20\d{2})\b',  # "of 2024"
    r'\bfor\s+(20\d{2})\b',  # "for 2024"
]

print(f"\nTesting regex patterns for query: '{query}'")
for i, pattern in enumerate(year_patterns):
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        print(f"Pattern {i + 1} matched: {pattern} -> Year: {match.group(1)}")
        break
else:
    print("No year pattern matched!")

# Simulate the fixed temporal expression
temporal_expression = {
    'type': 'specific_period',
    'value': '2024',
    'start_date': '2024-01-01',
    'end_date': '2024-12-31'
}

print(f"\nTemporal expression: {temporal_expression}")

# Test the filtering logic
if 'year' in df.columns:
    target_year = 2024
    filtered_test = df[df['year'] == target_year]
    print(f"Test filter result: {len(filtered_test)} rows for year {target_year}")
else:
    print("No 'year' column found!")