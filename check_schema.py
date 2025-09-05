#!/usr/bin/env python3
"""Check the actual schema of the downloaded parquet files."""

import duckdb
from pathlib import Path

# Connect to DuckDB
conn = duckdb.connect(":memory:")

# Read the parquet files
cache_dir = Path("./cache")
parquet_path = str(cache_dir / "*.parquet")

# Create view
conn.execute(f"CREATE VIEW open_data AS SELECT * FROM read_parquet('{parquet_path}')")

# Get schema
result = conn.execute("DESCRIBE open_data")
columns = result.fetchall()

print("Schema of open_data:")
print("-" * 50)
for col in columns:
    print(f"{col[0]}: {col[1]}")

print("\n" + "=" * 50)
print("\nSample of actual data (first row):")
print("-" * 50)

# Get one row to see actual data
sample = conn.execute("SELECT * FROM open_data LIMIT 1").fetchone()
col_names = [desc[0] for desc in conn.execute("SELECT * FROM open_data LIMIT 0").description]

for i, (col_name, value) in enumerate(zip(col_names, sample)):
    # Truncate long values
    str_value = str(value)
    if len(str_value) > 100:
        str_value = str_value[:100] + "..."
    print(f"{col_name}: {str_value}")

conn.close()