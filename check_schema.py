import duckdb
print(duckdb.execute("DESCRIBE SELECT * FROM read_parquet('data/raw/spy/underlying_prices.parquet')").df())
