import os
import logging
import urllib.request
import duckdb
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_and_filter_real_data():
    raw_dir = Path("data/raw/spy")
    interim_dir = Path("data/interim/spy")
    
    opt_files = [f"data/raw/spy/options_{y}.parquet" for y in range(2020, 2025)]
    und_raw_path = "data/raw/spy/underlying_prices.parquet"
    
    opt_interim = interim_dir / "options_2020_2024.parquet"
    und_interim = interim_dir / "underlying_2020_2024.parquet"
    
    con = duckdb.connect()
    files_str = ", ".join([f"'{f}'" for f in opt_files])
    
    # Cast date and expiration to actual DATE types
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (date, expiration),
                   CAST(date AS DATE) AS date,
                   CAST(expiration AS DATE) AS expiration
            FROM read_parquet([{files_str}])
            WHERE CAST(date AS DATE) >= DATE '2020-01-01' AND CAST(date AS DATE) <= DATE '2024-12-31'
        ) TO '{opt_interim}' (FORMAT PARQUET)
    """)
    logger.info(f"Saved correctly typed interim options data to {opt_interim}")
    
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (date),
                   CAST(date AS DATE) AS date
            FROM read_parquet('{und_raw_path}')
            WHERE CAST(date AS DATE) >= DATE '2020-01-01' AND CAST(date AS DATE) <= DATE '2024-12-31'
        ) TO '{und_interim}' (FORMAT PARQUET)
    """)
    logger.info(f"Saved correctly typed interim underlying data to {und_interim}")
    con.close()

if __name__ == "__main__":
    download_and_filter_real_data()
