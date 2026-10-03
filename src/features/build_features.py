import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_all_features(interim_dir: Path, processed_dir: Path):
    """
    Executes feature engineering (Underlying, Volatility, Options) on the joined dataset.
    Uses DuckDB window functions to strictly avoid lookahead bias.
    """
    logger.info("Building features (Underlying, Volatility, Options)...")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    joined_file = interim_dir / "joined.parquet"
    output_file = processed_dir / "features.parquet"
    
    con = duckdb.connect()
    
    # 1. Base returns
    con.execute(f"""
    CREATE TEMP VIEW underlying_base AS
    SELECT DISTINCT symbol, date, underlying_price AS close, underlying_volume AS volume,
           close / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) - 1 AS return_1d,
           close / LAG(close, 5) OVER (PARTITION BY symbol ORDER BY date) - 1 AS return_5d,
           close / LAG(close, 10) OVER (PARTITION BY symbol ORDER BY date) - 1 AS return_10d,
           close / LAG(close, 20) OVER (PARTITION BY symbol ORDER BY date) - 1 AS return_20d,
           (close - LAG(close, 20) OVER (PARTITION BY symbol ORDER BY date)) / LAG(close, 20) OVER (PARTITION BY symbol ORDER BY date) AS momentum_20d
    FROM read_parquet('{joined_file}');
    """)
    
    # 2. Add volatility
    con.execute("""
    CREATE TEMP VIEW underlying_daily AS
    SELECT *,
           stddev_samp(return_1d) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) * sqrt(252) AS rv_20
    FROM underlying_base;
    """)
    
    # 3. Join back to options and create option-specific features
    query = f"""
    COPY (
        SELECT 
            j.*,
            ud.return_1d,
            ud.return_5d,
            ud.return_10d,
            ud.return_20d,
            ud.momentum_20d,
            ud.rv_20,
            
            -- Volatility Features
            j.iv - ud.rv_20 AS iv_minus_rv20,
            CASE WHEN ud.rv_20 > 0 THEN j.iv / ud.rv_20 ELSE NULL END AS iv_rv_ratio,
            
            -- Option Features
            ABS(j.log_moneyness) AS abs_log_moneyness,
            CASE WHEN j.intrinsic_value > 0 THEN 1 ELSE 0 END AS itm_flag,
            
            -- Greeks ratios
            ABS(j.delta) AS abs_delta,
            CASE WHEN j.vega > 0 THEN j.gamma / j.vega ELSE NULL END AS gamma_to_vega,
            CASE WHEN j.vega > 0 THEN j.theta / j.vega ELSE NULL END AS theta_to_vega,
            
            -- Liquidity features
            CASE WHEN (j.bid_size + j.ask_size) > 0 THEN (j.bid_size - j.ask_size) / (j.bid_size + j.ask_size) ELSE 0 END AS quote_imbalance
            
        FROM read_parquet('{joined_file}') AS j
        LEFT JOIN underlying_daily ud
        ON j.symbol = ud.symbol AND j.date = ud.date
    ) TO '{output_file}' (FORMAT PARQUET);
    """
    
    con.execute(query)
    logger.info(f"Features generated and saved to {output_file}")
    con.close()

if __name__ == "__main__":
    build_all_features(Path("data/interim"), Path("data/processed"))
