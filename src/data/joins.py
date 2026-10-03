import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def join_options_and_underlying(raw_dir: Path, interim_dir: Path):
    """
    Creates a point-in-time join between options and underlying price data.
    Builds the clean analytical table (Step 4).
    """
    logger.info("Building canonical schema (Point-In-Time join)...")
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    options_file = raw_dir / "options.parquet"
    underlying_file = raw_dir / "underlying.parquet"
    output_file = interim_dir / "joined.parquet"
    
    con = duckdb.connect()
    
    query = f"""
    COPY (
        SELECT 
            o.date,
            o.symbol,
            o.contract_id,
            o.type AS option_type,
            o.strike,
            o.expiration,
            
            u.close AS underlying_price,
            u.adjusted_close,
            u.volume AS underlying_volume,
            
            o.bid,
            o.bid_size,
            o.ask,
            o.ask_size,
            o.last,
            o.mark AS mid,
            (o.ask - o.bid) AS spread,
            CASE WHEN o.mark > 0 THEN (o.ask - o.bid) / o.mark ELSE NULL END AS spread_pct,
            
            o.volume,
            o.open_interest,
            
            o.implied_volatility AS iv,
            o.delta,
            o.gamma,
            o.theta,
            o.vega,
            o.rho,
            
            date_diff('day', o.date, o.expiration) AS dte,
            
            CASE 
                WHEN o.type = 'CALL' THEN u.close / o.strike
                ELSE o.strike / u.close 
            END AS moneyness,
            
            CASE 
                WHEN o.type = 'CALL' THEN ln(u.close / o.strike)
                ELSE ln(o.strike / u.close)
            END AS log_moneyness,
            
            CASE 
                WHEN o.type = 'CALL' THEN GREATEST(0, u.close - o.strike)
                ELSE GREATEST(0, o.strike - u.close)
            END AS intrinsic_value
            
        FROM read_parquet('{options_file}') AS o
        LEFT JOIN read_parquet('{underlying_file}') AS u
        ON o.symbol = u.symbol AND o.date = u.date
    ) TO '{output_file}' (FORMAT PARQUET);
    """
    
    con.execute(query)
    
    con.execute(f"""
    COPY (
        SELECT *,
               GREATEST(0, mid - intrinsic_value) AS extrinsic_value
        FROM read_parquet('{output_file}')
    ) TO '{output_file}' (FORMAT PARQUET);
    """)
    
    logger.info(f"Joined data written to {output_file}")
    con.close()

if __name__ == "__main__":
    join_options_and_underlying(Path("data/raw"), Path("data/interim"))
