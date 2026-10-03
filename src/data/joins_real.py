import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def join_options_and_underlying():
    logger.info("Building canonical schema for real data (Point-In-Time join)...")
    opt_file = "data/interim/spy/options_2020_2024.parquet"
    und_file = "data/interim/spy/underlying_2020_2024.parquet"
    output_file = "data/interim/spy/joined_real.parquet"
    
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
            u.volume AS underlying_volume,
            
            o.bid,
            o.ask,
            o.last,
            o.mark,
            
            o.volume,
            o.open_interest,
            
            o.implied_volatility AS iv,
            o.delta,
            o.gamma,
            o.theta,
            o.vega,
            o.rho,
            
            -- Derived
            (o.bid + o.ask) / 2.0 AS mid,
            (o.ask - o.bid) AS spread,
            CASE WHEN (o.bid + o.ask) / 2.0 > 0 THEN (o.ask - o.bid) / ((o.bid + o.ask) / 2.0) ELSE NULL END AS spread_pct,
            
            date_diff('day', o.date, o.expiration) AS dte,
            
            CASE 
                WHEN o.type = 'call' OR o.type = 'CALL' THEN ln(u.close / o.strike)
                ELSE ln(o.strike / u.close)
            END AS log_moneyness,
            
            o.bid_size,
            o.ask_size
            
        FROM read_parquet('{opt_file}') AS o
        LEFT JOIN read_parquet('{und_file}') AS u
        ON o.symbol = u.symbol AND o.date = u.date
    ) TO '{output_file}' (FORMAT PARQUET);
    """
    
    con.execute(query)
    logger.info(f"Joined real data written to {output_file}")
    con.close()

if __name__ == "__main__":
    join_options_and_underlying()
