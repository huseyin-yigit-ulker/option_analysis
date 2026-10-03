import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_targets(processed_dir: Path):
    """
    Creates future-return targets (Step 6) using conservative execution assumptions (Ask -> Bid).
    Uses window functions over contract_id to look forward 5 trading days.
    """
    logger.info("Building prediction targets...")
    features_file = processed_dir / "features.parquet"
    output_file = processed_dir / "dataset.parquet"
    
    con = duckdb.connect()
    
    query = f"""
    COPY (
        WITH forward_data AS (
            SELECT 
                contract_id,
                date,
                
                -- Target A: Future option return (Conservative: Buy at Ask, Sell at Bid 5 days later)
                -- We use LEAD(bid, 5) meaning the bid price 5 trading days in the future
                LEAD(bid, 5) OVER (PARTITION BY contract_id ORDER BY date) AS future_bid_5d,
                
                -- Also calculate 1d and 10d just to have them available
                LEAD(bid, 1) OVER (PARTITION BY contract_id ORDER BY date) AS future_bid_1d,
                LEAD(bid, 10) OVER (PARTITION BY contract_id ORDER BY date) AS future_bid_10d,
                
                -- Target B: Future underlying direction
                LEAD(underlying_price, 5) OVER (PARTITION BY contract_id ORDER BY date) AS future_underlying_5d
                
            FROM read_parquet('{features_file}')
        )
        SELECT 
            f.*,
            
            -- Calculate actual returns. If ask == 0, avoid division by zero
            CASE WHEN f.ask > 0 AND fd.future_bid_5d IS NOT NULL 
                 THEN (fd.future_bid_5d - f.ask) / f.ask 
                 ELSE NULL END AS target_return_5d,
                 
            CASE WHEN f.ask > 0 AND fd.future_bid_1d IS NOT NULL 
                 THEN (fd.future_bid_1d - f.ask) / f.ask 
                 ELSE NULL END AS target_return_1d,
                 
            CASE WHEN f.ask > 0 AND fd.future_bid_10d IS NOT NULL 
                 THEN (fd.future_bid_10d - f.ask) / f.ask 
                 ELSE NULL END AS target_return_10d,
                 
            CASE WHEN fd.future_underlying_5d IS NOT NULL 
                 THEN (fd.future_underlying_5d - f.underlying_price) / f.underlying_price 
                 ELSE NULL END AS target_underlying_return_5d
                 
        FROM read_parquet('{features_file}') AS f
        LEFT JOIN forward_data fd
        ON f.contract_id = fd.contract_id AND f.date = fd.date
    ) TO '{output_file}' (FORMAT PARQUET);
    """
    
    con.execute(query)
    logger.info(f"Final dataset with targets written to {output_file}")
    con.close()

if __name__ == "__main__":
    build_targets(Path("data/processed"))
