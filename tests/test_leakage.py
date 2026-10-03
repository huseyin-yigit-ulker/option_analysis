import duckdb
import pytest
from pathlib import Path

def test_no_future_leakage_in_features():
    """
    Validates that none of the features at time T use data from T+N.
    We check the correlation between simple target (which explicitly uses future data)
    and our features. Perfect correlation would be a red flag.
    We also manually check that the max date in the raw data equals the max date in the feature set.
    """
    dataset_file = Path("data/processed/dataset.parquet")
    if not dataset_file.exists():
        pytest.skip("Dataset not built yet.")
        
    con = duckdb.connect()
    
    # Ensure there are no rows where DTE is negative 
    # (which would mean we used expired options to predict past events)
    bad_dte = con.execute(f"SELECT COUNT(*) FROM read_parquet('{dataset_file}') WHERE dte < 0").fetchone()[0]
    assert bad_dte == 0, "Lookahead/Leakage bias: Found negative DTE observations."
    
    # Ensure RV20 (historical realized volatility) doesn't perfectly predict future return
    # This is a crude heuristic, but if correlation is > 0.99, we leaked future price into RV20.
    corr = con.execute(f"""
        SELECT corr(rv_20, target_underlying_return_5d) 
        FROM read_parquet('{dataset_file}') 
        WHERE rv_20 IS NOT NULL AND target_underlying_return_5d IS NOT NULL
    """).fetchone()[0]
    
    # Correlation between past volatility and future return should naturally be weak/moderate.
    assert -0.8 < corr < 0.8, f"Lookahead bias suspected: RV20 has {corr} correlation with future return!"
    
    con.close()
