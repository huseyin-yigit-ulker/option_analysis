import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def validate_dataset(raw_dir: Path, output_report_dir: Path):
    """
    Validates the raw options and underlying dataset according to the project rules.
    Outputs a reproducibility data-validation report.
    """
    logger.info("Starting data validation...")
    output_report_dir.mkdir(parents=True, exist_ok=True)
    report_lines = ["# Data Validation Report\n\n"]
    
    # 1. Check if files exist
    options_path = raw_dir / "options.parquet"
    underlying_path = raw_dir / "underlying.parquet"
    
    if not options_path.exists() or not underlying_path.exists():
        logger.error("Data files not found in raw directory.")
        return
        
    # We use Pandas here as the MVP dataset is small enough (synthetic 5-year data).
    # For the real 17-year 50M row dataset, this would use DuckDB or Polars.
    df_opt = pd.read_parquet(options_path)
    df_und = pd.read_parquet(underlying_path)
    
    # Basic Stats
    report_lines.append("## Dataset Statistics\n")
    report_lines.append(f"- **Options rows:** {len(df_opt):,}\n")
    report_lines.append(f"- **Underlying rows:** {len(df_und):,}\n")
    
    # Date Coverage
    opt_start, opt_end = df_opt['date'].min(), df_opt['date'].max()
    report_lines.append(f"- **Options Date Range:** {opt_start.date()} to {opt_end.date()}\n")
    
    # Missing Values
    report_lines.append("\n## Missing Values\n")
    report_lines.append("### Options\n")
    missing_opt = df_opt.isnull().sum()
    for col, val in missing_opt[missing_opt > 0].items():
        report_lines.append(f"- {col}: {val:,}\n")
    if missing_opt.sum() == 0:
        report_lines.append("- No missing values.\n")
        
    # Duplicate Checks
    report_lines.append("\n## Duplicates\n")
    duplicate_obs = df_opt.duplicated(subset=['contract_id', 'date']).sum()
    report_lines.append(f"- **Duplicate observations (contract + date):** {duplicate_obs:,}\n")
    
    # Quality Rule Checks
    report_lines.append("\n## Data Quality Checks (Violations)\n")
    
    # Invalid Bid/Ask
    invalid_bid = (df_opt['bid'] < 0).sum()
    invalid_ask = (df_opt['ask'] < 0).sum()
    ask_less_than_bid = (df_opt['ask'] < df_opt['bid']).sum()
    
    report_lines.append(f"- **Bid < 0:** {invalid_bid:,}\n")
    report_lines.append(f"- **Ask < 0:** {invalid_ask:,}\n")
    report_lines.append(f"- **Ask < Bid:** {ask_less_than_bid:,}\n")
    
    # Impossible Prices
    impossible_prices = (df_opt['mark'] < 0).sum()
    report_lines.append(f"- **Option mark < 0:** {impossible_prices:,}\n")
    
    # Zero / Negative IV
    neg_zero_iv = (df_opt['implied_volatility'] <= 0).sum()
    report_lines.append(f"- **IV <= 0:** {neg_zero_iv:,}\n")
    
    # Expired Contracts Observation (date > expiration)
    expired_obs = (df_opt['date'] > df_opt['expiration']).sum()
    report_lines.append(f"- **Observations past expiration date:** {expired_obs:,}\n")
    
    # Save Report
    report_path = output_report_dir / "data_validation_report.md"
    with open(report_path, "w") as f:
        f.writelines(report_lines)
        
    logger.info(f"Data validation report generated at {report_path}")

if __name__ == "__main__":
    validate_dataset(Path("data/raw"), Path("results/reports"))
