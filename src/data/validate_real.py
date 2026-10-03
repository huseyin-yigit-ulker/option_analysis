import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def validate_real_data():
    """Step 2: Inspect the real data."""
    logger.info("Starting real data quality validation...")
    opt_file = "data/interim/spy/options_2020_2024.parquet"
    und_file = "data/interim/spy/underlying_2020_2024.parquet"
    out_file = "results/data_quality_report.html"
    Path("results").mkdir(exist_ok=True)
    
    con = duckdb.connect()
    html = ["<html><body><h1>SPY Real Data Quality Report (2020-2024)</h1>"]
    
    # Schema inspection
    html.append("<h2>1. Schema Inspection</h2><pre>")
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{opt_file}')").df()
    html.append(schema.to_string())
    html.append("</pre>")
    
    # Row Count & Date Coverage
    res = con.execute(f"SELECT COUNT(*), MIN(date), MAX(date) FROM read_parquet('{opt_file}')").fetchone()
    html.append(f"<h2>2. Row Count & 3. Date Coverage</h2>")
    html.append(f"<p>Total Rows: {res[0]:,}</p>")
    html.append(f"<p>Date Range: {res[1]} to {res[2]}</p>")
    
    # Missing Values
    html.append("<h2>4. Missing Values</h2><ul>")
    cols = schema['column_name'].tolist()
    for col in cols:
        miss = con.execute(f"SELECT COUNT(*) FROM read_parquet('{opt_file}') WHERE {col} IS NULL").fetchone()[0]
        if miss > 0:
            html.append(f"<li>{col}: {miss:,} missing</li>")
    html.append("</ul>")
    
    # Duplicate observations
    dups = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT contract_id, date, COUNT(*) as cnt 
            FROM read_parquet('{opt_file}') 
            GROUP BY contract_id, date HAVING cnt > 1
        )
    """).fetchone()[0]
    html.append(f"<h2>5. Duplicate Observations</h2><p>{dups:,}</p>")
    
    # Invalid Bid/Ask
    inv_bid = con.execute(f"SELECT COUNT(*) FROM read_parquet('{opt_file}') WHERE bid < 0 OR ask < 0 OR bid > ask").fetchone()[0]
    html.append(f"<h2>6. Invalid Bid/Ask (Bid<0, Ask<0, Bid>Ask)</h2><p>{inv_bid:,}</p>")
    
    # Invalid IV
    inv_iv = con.execute(f"SELECT COUNT(*) FROM read_parquet('{opt_file}') WHERE implied_volatility <= 0").fetchone()[0]
    html.append(f"<h2>7. Invalid IV (<=0)</h2><p>{inv_iv:,}</p>")
    
    # Invalid Greeks
    inv_greeks = con.execute(f"SELECT COUNT(*) FROM read_parquet('{opt_file}') WHERE gamma < 0 OR vega < 0").fetchone()[0]
    html.append(f"<h2>8. Invalid Greeks (Gamma < 0 or Vega < 0)</h2><p>{inv_greeks:,}</p>")
    
    # Expiration consistency
    exp_cons = con.execute(f"SELECT COUNT(*) FROM read_parquet('{opt_file}') WHERE date > expiration").fetchone()[0]
    html.append(f"<h2>9. Expiration Consistency (Date > Exp)</h2><p>{exp_cons:,}</p>")
    
    # Option Type Distribution
    html.append("<h2>10. CALL/PUT Distribution</h2><ul>")
    dist = con.execute(f"SELECT type, COUNT(*) FROM read_parquet('{opt_file}') GROUP BY type").fetchall()
    for row in dist:
        html.append(f"<li>{row[0]}: {row[1]:,}</li>")
    html.append("</ul>")
    
    # DTE Distribution (approx)
    html.append("<h2>11. DTE Distribution</h2><ul>")
    dte_dist = con.execute(f"""
        SELECT 
            CASE 
                WHEN date_diff('day', date, expiration) < 7 THEN '< 7 days'
                WHEN date_diff('day', date, expiration) BETWEEN 7 AND 30 THEN '7-30 days'
                WHEN date_diff('day', date, expiration) BETWEEN 31 AND 90 THEN '31-90 days'
                ELSE '> 90 days'
            END as dte_bucket,
            COUNT(*) 
        FROM read_parquet('{opt_file}')
        GROUP BY 1
    """).fetchall()
    for row in dte_dist:
        html.append(f"<li>{row[0]}: {row[1]:,}</li>")
    html.append("</ul>")
    
    html.append("</body></html>")
    
    with open(out_file, "w") as f:
        f.write("\n".join(html))
    logger.info(f"Data quality report saved to {out_file}")
    con.close()

if __name__ == "__main__":
    validate_real_data()
