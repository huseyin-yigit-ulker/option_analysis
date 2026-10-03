import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_drawdown(equity_curve):
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    return drawdown

def generate_economic_metrics(trades: pd.DataFrame, capital=100000):
    if len(trades) == 0:
        return {}
    
    trades = trades.sort_values('date')
    trades['pnl_pct'] = trades['pnl'] / trades['entry_cost']
    
    # Simple compounding for the equity curve
    trades['equity'] = capital + trades['pnl'].cumsum()
    
    win_rate = (trades['pnl'] > 0).mean()
    gross_profit = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
    gross_loss = abs(trades.loc[trades['pnl'] < 0, 'pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    avg_win = trades.loc[trades['pnl'] > 0, 'pnl'].mean() if win_rate > 0 else 0
    avg_loss = trades.loc[trades['pnl'] < 0, 'pnl'].mean() if win_rate < 1 else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    daily_returns = trades.groupby('date')['pnl_pct'].mean().fillna(0)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    downside_returns = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    drawdown = compute_drawdown(trades['equity'])
    max_drawdown = drawdown.min()
    
    return {
        'Total Trades': len(trades),
        'Win Rate': win_rate,
        'Median Trade Return (%)': trades['pnl_pct'].median() * 100,
        'Mean Trade Return (%)': trades['pnl_pct'].mean() * 100,
        'Profit Factor': profit_factor,
        'Expectancy ($)': expectancy,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_drawdown * 100,
        'Avg Entry Spread ($)': trades['spread'].mean(),
        'Avg Transaction Cost ($)': (trades.get('entry_cost', 0) - trades.get('ask', 0)*100).mean() # Simplified
    }

def rank_bucket_analysis(df, prob_col='prob_profit', target_col='target_return_5d'):
    if prob_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()
        
    df['percentile_bucket'] = pd.qcut(df[prob_col], 10, labels=False, duplicates='drop')
    bucket_stats = df.groupby('percentile_bucket')[target_col].mean() * 100
    return bucket_stats

def plot_equity_curve(trades, output_dir: Path):
    if len(trades) == 0:
        return
        
    capital = 100000
    trades = trades.sort_values('date')
    trades['equity'] = capital + trades['pnl'].cumsum()
    
    plt.figure(figsize=(10, 6))
    plt.plot(trades['date'], trades['equity'], label='Equity')
    plt.title('Backtest Equity Curve (Ask->Bid)')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.grid(True)
    plt.savefig(output_dir / "equity_curve.png")
    plt.close()
    
    drawdown = compute_drawdown(trades['equity'])
    plt.figure(figsize=(10, 4))
    plt.fill_between(trades['date'], drawdown * 100, 0, color='red', alpha=0.3)
    plt.title('Drawdown (%)')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.savefig(output_dir / "drawdown.png")
    plt.close()

def generate_full_report(predictions_df: pd.DataFrame, trades_df: pd.DataFrame, output_dir: Path):
    logger.info("Generating economic and ranking reports...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Economic Metrics
    metrics = generate_economic_metrics(trades_df)
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_df.to_csv(output_dir / "walk_forward_results.csv", index=False)
    
    # 2. Rank Bucket Analysis
    bucket_stats = rank_bucket_analysis(predictions_df)
    if not bucket_stats.empty:
        bucket_stats.to_csv(output_dir / "ranking_analysis.csv")
    
    # 3. Plots
    plot_equity_curve(trades_df, output_dir)
    
    # 4. Save Trades
    trades_df.to_csv(output_dir / "backtest_trades.csv", index=False)
    logger.info(f"Reports saved to {output_dir}")

if __name__ == "__main__":
    pass
