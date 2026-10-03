import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class OptionsBacktester:
    """
    Step 29: Backtest Engine
    Simulates portfolio performance by picking the Top N highest probability 
    options every day based on the model's output.
    """
    def __init__(self, top_n=1, slippage=0.01, commission=0.65):
        self.top_n = top_n
        self.slippage = slippage
        self.commission = commission
        
    def run(self, df, prob_col='prob_profit'):
        logger.info(f"Running backtest picking Top {self.top_n} contracts per day...")
        
        # Ensure we only pick valid liquid options per rules
        eligible = df[(df['bid'] > 0) & (df['ask'] > df['bid']) & (df['spread_pct'] <= 0.15)].copy()
        
        # Rank options each day
        eligible['rank'] = eligible.groupby(['date', 'option_type'])[prob_col].rank("dense", ascending=False)
        
        # Select top candidates (we'll just do Top 1 Call and Top 1 Put for simplicity)
        trades = eligible[eligible['rank'] <= self.top_n].copy()
        
        # Calculate realistic PnL per contract (Assuming 1 contract = 100 shares)
        # Entry = Ask + Slippage
        trades['entry_cost'] = (trades['ask'] + self.slippage) * 100 + self.commission
        # Exit = Bid 5d - Slippage
        # We rely on the fact that target_return_5d was calculated using future_bid_5d
        trades['future_bid_5d'] = trades['ask'] * (1 + trades['target_return_5d'])
        trades['exit_revenue'] = (trades['future_bid_5d'] - self.slippage) * 100 - self.commission
        
        trades['pnl'] = trades['exit_revenue'] - trades['entry_cost']
        trades['return_on_capital'] = trades['pnl'] / trades['entry_cost']
        
        # Metrics
        total_pnl = trades['pnl'].sum()
        win_rate = (trades['pnl'] > 0).mean()
        avg_trade = trades['pnl'].mean()
        
        logger.info(f"Backtest Results:")
        logger.info(f"Total Trades: {len(trades)}")
        logger.info(f"Total PnL: ${total_pnl:,.2f}")
        logger.info(f"Win Rate: {win_rate*100:.1f}%")
        logger.info(f"Avg PnL per trade: ${avg_trade:.2f}")
        
        return trades

if __name__ == "__main__":
    # Dummy run
    logger.info("Backtest engine ready.")
