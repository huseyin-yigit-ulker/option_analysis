import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WalkForwardEvaluator:
    """
    Implements strictly chronological walk-forward validation (Step 27 & 8).
    Never uses random train/test splitting.
    """
    def __init__(self, df, date_col='date', initial_train_years=2, val_years=1, test_years=1):
        self.df = df
        self.date_col = date_col
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.initial_train_years = initial_train_years
        self.val_years = val_years
        self.test_years = test_years
        
        self.min_year = self.df[self.date_col].dt.year.min()
        self.max_year = self.df[self.date_col].dt.year.max()

    def get_splits(self):
        """
        Yields (train_idx, val_idx, test_idx) chronologically.
        """
        current_train_end = self.min_year + self.initial_train_years - 1
        
        while current_train_end + self.val_years + self.test_years <= self.max_year:
            
            # Using expanding window for training
            train_mask = (self.df[self.date_col].dt.year >= self.min_year) & \
                         (self.df[self.date_col].dt.year <= current_train_end)
            
            val_start = current_train_end + 1
            val_end = val_start + self.val_years - 1
            val_mask = (self.df[self.date_col].dt.year >= val_start) & \
                       (self.df[self.date_col].dt.year <= val_end)
                       
            test_start = val_end + 1
            test_end = test_start + self.test_years - 1
            test_mask = (self.df[self.date_col].dt.year >= test_start) & \
                        (self.df[self.date_col].dt.year <= test_end)
                        
            yield (
                self.df[train_mask].index,
                self.df[val_mask].index,
                self.df[test_mask].index,
                (self.min_year, current_train_end, val_end, test_end)
            )
            
            # Advance window
            current_train_end += 1
