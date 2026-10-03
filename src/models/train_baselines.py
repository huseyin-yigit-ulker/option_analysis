import pandas as pd
import numpy as np
import logging
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.walk_forward import WalkForwardEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d', 'momentum_20d', 'rv_20',
    'iv_minus_rv20', 'iv_rv_ratio', 'abs_log_moneyness', 'itm_flag',
    'abs_delta', 'gamma_to_vega', 'theta_to_vega', 'quote_imbalance',
    'spread_pct', 'iv', 'dte'
]

def prepare_data(dataset_path: Path):
    df = pd.read_parquet(dataset_path)
    
    # Filter out NaNs in target
    df = df.dropna(subset=['target_return_5d']).copy()
    
    # Binary Classification Target
    df['target_class'] = (df['target_return_5d'] > 0).astype(int)
    
    # Fill feature NaNs for tree models (though xgb/lgb can handle them, 
    # it's safer for logistic regression to impute)
    # But let's build pipelines for each model.
    return df

def run_baselines(dataset_path: Path):
    df = prepare_data(dataset_path)
    evaluator = WalkForwardEvaluator(df, date_col='date', initial_train_years=2, val_years=1, test_years=1)
    
    for train_idx, val_idx, test_idx, (t_start, t_end, v_end, te_end) in evaluator.get_splits():
        logger.info(f"--- Fold Train: {t_start}-{t_end}, Val: {t_end+1}-{v_end}, Test: {v_end+1}-{te_end} ---")
        
        X_train, y_train = df.loc[train_idx, FEATURES], df.loc[train_idx, 'target_class']
        X_val, y_val = df.loc[val_idx, FEATURES], df.loc[val_idx, 'target_class']
        X_test, y_test = df.loc[test_idx, FEATURES], df.loc[test_idx, 'target_class']
        
        # 1. Logistic Regression
        lr = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
        ])
        lr.fit(X_train, y_train)
        lr_preds = lr.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_preds)
        logger.info(f"Logistic Regression Test AUC: {lr_auc:.4f}")
        
        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            eval_metric="auc", early_stopping_rounds=10, random_state=42, n_jobs=-1
        )
        # XGBoost handles NaNs natively
        xgb_model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            verbose=False
        )
        xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_preds)
        logger.info(f"XGBoost Test AUC: {xgb_auc:.4f}")
        
        # 3. LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            random_state=42, n_jobs=-1
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )
        lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
        lgb_auc = roc_auc_score(y_test, lgb_preds)
        logger.info(f"LightGBM Test AUC: {lgb_auc:.4f}")
        logger.info("-" * 40)

if __name__ == "__main__":
    run_baselines(Path("data/processed/dataset.parquet"))
