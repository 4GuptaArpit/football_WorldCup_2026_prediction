import os
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
# Custom metrics are used instead of sklearn to avoid DLL load policies.


from pathlib import Path
PROJECT_ROOT = Path("e:/ML/football_WorldCup_2026_predictions")
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

from src.prediction import CustomRegularizedPoisson



def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def main():
    print("=== Custom Football Prediction Model Training Pipeline ===")
    
    # 1. Load match features dataset
    df_path = PROCESSED_DIR / "df_match_features.csv"
    print(f"Loading matches from {df_path}...")
    df = pd.read_csv(df_path, parse_dates=['date'])
    
    # 2. Filter competitive matches from 2000 onwards
    train_df = df[df['is_competitive'] & (df['date'] >= '2000-01-01')].copy()
    
    # 3. Train-validation split (Train: 2000-2023, Valid: 2024-2026)
    train_split = train_df[train_df['date'] < '2024-01-01']
    valid_split = train_df[train_df['date'] >= '2024-01-01']
    
    print(f"Train set: {len(train_split):,} matches")
    print(f"Validation set: {len(valid_split):,} matches")
    
    # 4. Feature engineering
    feature_cols = ['const', 'home_elo_pre', 'away_elo_pre', 'elo_diff', 'elo_ratio', 'tournament_weight', 'neutral']
    
    def prepare_features(data_frame):
        X = pd.DataFrame()
        X['home_elo_pre'] = data_frame['home_elo_pre']
        X['away_elo_pre'] = data_frame['away_elo_pre']
        X['elo_diff'] = data_frame['home_elo_pre'] - data_frame['away_elo_pre']
        X['elo_ratio'] = data_frame['home_elo_pre'] / data_frame['away_elo_pre']
        X['tournament_weight'] = data_frame['tournament_weight']
        X['neutral'] = data_frame['neutral'].astype(float)
        X['const'] = 1.0
        return X[feature_cols]

    X_train = prepare_features(train_split)
    X_valid = prepare_features(valid_split)
    
    y_home_train = train_split['home_score'].values
    y_away_train = train_split['away_score'].values
    
    y_home_valid = valid_split['home_score'].values
    y_away_valid = valid_split['away_score'].values
    
    print("\n--- Model A: Baseline statsmodels Poisson (GLM) ---")
    model_home_baseline = sm.GLM(y_home_train, X_train, family=sm.families.Poisson()).fit()
    model_away_baseline = sm.GLM(y_away_train, X_train, family=sm.families.Poisson()).fit()
    
    # Evaluate baseline
    pred_home_base = model_home_baseline.predict(X_valid)
    pred_away_base = model_away_baseline.predict(X_valid)
    mae_home_base, mse_home_base, _ = compute_metrics(y_home_valid, pred_home_base)
    mae_away_base, mse_away_base, _ = compute_metrics(y_away_valid, pred_away_base)
    print(f"Baseline Home goals MAE: {mae_home_base:.4f}, MSE: {mse_home_base:.4f}")
    print(f"Baseline Away goals MAE: {mae_away_base:.4f}, MSE: {mse_away_base:.4f}")
    print(f"Baseline Average MAE: {(mae_home_base + mae_away_base)/2:.4f}")

    print("\n--- Model B: Custom Regularized Poisson Regression (Pure SciPy/NumPy) ---")
    best_alpha_home, best_alpha_away = 1.0, 1.0
    best_mae_home, best_mae_away = 999.0, 999.0
    
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    for alpha in alphas:
        m_home = CustomRegularizedPoisson(alpha=alpha).fit(X_train, y_home_train)
        m_away = CustomRegularizedPoisson(alpha=alpha).fit(X_train, y_away_train)
        
        pred_h = m_home.predict(X_valid)
        pred_a = m_away.predict(X_valid)
        
        mae_h, _, _ = compute_metrics(y_home_valid, pred_h)
        mae_a, _, _ = compute_metrics(y_away_valid, pred_a)
        
        if mae_h < best_mae_home:
            best_mae_home = mae_h
            best_alpha_home = alpha
        if mae_a < best_mae_away:
            best_mae_away = mae_a
            best_alpha_away = alpha
            
    print(f"Best Alpha for Home Model: {best_alpha_home} (MAE: {best_mae_home:.4f})")
    print(f"Best Alpha for Away Model: {best_alpha_away} (MAE: {best_mae_away:.4f})")
    
    model_home_reg = CustomRegularizedPoisson(alpha=best_alpha_home).fit(X_train, y_home_train)
    model_away_reg = CustomRegularizedPoisson(alpha=best_alpha_away).fit(X_train, y_away_train)
    
    print("\n--- Model C: Custom Polynomial Regularized Poisson (Non-Linear Interaction GLM) ---")
    poly_cols = feature_cols + ['elo_diff_sq', 'elo_diff_neutral', 'elo_diff_weight', 'elo_product']
    
    def prepare_poly_features(data_frame):
        X = prepare_features(data_frame).copy()
        X['elo_diff_sq'] = X['elo_diff'] ** 2
        X['elo_diff_neutral'] = X['elo_diff'] * X['neutral']
        X['elo_diff_weight'] = X['elo_diff'] * X['tournament_weight']
        X['elo_product'] = X['home_elo_pre'] * X['away_elo_pre']
        return X[poly_cols]

    X_train_poly = prepare_poly_features(train_split)
    X_valid_poly = prepare_poly_features(valid_split)
    
    best_poly_alpha_home, best_poly_alpha_away = 1.0, 1.0
    best_poly_mae_home, best_poly_mae_away = 999.0, 999.0
    
    for alpha in alphas:
        m_home = CustomRegularizedPoisson(alpha=alpha).fit(X_train_poly, y_home_train)
        m_away = CustomRegularizedPoisson(alpha=alpha).fit(X_train_poly, y_away_train)
        
        pred_h = m_home.predict(X_valid_poly)
        pred_a = m_away.predict(X_valid_poly)
        
        mae_h, _, _ = compute_metrics(y_home_valid, pred_h)
        mae_a, _, _ = compute_metrics(y_away_valid, pred_a)
        
        if mae_h < best_poly_mae_home:
            best_poly_mae_home = mae_h
            best_poly_alpha_home = alpha
        if mae_a < best_poly_mae_away:
            best_poly_mae_away = mae_a
            best_poly_alpha_away = alpha
            
    print(f"Best Poly Alpha for Home Model: {best_poly_alpha_home} (MAE: {best_poly_mae_home:.4f})")
    print(f"Best Poly Alpha for Away Model: {best_poly_alpha_away} (MAE: {best_poly_mae_away:.4f})")
    
    model_home_poly = CustomRegularizedPoisson(alpha=best_poly_alpha_home).fit(X_train_poly, y_home_train)
    model_away_poly = CustomRegularizedPoisson(alpha=best_poly_alpha_away).fit(X_train_poly, y_away_train)

    # 5. Extract feature importances (for model C weights absolute values as proxy for importance)
    # Intercept column is usually ignored in feature importance
    importance_home = np.abs(model_home_poly.w)
    importance_away = np.abs(model_away_poly.w)
    
    df_importances = pd.DataFrame({
        'feature': poly_cols,
        'importance_home': importance_home,
        'importance_away': importance_away,
        'importance_avg': (importance_home + importance_away) / 2
    })
    # Drop 'const' from feature importance view
    df_importances = df_importances[df_importances['feature'] != 'const']
    # Normalize importance to sum to 100%
    tot = df_importances['importance_avg'].sum()
    df_importances['importance_avg'] = (df_importances['importance_avg'] / tot) * 100
    df_importances = df_importances.sort_values('importance_avg', ascending=False)
    
    # Save feature importances to CSV
    df_importances.to_csv(PROCESSED_DIR / "rf_feature_importances.csv", index=False)
    print("\nFeature importances calculated and saved to data/processed/rf_feature_importances.csv")
    print(df_importances.to_string(index=False))

    # 6. Save all models
    print("\nSaving model checkpoints to models/ folder...")
    
    # Baseline models
    with open(MODELS_DIR / "poisson_home_baseline.pkl", "wb") as f:
        pickle.dump(model_home_baseline, f)
    with open(MODELS_DIR / "poisson_away_baseline.pkl", "wb") as f:
        pickle.dump(model_away_baseline, f)
        
    # Regularized Poisson models
    with open(MODELS_DIR / "poisson_home_regularized.pkl", "wb") as f:
        pickle.dump(model_home_reg, f)
    with open(MODELS_DIR / "poisson_away_regularized.pkl", "wb") as f:
        pickle.dump(model_away_reg, f)
        
    # Polynomial Poisson models
    with open(MODELS_DIR / "poisson_home_polynomial.pkl", "wb") as f:
        pickle.dump(model_home_poly, f)
    with open(MODELS_DIR / "poisson_away_polynomial.pkl", "wb") as f:
        pickle.dump(model_away_poly, f)
        
    # Overwrite default pkl files for backward compatibility
    with open(MODELS_DIR / "poisson_home.pkl", "wb") as f:
        pickle.dump(model_home_baseline, f)
    with open(MODELS_DIR / "poisson_away.pkl", "wb") as f:
        pickle.dump(model_away_baseline, f)
        
    # Save the feature lists
    with open(MODELS_DIR / "feature_columns.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    with open(MODELS_DIR / "feature_columns_poly.pkl", "wb") as f:
        pickle.dump(poly_cols, f)

    print("\n=== Model training completed successfully! ===")

if __name__ == "__main__":
    main()
