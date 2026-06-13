import pandas as pd
import numpy as np
from scipy.stats import poisson

from src.data_loader import load_models, build_lookups


# Custom regularized Poisson class to avoid pickle import errors
class CustomRegularizedPoisson:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None
        self.feature_names = None
        
    def fit(self, X, y):
        self.feature_names = list(X.columns)
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        w0 = np.zeros(X_arr.shape[1])
        
        def loss(w):
            pred = np.exp(X_arr @ w)
            pred = np.clip(pred, 1e-10, 1e10)
            neg_log_lik = np.sum(pred - y_arr * (X_arr @ w))
            const_idx = self.feature_names.index('const') if 'const' in self.feature_names else 0
            l2_weights = np.copy(w)
            l2_weights[const_idx] = 0.0
            l2_penalty = 0.5 * self.alpha * np.sum(l2_weights ** 2)
            grad_nll = X_arr.T @ (pred - y_arr)
            grad_l2 = self.alpha * l2_weights
            return neg_log_lik + l2_penalty, grad_nll + grad_l2
            
        from scipy.optimize import minimize
        res = minimize(loss, w0, method='L-BFGS-B', jac=True, options={'maxiter': 1000})
        self.w = res.x
        return self
        
    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.exp(X_arr @ self.w)


def build_match_features(home_team, away_team, neutral=True, tournament_weight=5, model_type="baseline_poisson"):
    """
    Build one model-ready feature row.

    Features:
        const
        home_elo_pre
        away_elo_pre
        elo_diff
        elo_ratio
        tournament_weight
        neutral
        + polynomial features (for polynomial model)
    """

    _, _, feature_columns = load_models(model_type=model_type)
    lookups = build_lookups()

    team_to_elo = lookups["team_to_elo"]

    home_elo = team_to_elo.get(home_team)
    away_elo = team_to_elo.get(away_team)

    if home_elo is None:
        raise ValueError(f"Missing Elo rating for {home_team}")

    if away_elo is None:
        raise ValueError(f"Missing Elo rating for {away_team}")

    elo_diff = home_elo - away_elo
    elo_ratio = home_elo / away_elo

    row = pd.DataFrame([{
        "const": 1.0,
        "home_elo_pre": float(home_elo),
        "away_elo_pre": float(away_elo),
        "elo_diff": float(elo_diff),
        "elo_ratio": float(elo_ratio),
        "tournament_weight": float(tournament_weight),
        "neutral": float(neutral),
    }])

    if model_type == "polynomial_poisson":
        row["elo_diff_sq"] = float(elo_diff ** 2)
        row["elo_diff_neutral"] = float(elo_diff * float(neutral))
        row["elo_diff_weight"] = float(elo_diff * float(tournament_weight))
        row["elo_product"] = float(home_elo * away_elo)

    X = row.reindex(columns=feature_columns, fill_value=0.0)
    X = X.astype(float)

    return X


def predict_match(home_team, away_team, neutral=True, tournament_weight=5, max_goals=10, model_type="baseline_poisson"):
    """Predict expected goals and win/draw/loss probabilities."""

    model_home, model_away, _ = load_models(model_type=model_type)

    X = build_match_features(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
        tournament_weight=tournament_weight,
        model_type=model_type,
    )

    home_xg = float(model_home.predict(X)[0])
    away_xg = float(model_away.predict(X)[0])

    score_probs = []

    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = float(
                poisson.pmf(home_goals, home_xg)
                * poisson.pmf(away_goals, away_xg)
            )

            score_probs.append({
                "home_goals": home_goals,
                "away_goals": away_goals,
                "probability": prob,
            })

            if home_goals > away_goals:
                home_win_prob += prob
            elif home_goals == away_goals:
                draw_prob += prob
            else:
                away_win_prob += prob

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
        "score_probs": pd.DataFrame(score_probs),
    }



def get_h2h_score(team_a, team_b):
    """Return H2H score from team_a's point of view."""

    lookups = build_lookups()
    df_h2h = lookups["df_h2h"]

    direct_match = df_h2h[
        (df_h2h["team_a"] == team_a)
        & (df_h2h["team_b"] == team_b)
    ]

    if len(direct_match) > 0:
        return float(direct_match["h2h_score"].iloc[0])

    reverse_match = df_h2h[
        (df_h2h["team_a"] == team_b)
        & (df_h2h["team_b"] == team_a)
    ]

    if len(reverse_match) > 0:
        return -float(reverse_match["h2h_score"].iloc[0])

    return 0.0


def simulate_match(home_team, away_team, neutral=True, tournament_weight=5, model_type="baseline_poisson"):
    """Simulate one group-stage match. Draws are allowed."""

    pred = predict_match(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
        tournament_weight=tournament_weight,
        model_type=model_type,
    )

    home_goals = int(np.random.poisson(pred["home_xg"]))
    away_goals = int(np.random.poisson(pred["away_xg"]))

    if home_goals > away_goals:
        result = "H"
        winner = home_team
    elif away_goals > home_goals:
        result = "A"
        winner = away_team
    else:
        result = "D"
        winner = None

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": pred["home_xg"],
        "away_xg": pred["away_xg"],
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
        "winner": winner,
    }