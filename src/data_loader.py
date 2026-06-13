from pathlib import Path
import pickle
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_resource(show_spinner=False)
def load_models(model_type="baseline_poisson"):
    """Load trained models based on model_type."""

    if model_type == "baseline_poisson":
        home_path = MODELS_DIR / "poisson_home_baseline.pkl"
        away_path = MODELS_DIR / "poisson_away_baseline.pkl"
        cols_path = MODELS_DIR / "feature_columns.pkl"
    elif model_type == "regularized_poisson":
        home_path = MODELS_DIR / "poisson_home_regularized.pkl"
        away_path = MODELS_DIR / "poisson_away_regularized.pkl"
        cols_path = MODELS_DIR / "feature_columns.pkl"
    elif model_type == "polynomial_poisson":
        home_path = MODELS_DIR / "poisson_home_polynomial.pkl"
        away_path = MODELS_DIR / "poisson_away_polynomial.pkl"
        cols_path = MODELS_DIR / "feature_columns_poly.pkl"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    with open(home_path, "rb") as f:
        model_home = pickle.load(f)

    with open(away_path, "rb") as f:
        model_away = pickle.load(f)

    with open(cols_path, "rb") as f:
        feature_columns = pickle.load(f)

    return model_home, model_away, feature_columns


@st.cache_data(show_spinner=False)
def load_feature_importances():
    """Load feature importances from processed folder."""
    return pd.read_csv(PROCESSED_DIR / "rf_feature_importances.csv")



@st.cache_data(show_spinner=False)
def load_datasets():
    """Load all datasets needed by the app."""

    df_fixtures = pd.read_csv(INTERIM_DIR / "wc2026_fixtures.csv")

    df_match_features = pd.read_csv(PROCESSED_DIR / "df_match_features.csv")
    df_form = pd.read_csv(PROCESSED_DIR / "df_form_2026.csv")
    df_h2h = pd.read_csv(PROCESSED_DIR / "df_h2h_2026.csv")
    df_probabilities = pd.read_csv(PROCESSED_DIR / "wc2026_tournament_probabilities.csv")

    df_confederations = pd.read_csv(REFERENCE_DIR / "FIFA_confederations.csv")
    df_knockout = pd.read_csv(REFERENCE_DIR / "fixtures_knockout_wc2026.csv")
    df_groups = pd.read_csv(REFERENCE_DIR / "group_stages.csv", sep=";")

    df_fifa_rank = pd.read_csv(
        DATA_DIR / "raw" / "wc_2026_48_teams_fifa_rank_change_corrected.csv"
    )

    return {
        "fixtures": df_fixtures,
        "match_features": df_match_features,
        "form": df_form,
        "h2h": df_h2h,
        "probabilities": df_probabilities,
        "confederations": df_confederations,
        "knockout": df_knockout,
        "groups": df_groups,
        "fifa_rank": df_fifa_rank,
    }


@st.cache_data(show_spinner=False)
def build_lookups():
    """Build lookup dictionaries used by prediction and simulation functions."""

    data = load_datasets()

    df_match_features = data["match_features"].copy()
    df_confederations = data["confederations"]
    df_form = data["form"]
    df_fifa_rank = data["fifa_rank"]
    df_h2h = data["h2h"]
    df_groups = data["groups"]

    df_match_features["date"] = pd.to_datetime(df_match_features["date"])

    home_elo = df_match_features[["date", "home_team", "home_elo_pre"]].rename(
        columns={"home_team": "team", "home_elo_pre": "elo"}
    )

    away_elo = df_match_features[["date", "away_team", "away_elo_pre"]].rename(
        columns={"away_team": "team", "away_elo_pre": "elo"}
    )

    df_team_elo = pd.concat([home_elo, away_elo], ignore_index=True)

    df_latest_elo = (
        df_team_elo
        .sort_values("date")
        .drop_duplicates(subset="team", keep="last")
        .reset_index(drop=True)
    )

    team_to_elo = dict(zip(df_latest_elo["team"], df_latest_elo["elo"]))

    team_to_confederation = dict(
        zip(df_confederations["nation"], df_confederations["confederation"])
    )

    team_to_group = dict(
        zip(df_groups["nation"], df_groups["group"])
    )

    team_to_form = dict(
        zip(df_form["team"], df_form["form_score"])
    )

    team_to_fifa_rank = dict(
        zip(df_fifa_rank["Nation"], df_fifa_rank["FIFA_2026_rank"])
    )

    team_to_fifa_rank_change = dict(
        zip(df_fifa_rank["Nation"], df_fifa_rank["rank_change"])
    )

    return {
        "team_to_elo": team_to_elo,
        "team_to_confederation": team_to_confederation,
        "team_to_group": team_to_group,
        "team_to_form": team_to_form,
        "team_to_fifa_rank": team_to_fifa_rank,
        "team_to_fifa_rank_change": team_to_fifa_rank_change,
        "df_h2h": df_h2h,
    }