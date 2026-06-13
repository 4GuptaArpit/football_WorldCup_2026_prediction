import numpy as np
import pandas as pd

from src.prediction import predict_match


def simulate_knockout_match(home_team, away_team, neutral=True, tournament_weight=5, model_type="baseline_poisson"):
    """Simulate one knockout match. Draws are resolved."""

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
        winner = home_team
        loser = away_team
        result_type = "normal_time"

    elif away_goals > home_goals:
        winner = away_team
        loser = home_team
        result_type = "normal_time"

    else:
        home_strength = pred["home_win_prob"]
        away_strength = pred["away_win_prob"]

        home_advance_prob = home_strength / (home_strength + away_strength)

        if np.random.random() < home_advance_prob:
            winner = home_team
            loser = away_team
        else:
            winner = away_team
            loser = home_team

        result_type = "draw_resolved"

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": pred["home_xg"],
        "away_xg": pred["away_xg"],
        "home_goals": home_goals,
        "away_goals": away_goals,
        "winner": winner,
        "loser": loser,
        "result_type": result_type,
        "home_win_prob": pred["home_win_prob"],
        "draw_prob": pred["draw_prob"],
        "away_win_prob": pred["away_win_prob"],
    }


def simulate_knockout_round(df_round_matches, model_type="baseline_poisson"):
    """Simulate one knockout round."""

    simulated_results = []
    winners_by_match = {}

    for _, row in df_round_matches.iterrows():
        match_result = simulate_knockout_match(
            home_team=row["home_team"],
            away_team=row["away_team"],
            model_type=model_type,
        )

        match_result["match_id"] = row["match_id"]
        match_result["round"] = row["round"]
        match_result["winner_advances_to"] = row["winner_advances_to"]
        match_result["loser_advances_to"] = row["loser_advances_to"]

        simulated_results.append(match_result)
        winners_by_match[row["match_id"]] = match_result["winner"]

    return pd.DataFrame(simulated_results), winners_by_match



def build_next_round(df_knockout, previous_round_results, next_round_name):
    """Fill the next knockout round with winners from the previous round."""

    df_next_round = df_knockout[df_knockout["round"] == next_round_name].copy()

    next_match_teams = {}

    for _, row in previous_round_results.iterrows():
        next_match_id = row["winner_advances_to"]
        winner = row["winner"]

        if pd.isna(next_match_id):
            continue

        if next_match_id not in next_match_teams:
            next_match_teams[next_match_id] = []

        next_match_teams[next_match_id].append(winner)

    filled_matches = []

    for _, row in df_next_round.iterrows():
        match_id = row["match_id"]
        teams = next_match_teams.get(match_id, [])

        if len(teams) != 2:
            raise ValueError(
                f"Expected 2 teams for match {match_id}, got {len(teams)}: {teams}"
            )

        match = row.to_dict()
        match["home_team"] = teams[0]
        match["away_team"] = teams[1]

        filled_matches.append(match)

    return pd.DataFrame(filled_matches)


def simulate_knockout_stage(df_knockout, df_round_32_filled, model_type="baseline_poisson"):
    """Simulate R32 to Final."""

    df_r32_results, _ = simulate_knockout_round(df_round_32_filled, model_type=model_type)

    df_r16_filled = build_next_round(df_knockout, df_r32_results, "R16")
    df_r16_results, _ = simulate_knockout_round(df_r16_filled, model_type=model_type)

    df_qf_filled = build_next_round(df_knockout, df_r16_results, "QF")
    df_qf_results, _ = simulate_knockout_round(df_qf_filled, model_type=model_type)

    df_sf_filled = build_next_round(df_knockout, df_qf_results, "SF")
    df_sf_results, _ = simulate_knockout_round(df_sf_filled, model_type=model_type)

    df_final_filled = build_next_round(df_knockout, df_sf_results, "Final")
    df_final_results, _ = simulate_knockout_round(df_final_filled, model_type=model_type)

    df_knockout_results = pd.concat(
        [
            df_r32_results,
            df_r16_results,
            df_qf_results,
            df_sf_results,
            df_final_results,
        ],
        ignore_index=True,
    )

    winner = df_final_results["winner"].iloc[0]
    runner_up = df_final_results["loser"].iloc[0]

    return df_knockout_results, winner, runner_up