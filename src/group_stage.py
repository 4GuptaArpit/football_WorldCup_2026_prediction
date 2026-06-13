import numpy as np
import pandas as pd

from src.data_loader import load_datasets, build_lookups
from src.prediction import simulate_match


def prepare_group_fixtures():
    """Add group labels to group-stage fixtures."""

    data = load_datasets()
    lookups = build_lookups()

    df_fixtures = data["fixtures"].copy()
    team_to_group = lookups["team_to_group"]

    df_fixtures["group"] = df_fixtures["home_team"].map(team_to_group)

    return df_fixtures


def create_empty_group_table(group_teams):
    """Create an empty table for one group."""

    return pd.DataFrame({
        "team": group_teams,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "points": 0,
    })


def update_group_table(table, match):
    """Update a group table after one match."""

    table = table.copy()

    home_team = match["home_team"]
    away_team = match["away_team"]
    home_goals = match["home_goals"]
    away_goals = match["away_goals"]

    table.loc[table["team"] == home_team, "played"] += 1
    table.loc[table["team"] == away_team, "played"] += 1

    table.loc[table["team"] == home_team, "goals_for"] += home_goals
    table.loc[table["team"] == home_team, "goals_against"] += away_goals

    table.loc[table["team"] == away_team, "goals_for"] += away_goals
    table.loc[table["team"] == away_team, "goals_against"] += home_goals

    if home_goals > away_goals:
        table.loc[table["team"] == home_team, "wins"] += 1
        table.loc[table["team"] == away_team, "losses"] += 1
        table.loc[table["team"] == home_team, "points"] += 3

    elif away_goals > home_goals:
        table.loc[table["team"] == away_team, "wins"] += 1
        table.loc[table["team"] == home_team, "losses"] += 1
        table.loc[table["team"] == away_team, "points"] += 3

    else:
        table.loc[table["team"] == home_team, "draws"] += 1
        table.loc[table["team"] == away_team, "draws"] += 1
        table.loc[table["team"] == home_team, "points"] += 1
        table.loc[table["team"] == away_team, "points"] += 1

    table["goal_difference"] = table["goals_for"] - table["goals_against"]

    return table


def rank_group_table(table):
    """Rank teams inside one group."""

    table = table.copy()
    table["random_tiebreaker"] = np.random.random(len(table))

    table = (
        table
        .sort_values(
            by=["points", "goal_difference", "goals_for", "random_tiebreaker"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )

    table["group_rank"] = table.index + 1

    return table.drop(columns=["random_tiebreaker"])


def simulate_group(group_name, model_type="baseline_poisson"):
    """Simulate all matches in one group."""

    data = load_datasets()
    df_groups = data["groups"]
    df_group_fixtures = prepare_group_fixtures()

    group_teams = (
        df_groups[df_groups["group"] == group_name]
        .sort_values("position")["nation"]
        .tolist()
    )

    group_matches = df_group_fixtures[df_group_fixtures["group"] == group_name]

    table = create_empty_group_table(group_teams)
    simulated_matches = []

    for _, row in group_matches.iterrows():
        match = simulate_match(
            home_team=row["home_team"],
            away_team=row["away_team"],
            neutral=bool(row["neutral"]),
            model_type=model_type,
        )

        simulated_matches.append(match)
        table = update_group_table(table, match)

    ranked_table = rank_group_table(table)
    ranked_table["group"] = group_name

    return ranked_table, pd.DataFrame(simulated_matches)


def simulate_group_stage(model_type="baseline_poisson"):
    """Simulate all 12 groups."""

    data = load_datasets()
    df_groups = data["groups"]

    all_group_tables = []
    all_group_matches = []

    for group_name in sorted(df_groups["group"].unique()):
        group_table, group_matches = simulate_group(group_name, model_type=model_type)

        all_group_tables.append(group_table)
        all_group_matches.append(group_matches)

    df_group_tables = pd.concat(all_group_tables, ignore_index=True)
    df_group_matches = pd.concat(all_group_matches, ignore_index=True)

    return df_group_tables, df_group_matches