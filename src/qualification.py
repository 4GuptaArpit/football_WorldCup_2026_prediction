import pandas as pd

from src.data_loader import build_lookups


def get_direct_qualifiers(df_group_tables):
    """Get group winners and runners-up."""

    direct_qualifiers = {}

    top_two = df_group_tables[df_group_tables["group_rank"].isin([1, 2])].copy()

    for _, row in top_two.iterrows():
        slot = f"{int(row['group_rank'])}{row['group']}"
        direct_qualifiers[slot] = row["team"]

    return direct_qualifiers


def get_best_third_placed_teams(df_group_tables, n_teams=8):
    """
    Get the best third-placed teams.

    We use FIFA rank as a practical final tie-breaker because
    fair play cards are not simulated.
    """

    lookups = build_lookups()
    team_to_fifa_rank = lookups["team_to_fifa_rank"]

    third_placed = df_group_tables[df_group_tables["group_rank"] == 3].copy()
    third_placed["fifa_rank"] = third_placed["team"].map(team_to_fifa_rank)

    best_third_placed = (
        third_placed
        .sort_values(
            by=["points", "goal_difference", "goals_for", "fifa_rank"],
            ascending=[False, False, False, True],
        )
        .head(n_teams)
        .reset_index(drop=True)
    )

    best_third_placed["third_place_rank"] = best_third_placed.index + 1

    return best_third_placed


def get_qualified_teams(df_group_tables):
    """Get all 32 knockout qualifiers."""

    direct_qualifiers = get_direct_qualifiers(df_group_tables)
    best_third_placed = get_best_third_placed_teams(df_group_tables)

    direct_teams = list(direct_qualifiers.values())
    third_placed_teams = best_third_placed["team"].tolist()

    qualified_teams = direct_teams + third_placed_teams

    return direct_qualifiers, best_third_placed, qualified_teams