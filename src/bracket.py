from itertools import permutations
import pandas as pd

from src.data_loader import load_datasets


def get_round_32_template():
    """Load Round of 32 bracket template."""

    data = load_datasets()
    df_knockout = data["knockout"]

    return df_knockout[df_knockout["round"] == "R32"].copy()


def assign_third_place_slots(third_place_slots, best_third_placed):
    """Assign third-placed teams to third-place slots."""

    third_teams = best_third_placed[["team", "group", "third_place_rank"]].copy()

    if len(third_place_slots) != len(third_teams):
        raise ValueError(
            f"Third-place slots ({len(third_place_slots)}) do not match "
            f"third-place teams ({len(third_teams)})."
        )

    for perm in permutations(third_teams.to_dict("records")):
        assignment = {}
        valid = True

        for slot, team_row in zip(third_place_slots, perm):
            allowed_groups = list(str(slot).replace("3", ""))

            if team_row["group"] not in allowed_groups:
                valid = False
                break

            assignment[slot] = team_row["team"]

        if valid:
            return assignment

    raise ValueError("No valid third-place assignment found.")


def build_round_32_bracket(df_round_32, direct_qualifiers, best_third_placed):
    """Fill the Round of 32 bracket with real teams."""

    df_round_32 = df_round_32.copy()

    third_place_slots = []

    for col in ["home_slot", "away_slot"]:
        slots = (
            df_round_32[col]
            .astype(str)
            .loc[lambda s: s.str.startswith("3")]
            .unique()
            .tolist()
        )
        third_place_slots.extend(slots)

    third_place_slots = list(dict.fromkeys(third_place_slots))

    third_place_assignment = assign_third_place_slots(
        third_place_slots=third_place_slots,
        best_third_placed=best_third_placed,
    )

    filled_matches = []

    for _, row in df_round_32.iterrows():
        home_slot = str(row["home_slot"])
        away_slot = str(row["away_slot"])

        if home_slot in direct_qualifiers:
            home_team = direct_qualifiers[home_slot]
        elif home_slot.startswith("3"):
            home_team = third_place_assignment[home_slot]
        else:
            raise ValueError(f"Unsupported home slot: {home_slot}")

        if away_slot in direct_qualifiers:
            away_team = direct_qualifiers[away_slot]
        elif away_slot.startswith("3"):
            away_team = third_place_assignment[away_slot]
        else:
            raise ValueError(f"Unsupported away slot: {away_slot}")

        match = row.to_dict()
        match["home_team"] = home_team
        match["away_team"] = away_team

        filled_matches.append(match)

    return pd.DataFrame(filled_matches)