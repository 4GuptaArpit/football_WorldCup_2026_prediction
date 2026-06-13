from src.data_loader import load_datasets
from src.group_stage import simulate_group_stage
from src.qualification import get_qualified_teams
from src.bracket import get_round_32_template, build_round_32_bracket
from src.knockout import simulate_knockout_stage


def simulate_tournament(model_type="baseline_poisson"):
    """Simulate one full World Cup tournament."""

    data = load_datasets()
    df_knockout = data["knockout"]

    df_group_tables, df_group_matches = simulate_group_stage(model_type=model_type)

    direct_qualifiers, best_third_placed, qualified_teams = get_qualified_teams(
        df_group_tables
    )

    df_round_32 = get_round_32_template()

    df_round_32_filled = build_round_32_bracket(
        df_round_32=df_round_32,
        direct_qualifiers=direct_qualifiers,
        best_third_placed=best_third_placed,
    )

    df_knockout_results, winner, runner_up = simulate_knockout_stage(
        df_knockout=df_knockout,
        df_round_32_filled=df_round_32_filled,
        model_type=model_type,
    )


    r32_teams = qualified_teams

    r16_teams = (
        df_knockout_results
        .loc[df_knockout_results["round"] == "R32", "winner"]
        .tolist()
    )

    qf_teams = (
        df_knockout_results
        .loc[df_knockout_results["round"] == "R16", "winner"]
        .tolist()
    )

    sf_teams = (
        df_knockout_results
        .loc[df_knockout_results["round"] == "QF", "winner"]
        .tolist()
    )

    final_teams = (
        df_knockout_results
        .loc[df_knockout_results["round"] == "SF", "winner"]
        .tolist()
    )

    summary = {
        "winner": winner,
        "runner_up": runner_up,
        "r32_teams": r32_teams,
        "r16_teams": r16_teams,
        "qf_teams": qf_teams,
        "sf_teams": sf_teams,
        "final_teams": final_teams,
    }

    return {
        "summary": summary,
        "group_tables": df_group_tables,
        "group_matches": df_group_matches,
        "round_32_bracket": df_round_32_filled,
        "knockout_results": df_knockout_results,
    }