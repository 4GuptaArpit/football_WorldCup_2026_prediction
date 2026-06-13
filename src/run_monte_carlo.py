import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("e:/ML/football_WorldCup_2026_predictions")
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_datasets, build_lookups
from src.tournament import simulate_tournament

def main():
    print("=== World Cup 2026 Monte Carlo Simulation Runner ===")
    
    # 1. Load baseline data
    datasets = load_datasets()
    df_groups = datasets["groups"]
    wc_teams = sorted(df_groups["nation"].unique())
    
    # Initialize count dictionaries for each stage
    r32_counts = {t: 0 for t in wc_teams}
    r16_counts = {t: 0 for t in wc_teams}
    qf_counts = {t: 0 for t in wc_teams}
    sf_counts = {t: 0 for t in wc_teams}
    final_counts = {t: 0 for t in wc_teams}
    winner_counts = {t: 0 for t in wc_teams}
    
    # 2. Run simulation loop
    n_sims = 1000
    print(f"Simulating {n_sims} tournaments using the Regularized Poisson model...")
    
    # Track time and print progress
    import time
    start_time = time.time()
    
    for i in range(1, n_sims + 1):
        tourney = simulate_tournament(model_type="regularized_poisson")
        summary = tourney["summary"]
        
        # Accumulate winners and runners up
        winner_counts[summary["winner"]] += 1
        final_counts[summary["winner"]] += 1
        final_counts[summary["runner_up"]] += 1
        
        # Accumulate stages
        for t in summary["sf_teams"]:
            sf_counts[t] += 1
        for t in summary["qf_teams"]:
            qf_counts[t] += 1
        for t in summary["r16_teams"]:
            r16_counts[t] += 1
        for t in summary["r32_teams"]:
            r32_counts[t] += 1
            
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Completed {i}/{n_sims} simulations ({elapsed:.1f}s elapsed)...")
            
    # 3. Compute probabilities
    print("\nComputing probabilities and formatting final table...")
    results_list = []
    
    lookups = build_lookups()
    team_to_elo = lookups["team_to_elo"]
    team_to_form = lookups["team_to_form"]
    team_to_fifa_rank = lookups["team_to_fifa_rank"]
    team_to_fifa_rank_change = lookups["team_to_fifa_rank_change"]
    team_to_confederation = lookups["team_to_confederation"]
    
    for t in wc_teams:
        results_list.append({
            "team": t,
            "confederation": team_to_confederation.get(t, "Unknown"),
            "fifa_rank": int(team_to_fifa_rank.get(t, 50)),
            "rank_change": int(team_to_fifa_rank_change.get(t, 0)),
            "elo": float(team_to_elo.get(t, 1500.0)),
            "form_score": float(team_to_form.get(t, 0.0)),
            "r32_prob": r32_counts[t] / n_sims,
            "r16_prob": r16_counts[t] / n_sims,
            "qf_prob": qf_counts[t] / n_sims,
            "sf_prob": sf_counts[t] / n_sims,
            "final_prob": final_counts[t] / n_sims,
            "winner_prob": winner_counts[t] / n_sims,
        })
        
    df_probs = pd.DataFrame(results_list)
    df_probs = df_probs.sort_values("winner_prob", ascending=False).reset_index(drop=True)
    
    # 4. Save results to CSV
    output_path = PROJECT_ROOT / "data" / "processed" / "wc2026_tournament_probabilities.csv"
    df_probs.to_csv(output_path, index=False)
    
    print(f"\nMonte Carlo simulation complete! Probabilities saved to: {output_path}")
    print("\n--- Top 10 Winner Probabilities ---")
    print(df_probs[["team", "fifa_rank", "elo", "winner_prob"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
