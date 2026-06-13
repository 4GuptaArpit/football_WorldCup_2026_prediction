# World Cup 2026 Prediction Project

A football analytics project that predicts World Cup 2026 team probabilities using custom Poisson models, Monte Carlo simulation, and Explainable AI (XAI) diagnostics.

> [!NOTE]
> **Academic Integrity / Credits**: The raw datasets and tournament fixtures structure compilation are credited to original author Anas Riad (LinkedIn: [Anas Riad](https://www.linkedin.com/in/riadanas/)). 
> This repository has been customized and optimized with advanced feature engineering, regularized optimization, model benchmarking pipelines, and XAI visualizations for submission to the **Amazon ML School**.

## Custom ML Enhancements

This project has been enhanced beyond the baseline Poisson model to implement professional-grade machine learning workflows:

1. **Custom Regularized Optimization (`CustomRegularizedPoisson`)**:
   - Built a regularized Poisson regression model from scratch with L2 regularization using `scipy.optimize.minimize` (L-BFGS-B optimization on negative log-likelihood with analytical gradients).
   - This bypasses environment restrictions on loading compiled Cython binaries (such as `scikit-learn` estimators) under restrictive workstation execution policies.
2. **Feature Engineering**:
   - Added Elo rating difference, Elo ratio, FIFA rank difference/ratio, recent form, head-to-head (H2H) historic indicators, neutral venue, and tournament weights.
   - Built a polynomial/interaction variant (`Polynomial Poisson`) utilizing quadratic Elo terms ($EloDiff^2$) and cross-feature products.
3. **Robust Holdout Validation**:
   - Evaluated models using competitive international matches post-2000 for training (14,820 matches) and a holdout set (2024–2026, 1,854 matches) to check for overfitting.
4. **Explainable AI (XAI)**:
   - Extracted normalized feature importances from optimization coefficients to show which indicators (e.g., Elo difference, tournament weight, neutral venue) hold the most predictive power.
5. **Interactive Web Dashboard**:
   - Redesigned the Streamlit app with premium Outfit typography, glassmorphism aesthetics, dynamic probability bars, and Plotly Radar Charts to compare team attributes.

## Model Benchmarks & Validation Results

Evaluated on the 2024–2026 holdout dataset (Mean Absolute Error for goal predictions):

| Model Family | Key Features Used | Holdout MAE | Relative Performance |
| :--- | :--- | :---: | :---: |
| **Model B: Regularized Poisson (Custom)** | Elo, Elo Ratio, Elo Diff, Weight, Neutral venue (L2 penalized) | **0.9637** | **Best Performance (+0.22%)** |
| **Model A: Baseline Poisson Regression** | Elo, Weight, Neutral venue (unregularized) | 0.9658 | *Baseline Reference* |
| **Model C: Polynomial Poisson (Interaction)** | Elo, Interaction products, Quadratic Elo ($EloDiff^2$) | 0.9718 | Overfitting Penalty (-0.62%) |

## Project Structure

```text
football_wc2026/
│
├── data/
│   ├── interim/
│   ├── processed/
│   │   ├── wc2026_tournament_probabilities.csv (Monte Carlo outputs)
│   │   └── rf_feature_importances.csv
│   ├── raw/
│   └── reference/
│
├── models/
│   ├── poisson_home_baseline.pkl / poisson_away_baseline.pkl
│   ├── poisson_home_regularized.pkl / poisson_away_regularized.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   ├── 0-data_fetch.ipynb
│   ├── 1-data_cleaning.ipynb
│   ├── 2-feature_engineering_ELO.ipynb
│   ├── 3-feature_engineering_match_metadata.ipynb
│   ├── 4-model_training.ipynb
│   └── 5-tournament_simulation.ipynb
│
├── src/
│   ├── data_loader.py (Loads datasets, features, models)
│   ├── prediction.py (Unpickles estimators & predicts goals/match outcomes)
│   ├── group_stage.py (Simulates groups and third-place rankings)
│   ├── qualification.py
│   ├── bracket.py
│   ├── knockout.py (Simulates knockout rounds)
│   ├── tournament.py (Assembles tournament flow)
│   ├── run_monte_carlo.py (Runs 1,000 simulations using the best model)
│   └── styling.py (Premium UI styles)
│
├── simulation_app.py (Streamlit dashboard app)
├── pyproject.toml
├── uv.lock
└── README.md
```

## How to Run

1. Install dependencies using your preferred package manager (e.g. `uv` or `pip`):
   ```bash
   pip install pandas numpy scipy statsmodels streamlit plotly
   ```
2. Start the interactive Streamlit app:
   ```bash
   streamlit run simulation_app.py
   ```
3. Open `http://localhost:8501` in your browser.

## Summary of results

The final output is a probability table showing each team’s chance of reaching:

* Round of 32
* Round of 16
* Quarter-finals
* Semi-finals
* Final
* Winner

The 1,000-run Monte Carlo simulation showed that the model strongly favored teams with the highest Elo ratings, especially Argentina and Spain. The final results were saved to:

```text
data/processed/wc2026_tournament_probabilities.csv
```

The Streamlit app includes:

* a precomputed probability dashboard
* a live single-tournament simulation
* a match explorer for team-vs-team predictions
