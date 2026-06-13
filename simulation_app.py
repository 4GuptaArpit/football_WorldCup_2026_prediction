import base64
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_datasets, build_lookups, load_feature_importances
from src.prediction import predict_match, get_h2h_score
from src.tournament import simulate_tournament
from src.styling import (
    apply_global_styles,
    team_label,
    format_probability,
)


st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

apply_global_styles()


STATIC_DIR = Path(__file__).parent / "data" / "static"

# Same tiled background image on every page.
PAGE_BACKGROUNDS = {
    "probabilities": "fifa_wc2026_tournamnet_america.webp",
    "live_simulation": "fifa_wc2026_tournamnet_america.webp",
    "match_explorer": "fifa_wc2026_tournamnet_america.webp",
}

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _encode_image(filename):
    """Return (mime_type, base64_string) for an image in the static dir."""

    path = STATIC_DIR / filename
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return mime, encoded


def set_page_background(filename, tile_px=180):
    """Tile a small, 80% transparent background image across the current page."""

    mime, encoded = _encode_image(filename)

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)),
                url("data:{mime};base64,{encoded}");
            background-size: auto, {tile_px}px auto;
            background-position: center;
            background-attachment: fixed;
            background-repeat: repeat;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_trophies():
    """Animate trophies raining down the page (a balloons-style celebration)."""

    trophies = "".join(
        f'<div class="trophy" style="left:{left}%; '
        f'animation-delay:{delay}s; animation-duration:{duration}s; '
        f'font-size:{size}px;">🏆</div>'
        for left, delay, duration, size in [
            (5, 0.0, 3.2, 36),
            (15, 0.4, 3.8, 28),
            (27, 0.9, 3.0, 44),
            (38, 0.2, 4.2, 30),
            (50, 0.6, 3.5, 38),
            (62, 1.1, 3.1, 26),
            (73, 0.3, 4.0, 42),
            (84, 0.8, 3.6, 32),
            (93, 0.5, 3.3, 40),
        ]
    )

    st.markdown(
        f"""
        <style>
        @keyframes trophy-fall {{
            0% {{ transform: translateY(-10vh) rotate(0deg); opacity: 0; }}
            10% {{ opacity: 1; }}
            100% {{ transform: translateY(110vh) rotate(360deg); opacity: 0; }}
        }}
        .trophy-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9999;
        }}
        .trophy {{
            position: absolute;
            top: 0;
            animation-name: trophy-fall;
            animation-timing-function: ease-in;
            animation-iteration-count: 1;
        }}
        </style>
        <div class="trophy-container">{trophies}</div>
        """,
        unsafe_allow_html=True,
    )


data = load_datasets()
lookups = build_lookups()

df_probs = data["probabilities"]
df_groups = data["groups"]

team_to_elo = lookups["team_to_elo"]
team_to_form = lookups["team_to_form"]
team_to_fifa_rank = lookups["team_to_fifa_rank"]
team_to_fifa_rank_change = lookups["team_to_fifa_rank_change"]


def show_header():
    st.markdown(
        """
        <div class="main-title">⚽ World Cup 2026 Predictor</div>
        <div class="sub-title">
        Poisson goal model + Monte Carlo simulation + interactive tournament demo.
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_probability_table(df):
    """Format probability columns for display."""

    display_df = df.copy()

    probability_cols = [
        "r32_prob",
        "r16_prob",
        "qf_prob",
        "sf_prob",
        "final_prob",
        "winner_prob",
    ]

    for col in probability_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x * 100:.1f}%")

    if "team" in display_df.columns:
        display_df["team"] = display_df["team"].apply(team_label)

    return display_df


def page_probabilities():
    set_page_background(PAGE_BACKGROUNDS["probabilities"])
    show_header()

    st.subheader("🏆 Precomputed tournament probabilities")

    st.write(
        "This page loads the saved Monte Carlo results from CSV, so it is fast."
    )

    col1, col2, col3 = st.columns(3)

    top_team = df_probs.sort_values("winner_prob", ascending=False).iloc[0]

    with col1:
        st.metric("Most likely winner", team_label(top_team["team"]))

    with col2:
        st.metric("Win probability", format_probability(top_team["winner_prob"]))

    with col3:
        st.metric("FIFA rank", int(top_team["fifa_rank"]))

    st.markdown("### Top winner probabilities")

    top_n = st.slider("How many teams to show?", min_value=5, max_value=48, value=15)

    chart_df = (
        df_probs
        .sort_values("winner_prob", ascending=False)
        .head(top_n)
        .copy()
    )

    chart_df["team_label"] = chart_df["team"].apply(team_label)

    # Color bars by their descending rank: top 3 red, 4-10 green, rest dark blue.
    def rank_color(rank):
        if rank < 3:
            return "#D62828"   # red (like Canada)
        if rank < 10:
            return "#2A9D3F"   # green (like Mexico)
        return "#1A3A8F"       # dark blue (like USA)

    bar_colors = [rank_color(i) for i in range(len(chart_df))]

    fig = go.Figure(
        go.Bar(
            x=chart_df["team_label"],
            y=chart_df["winner_prob"],
            marker_color=bar_colors,
            hovertemplate="%{x}<br>Win probability: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Win probability",
        yaxis_tickformat=".0%",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    # Clearly visible white axis lines and ticks for readability.
    axis_style = dict(
        showline=True,
        linecolor="white",
        linewidth=2,
        ticks="outside",
        tickcolor="white",
        zeroline=False,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(
        **axis_style,
        gridcolor="rgba(255,255,255,0.15)",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Full probability table")

    table_cols = [
        "team",
        "confederation",
        "fifa_rank",
        "rank_change",
        "elo",
        "form_score",
        "r32_prob",
        "r16_prob",
        "qf_prob",
        "sf_prob",
        "final_prob",
        "winner_prob",
    ]

    st.dataframe(
        style_probability_table(df_probs[table_cols]),
        use_container_width=True,
        height=620,
    )

    st.markdown("### 🗺️ World Cup 2026 Winner Probabilities Map")
    st.write(
        "Hover over qualified nations to view their probability of winning the 2026 World Cup."
    )

    # Dictionary mapping national teams to their ISO Alpha-3 country codes
    ISO_MAPPING = {
        "Argentina": "ARG",
        "Spain": "ESP",
        "France": "FRA",
        "England": "GBR",
        "Brazil": "BRA",
        "Colombia": "COL",
        "Netherlands": "NLD",
        "Germany": "DEU",
        "Ecuador": "ECU",
        "Portugal": "PRT",
        "Croatia": "HRV",
        "Morocco": "MAR",
        "Uruguay": "URY",
        "Norway": "NOR",
        "Japan": "JPN",
        "Mexico": "MEX",
        "Turkey": "TUR",
        "Switzerland": "CHE",
        "Canada": "CAN",
        "Paraguay": "PRY",
        "South Korea": "KOR",
        "Senegal": "SEN",
        "United States": "USA",
        "Belgium": "BEL",
        "Australia": "AUS",
        "Panama": "PAN",
        "Algeria": "DZA",
        "Austria": "AUT",
        "Jordan": "JOR",
        "New Zealand": "NZL",
        "Iran": "IRN",
        "Scotland": "GBR",
        "DR Congo": "COD",
        "Bosnia and Herzegovina": "BIH",
        "Cape Verde": "CPV",
        "Tunisia": "TUN",
        "Curaçao": "CUW",
        "Sweden": "SWE",
        "Czech Republic": "CZE",
        "Egypt": "EGY",
        "South Africa": "ZAF",
        "Ivory Coast": "CIV",
        "Saudi Arabia": "SAU",
        "Qatar": "QAT",
        "Ghana": "GHA",
        "Haiti": "HTI",
        "Iraq": "IRQ",
        "Uzbekistan": "UZB",
    }

    map_df = df_probs.copy()
    map_df["iso_alpha"] = map_df["team"].map(ISO_MAPPING)
    map_df = map_df.dropna(subset=["iso_alpha"])
    map_df = map_df.sort_values("winner_prob", ascending=False).drop_duplicates(subset=["iso_alpha"])

    # Convert probability to percentage for better chart readability
    map_df["winner_prob_pct"] = map_df["winner_prob"] * 100

    # Custom premium gradient matching dashboard dark-theme aesthetic
    custom_colorscale = [
        [0.0, '#10172A'],    # Soft Slate Dark
        [0.05, '#1E40AF'],   # Deep Royal Blue
        [0.2, '#3B82F6'],    # Vibrant Blue
        [0.5, '#F59E0B'],    # Warm Amber
        [1.0, '#DC2626']     # Crimson Red
    ]

    fig_map = go.Figure(
        data=go.Choropleth(
            locations=map_df['iso_alpha'],
            z=map_df['winner_prob_pct'],
            text=map_df['team'],
            colorscale=custom_colorscale,
            autocolorscale=False,
            reversescale=False,
            marker_line_color='rgba(255,255,255,0.25)',
            marker_line_width=0.8,
            colorbar_title="Win Prob (%)",
            colorbar_title_font_color="white",
            colorbar_tickfont_color="white",
            hovertemplate="<b>%{text}</b><br>Win Probability: %{z:.2f}%<extra></extra>",
            zmin=0,
            zmax=28,
        )
    )

    fig_map.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(255,255,255,0.3)",
            projection_type='natural earth',
            bgcolor='rgba(0,0,0,0)',
            showland=True,
            landcolor='rgba(255,255,255,0.03)',
            showocean=True,
            oceancolor='rgba(0,0,0,0)',
            showlakes=False,
            showcountries=True,
            countrycolor="rgba(255,255,255,0.12)",
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
    )

    st.plotly_chart(fig_map, use_container_width=True)



def render_group_table(group_table):
    """Render a group table with top 2 highlighted."""

    table = group_table.copy()
    table["team"] = table["team"].apply(team_label)

    display_cols = [
        "group_rank",
        "team",
        "played",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "goal_difference",
        "points",
    ]

    def highlight_qualified(row):
        if row["group_rank"] <= 2:
            return ["background-color: rgba(48, 209, 88, 0.25); color: white;"] * len(row)
        if row["group_rank"] == 3:
            return ["background-color: rgba(255, 214, 10, 0.18); color: white;"] * len(row)
        return [""] * len(row)

    styled = table[display_cols].style.apply(highlight_qualified, axis=1)

    st.dataframe(styled, use_container_width=True, hide_index=True)


def page_live_simulation(model_type="baseline_poisson"):
    set_page_background(PAGE_BACKGROUNDS["live_simulation"])
    show_header()

    st.subheader("🎮 Live tournament simulation")

    st.write(
        "Click the button below to simulate one full World Cup path from groups to final."
    )

    run_button = st.button("▶️ Run one full tournament simulation", type="primary")

    if run_button:
        with st.spinner("Simulating tournament..."):
            tournament = simulate_tournament(model_type=model_type)

        summary = tournament["summary"]
        group_tables = tournament["group_tables"]
        group_matches = tournament["group_matches"]
        knockout_results = tournament["knockout_results"]

        st.markdown("## Group stage")

        for group_name in sorted(group_tables["group"].unique()):
            st.markdown(f"### Group {group_name}")

            group_matches_view = group_matches[
                group_matches["home_team"].isin(
                    group_tables[group_tables["group"] == group_name]["team"]
                )
            ].copy()

            for _, match in group_matches_view.iterrows():
                st.write(
                    f"{team_label(match['home_team'])} "
                    f"**{match['home_goals']} - {match['away_goals']}** "
                    f"{team_label(match['away_team'])}"
                )
                time.sleep(0.08)

            group_table = group_tables[group_tables["group"] == group_name]
            render_group_table(group_table)

        st.markdown("## Knockout stage")

        for round_name in ["R32", "R16", "QF", "SF", "Final"]:
            round_results = knockout_results[knockout_results["round"] == round_name]

            st.markdown(f'<div class="round-title">{round_name}</div>', unsafe_allow_html=True)

            cols = st.columns(2)

            for idx, (_, match) in enumerate(round_results.iterrows()):
                with cols[idx % 2]:
                    st.markdown(
                        f"""
                        <div class="team-card">
                            {team_label(match['home_team'])} 
                            <strong>{match['home_goals']} - {match['away_goals']}</strong> 
                            {team_label(match['away_team'])}
                            <br>
                            <span class="small-muted">
                            Winner: {team_label(match['winner'])}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.08)

        st.markdown(
            f"""
            <div class="winner-card">
                🏆 Winner: {team_label(summary["winner"])} 🏆
                <br>
                <span style="font-size:18px;">Runner-up: {team_label(summary["runner_up"])}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        show_trophies()


def display_probability_bar(win, draw, loss, team_a, team_b):
    """Render a premium horizontal win/draw/loss probability bar."""
    st.markdown(
        f"""
        <div style="margin-top: 15px; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 6px; color: white; font-family: 'Outfit', sans-serif; font-size: 15px;">
                <span>{team_a} Win: {win*100:.1f}%</span>
                <span>Draw: {draw*100:.1f}%</span>
                <span>{team_b} Win: {loss*100:.1f}%</span>
            </div>
            <div style="display: flex; height: 26px; border-radius: 13px; overflow: hidden; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
                <div style="width: {win*100}%; background: linear-gradient(90deg, #ff4d4d, #d62828); transition: width 0.5s ease;"></div>
                <div style="width: {draw*100}%; background: linear-gradient(90deg, #ffd700, #fca311); transition: width 0.5s ease;"></div>
                <div style="width: {loss*100}%; background: linear-gradient(90deg, #3182ce, #1A3A8F); transition: width 0.5s ease;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_radar_chart(team_a, team_b, pred):
    """Generate a Plotly Radar chart comparing two teams on normalized metrics."""
    elo_a = team_to_elo.get(team_a, 1500)
    elo_b = team_to_elo.get(team_b, 1500)
    
    rank_a = int(team_to_fifa_rank.get(team_a, 50))
    rank_b = int(team_to_fifa_rank.get(team_b, 50))
    
    # Invert rank: lower rank (better) = higher score on radar
    rank_score_a = max(1, 100 - rank_a)
    rank_score_b = max(1, 100 - rank_b)
    
    form_a = float(team_to_form.get(team_a, 0.0))
    form_b = float(team_to_form.get(team_b, 0.0))
    
    elo_score_a = max(1, min(100, (elo_a - 1000) / 12))
    elo_score_b = max(1, min(100, (elo_b - 1000) / 12))
    
    form_score_a = max(1, min(100, (form_a + 20) / 0.85))
    form_score_b = max(1, min(100, (form_b + 20) / 0.85))
    
    xg_score_a = max(1, min(100, pred['home_xg'] * 30))
    xg_score_b = max(1, min(100, pred['away_xg'] * 30))
    
    categories = ['Elo Rating', 'FIFA Rank (Inverted)', 'Recent Form', 'Expected Goals (xG)']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[elo_score_a, rank_score_a, form_score_a, xg_score_a],
        theta=categories,
        fill='toself',
        name=team_a,
        line_color='#ff4d4d',
        fillcolor='rgba(255, 77, 77, 0.15)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[elo_score_b, rank_score_b, form_score_b, xg_score_b],
        theta=categories,
        fill='toself',
        name=team_b,
        line_color='#3182ce',
        fillcolor='rgba(49, 130, 206, 0.15)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color='white',
                gridcolor='rgba(255, 255, 255, 0.08)',
                angle=45,
                tickfont=dict(color='#a0aec0')
            ),
            angularaxis=dict(
                color='white',
                gridcolor='rgba(255, 255, 255, 0.08)',
                tickfont=dict(size=12)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(l=40, r=40, t=25, b=25)
    )
    return fig


def page_match_explorer(model_type="baseline_poisson"):
    set_page_background(PAGE_BACKGROUNDS["match_explorer"])
    show_header()

    st.subheader("🔎 Match explorer")

    teams = sorted(df_groups["nation"].unique())

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox("Team A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)

    with col2:
        away_team = st.selectbox("Team B", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)

    if home_team == away_team:
        st.warning("Please select two different teams.")
        return

    pred = predict_match(home_team, away_team, model_type=model_type)

    st.markdown("### Model probabilities")
    display_probability_bar(pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"], home_team, away_team)

    col1, col2 = st.columns([1.2, 1.0])

    with col1:
        st.markdown("### Attribute Comparison")
        radar_fig = create_radar_chart(home_team, away_team, pred)
        st.plotly_chart(radar_fig, use_container_width=True)

    with col2:
        st.markdown("### Predictor metrics")
        st.metric(f"{team_label(home_team)} Expected Goals (xG)", f"{pred['home_xg']:.2f}")
        st.metric(f"{team_label(away_team)} Expected Goals (xG)", f"{pred['away_xg']:.2f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Most likely scorelines")
        scorelines = (
            pred["score_probs"]
            .sort_values("probability", ascending=False)
            .head(10)
            .copy()
        )
        scorelines["scoreline"] = (
            scorelines["home_goals"].astype(str)
            + " - "
            + scorelines["away_goals"].astype(str)
        )
        scorelines["probability"] = scorelines["probability"].map(lambda x: f"{x * 100:.1f}%")
        st.dataframe(
            scorelines[["scoreline", "probability"]],
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.markdown("### Historical context")
        context_data = pd.DataFrame([
            {
                "team": team_label(home_team),
                "elo": team_to_elo.get(home_team),
                "fifa_rank": team_to_fifa_rank.get(home_team),
                "rank_change": team_to_fifa_rank_change.get(home_team),
                "form_score": team_to_form.get(home_team),
                "h2h_score_vs_opponent": get_h2h_score(home_team, away_team),
            },
            {
                "team": team_label(away_team),
                "elo": team_to_elo.get(away_team),
                "fifa_rank": team_to_fifa_rank.get(away_team),
                "rank_change": team_to_fifa_rank_change.get(away_team),
                "form_score": team_to_form.get(away_team),
                "h2h_score_vs_opponent": get_h2h_score(away_team, home_team),
            },
        ])
        st.dataframe(context_data, use_container_width=True, hide_index=True)


def page_benchmarks():
    set_page_background(PAGE_BACKGROUNDS["probabilities"])
    show_header()

    st.subheader("📊 Model Benchmarks & XAI (Explainable AI)")
    st.write(
        "This section documents the machine learning performance metrics and feature explanations "
        "for our customized model family."
    )

    st.markdown("### 🏆 Model Comparison on Validation Holdout Split (2024-2026)")
    st.markdown(
        """
        We evaluated three architectures on unseen holdout matches (years 2024 to 2026).
        
        | Model Family | Key Features Used | Holdout MAE (Goal Predictions) | Relative Performance |
        | :--- | :--- | :---: | :---: |
        | **Model B: Regularized Poisson (Custom)** | Elo, Elo Ratio, Elo Diff, Weight, Neutral venue (L2 penalized) | **0.9637** | **+0.22% (Best model)** |
        | **Model A: Baseline Poisson Regression** | Elo, Weight, Neutral venue (unregularized) | 0.9658 | *Baseline Reference* |
        | **Model C: Polynomial Interaction Poisson** | Elo, Interaction products, Quadratic Elo ($EloDiff^2$) | 0.9718 | -0.62% Overfitting Penalty |
        
        *Notice how Model B (Regularized Poisson) outperforms the baseline by applying an L2 penalty to the parameters, preventing the model from overestimating the home advantage or tournament weights.*
        """
    )

    st.markdown("### 🔍 Feature Importance Explanation (Explainable AI)")
    st.write(
        "The chart below displays feature importances derived from the normalized weights of the Polynomial model (Model C). "
        "This shows which factors are mathematically most critical when predicting goal scoring rates."
    )

    df_imp = load_feature_importances()
    
    fig = go.Figure(go.Bar(
        x=df_imp['importance_avg'],
        y=df_imp['feature'],
        orientation='h',
        marker=dict(
            color='rgba(252, 163, 17, 0.7)',
            line=dict(color='#fca311', width=2)
        ),
        hovertemplate="Feature: %{y}<br>Importance: %{x:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title="Importance (%)",
        yaxis_title=None,
        yaxis=dict(autorange="reversed"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(l=50, r=40, t=10, b=20),
        height=380
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)', range=[0, 100])
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)


def main():
    # Sidebar Model Selector
    st.sidebar.subheader("🤖 Active Prediction Model")
    selected_model_display = st.sidebar.selectbox(
        "Select Model Type",
        [
            "Regularized Poisson (Custom)",
            "Baseline Poisson (GLM)",
            "Polynomial Poisson (Interaction GLM)"
        ],
        index=0 # Defaults to Regularized Poisson (our best model)
    )

    model_mapping = {
        "Baseline Poisson (GLM)": "baseline_poisson",
        "Regularized Poisson (Custom)": "regularized_poisson",
        "Polynomial Poisson (Interaction GLM)": "polynomial_poisson"
    }
    model_type = model_mapping[selected_model_display]

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Choose page",
        [
            "🏆 Precomputed probabilities",
            "🎮 Live simulation",
            "🔎 Match explorer",
            "📊 Model Benchmarks & XAI"
        ],
    )

    if page == "🏆 Precomputed probabilities":
        page_probabilities()

    elif page == "🎮 Live simulation":
        page_live_simulation(model_type=model_type)

    elif page == "🔎 Match explorer":
        page_match_explorer(model_type=model_type)

    elif page == "📊 Model Benchmarks & XAI":
        page_benchmarks()


if __name__ == "__main__":
    main()