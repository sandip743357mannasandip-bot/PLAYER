"""
app.py - Streamlit Dashboard for Football Match Prediction
Deploy on Streamlit Cloud via GitHub
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from backend import load_all_players, get_all_clubs, get_squad, predict_match

# ── Page config ──
st.set_page_config(
    page_title="⚽ La Liga Predictor 2024-25",
    page_icon="⚽",
    layout="wide"
)

# ── Load players once ──
@st.cache_data
def load_data():
    return load_all_players()

PLAYERS = load_data()
SEASON  = "2024-25"
ALL_CLUBS = get_all_clubs(PLAYERS, SEASON)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("⚽ La Liga Match Predictor — 2024-25")
st.markdown(
    "Select teams, pick a match date, and choose your Playing XI. "
    "The model uses **only data before the selected date** — no future leakage."
)
st.divider()

# ─────────────────────────────────────────
# ROW 1: TEAM & DATE SELECTION
# ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    home_team = st.selectbox("🏠 Home Team", ALL_CLUBS, index=0)
with col2:
    away_options = [c for c in ALL_CLUBS if c != home_team]
    away_team = st.selectbox("✈️ Away Team", away_options, index=0)
with col3:
    match_date = st.date_input(
        "📅 Match Date",
        value=pd.to_datetime("2025-03-01"),
        min_value=pd.to_datetime("2024-08-01"),
        max_value=pd.to_datetime("2025-07-01"),
    )

st.divider()

# ─────────────────────────────────────────
# ROW 2: PLAYING XI SELECTORS
# ─────────────────────────────────────────
home_squad = get_squad(PLAYERS, home_team, SEASON)
away_squad = get_squad(PLAYERS, away_team, SEASON)

col_home, col_away = st.columns(2)

home_xi = []
away_xi = []

with col_home:
    st.markdown(f"### 🏠 {home_team} — Playing XI")
    if not home_squad:
        st.warning("No players found for this team. Add player CSVs to player_data/ folder.")
    else:
        for i in range(11):
            default_idx = i if i < len(home_squad) else 0
            player = st.selectbox(
                f"Player {i+1}",
                home_squad,
                index=min(default_idx, len(home_squad)-1),
                key=f"home_{i}"
            )
            home_xi.append(player)

with col_away:
    st.markdown(f"### ✈️ {away_team} — Playing XI")
    if not away_squad:
        st.warning("No players found for this team. Add player CSVs to player_data/ folder.")
    else:
        for i in range(11):
            default_idx = i if i < len(away_squad) else 0
            player = st.selectbox(
                f"Player {i+1}",
                away_squad,
                index=min(default_idx, len(away_squad)-1),
                key=f"away_{i}"
            )
            away_xi.append(player)

st.divider()

# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
predict_clicked = st.button("🔮 PREDICT MATCH", type="primary", use_container_width=True)

if predict_clicked:
    if home_team == away_team:
        st.error("⚠️ Home and Away teams must be different.")
    elif not home_xi or not away_xi:
        st.error("⚠️ Please select players for both teams.")
    else:
        with st.spinner("Calculating prediction..."):
            try:
                result = predict_match(
                    PLAYERS,
                    home_team, away_team,
                    list(set(home_xi)),   # deduplicate
                    list(set(away_xi)),
                    str(match_date),
                    season=SEASON
                )

                st.divider()
                st.markdown("## 📊 Prediction Results")

                # ── xG Row ──
                c1, c2, c3 = st.columns(3)
                c1.metric(f"🏠 {home_team} xG", result["xg_home"])
                c2.metric("VS", "")
                c3.metric(f"✈️ {away_team} xG", result["xg_away"])

                st.divider()

                # ── Probabilities ──
                c1, c2, c3 = st.columns(3)
                c1.metric(f"🏠 {home_team} Win", f"{result['home_win']}%")
                c2.metric("🤝 Draw",              f"{result['draw']}%")
                c3.metric(f"✈️ {away_team} Win",  f"{result['away_win']}%")

                # ── Probability bar chart ──
                prob_df = pd.DataFrame({
                    "Outcome": [f"{home_team} Win", "Draw", f"{away_team} Win"],
                    "Probability (%)": [result["home_win"], result["draw"], result["away_win"]]
                })
                st.bar_chart(prob_df.set_index("Outcome"))

                st.divider()

                # ── Top 5 Scorelines ──
                st.markdown(f"### 🎯 Top 5 Most Likely Scorelines")
                st.markdown(f"*(Home: {home_team} — Away: {away_team})*")
                top5_df = pd.DataFrame(
                    result["top5"],
                    columns=["Scoreline", "Probability (%)"]
                )
                st.dataframe(top5_df, use_container_width=True, hide_index=True)

                # ── Scoreline heatmap ──
                st.divider()
                st.markdown("### 🔥 Scoreline Probability Heatmap (%)")
                matrix_pct = np.round(result["matrix"] * 100, 2)
                heatmap_df = pd.DataFrame(
                    matrix_pct,
                    index=[f"{home_team} {i}" for i in range(7)],
                    columns=[f"{away_team} {j}" for j in range(7)]
                )
                st.dataframe(
                    heatmap_df.style.background_gradient(cmap="YlOrRd"),
                    use_container_width=True
                )

                # ── Data info ──
                st.divider()
                st.markdown("### ℹ️ Data Used for Prediction")
                st.info(f"Only matches **before {match_date}** were used.")
                match_date_dt = pd.to_datetime(str(match_date))
                info_rows = []
                for name, df in PLAYERS.items():
                    past = df[df["Date"] < match_date_dt]
                    if len(past) > 0:
                        info_rows.append({
                            "Player": name,
                            "Matches Used": len(past),
                            "Latest Match": str(past["Date"].max().date())
                        })
                if info_rows:
                    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.exception(e)

# ── Footer ──
st.divider()
st.markdown(
    "📌 **Add more players:** Upload player CSVs to `player_data/` folder in your GitHub repo. "
    "They auto-appear in squad dropdowns."
)
