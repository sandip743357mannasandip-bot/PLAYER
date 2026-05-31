"""
app.py — Streamlit Dashboard with Formation Selector
Fixed path handling for Streamlit Cloud deployment
"""

import os, sys
import streamlit as st
import pandas as pd
import numpy as np

# ── Ensure backend is importable on Streamlit Cloud ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import (
    load_all_players, get_all_clubs, get_squad,
    get_players_by_group, predict_match,
    FORMATIONS, SLOT_TO_GROUP
)

# ──────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ La Liga Predictor 2024-25",
    page_icon="⚽",
    layout="wide"
)

# ──────────────────────────────────────────────────────
# LOAD DATA — cached so it doesn't reload on every click
# ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    players = load_all_players()
    return players

PLAYERS   = load_data()
SEASON    = "2024-25"
ALL_CLUBS = get_all_clubs(PLAYERS, SEASON)

# ──────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────
st.title("⚽ La Liga Match Predictor — 2024-25")
st.markdown(
    "Select teams, pick a match date, choose your **formation**, "
    "then assign players to each position slot. "
    "The model uses **only data before the selected date**."
)

if not ALL_CLUBS:
    st.error(
        "❌ No player data found. "
        "Make sure CSV files are inside the `player_data/` folder in your repo."
    )
    st.stop()

st.divider()

# ──────────────────────────────────────────────────────
# SECTION 1: TEAM & DATE
# ──────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    home_team = st.selectbox("🏠 Home Team", ALL_CLUBS, index=0)
with c2:
    away_opts = [c for c in ALL_CLUBS if c != home_team]
    away_team = st.selectbox("✈️ Away Team", away_opts, index=0)
with c3:
    match_date = st.date_input(
        "📅 Match Date",
        value=pd.to_datetime("2025-03-01"),
        min_value=pd.to_datetime("2024-08-01"),
        max_value=pd.to_datetime("2025-07-01"),
    )

st.divider()

# ──────────────────────────────────────────────────────
# SECTION 2: FORMATION SELECTOR
# ──────────────────────────────────────────────────────
st.markdown("### 🗂️ Select Formations")
fc1, fc2 = st.columns(2)
with fc1:
    home_formation = st.selectbox(
        f"🏠 {home_team} Formation",
        list(FORMATIONS.keys()),
        index=0,
        key="home_form"
    )
with fc2:
    away_formation = st.selectbox(
        f"✈️ {away_team} Formation",
        list(FORMATIONS.keys()),
        index=0,
        key="away_form"
    )

st.divider()

# ──────────────────────────────────────────────────────
# SECTION 3: PLAYING XI BY POSITION SLOTS
# ──────────────────────────────────────────────────────
st.markdown("### 👕 Select Playing XI")
st.caption("Each slot shows only players who play that position for the selected club in 2024-25.")

home_slots = FORMATIONS[home_formation]   # e.g. ["GK","RB","CB","CB","LB",...]
away_slots = FORMATIONS[away_formation]

home_xi = []
away_xi = []

col_home, col_away = st.columns(2)

with col_home:
    st.markdown(f"#### 🏠 {home_team} — {home_formation}")
    all_home_squad = get_squad(PLAYERS, home_team, SEASON)
    if not all_home_squad:
        st.warning(f"No players found for {home_team}")
    for i, slot in enumerate(home_slots):
        options = all_home_squad if all_home_squad else ["No players"]
        player  = st.selectbox(
            f"{slot} — Player {i+1}",
            options,
            index=min(i, len(options)-1),
            key=f"home_slot_{i}"
        )
        home_xi.append(player)

with col_away:
    st.markdown(f"#### ✈️ {away_team} — {away_formation}")
    all_away_squad = get_squad(PLAYERS, away_team, SEASON)
    if not all_away_squad:
        st.warning(f"No players found for {away_team}")
    for i, slot in enumerate(away_slots):
        options = all_away_squad if all_away_squad else ["No players"]
        player  = st.selectbox(
            f"{slot} — Player {i+1}",
            options,
            index=min(i, len(options)-1),
            key=f"away_slot_{i}"
        )
        away_xi.append(player)

st.divider()

# ──────────────────────────────────────────────────────
# SECTION 4: PREDICT
# ──────────────────────────────────────────────────────
predict_btn = st.button("🔮 PREDICT MATCH", type="primary", use_container_width=True)

if predict_btn:
    valid_home = [p for p in home_xi if p]
    valid_away = [p for p in away_xi if p]

    if not valid_home or not valid_away:
        st.error("⚠️ Please make sure players are selected for both teams.")
    else:
        with st.spinner("Running prediction..."):
            try:
                result = predict_match(
                    PLAYERS,
                    home_team, away_team,
                    valid_home, valid_away,
                    str(match_date),
                    season=SEASON
                )

                st.divider()
                st.markdown("## 📊 Prediction Results")
                st.markdown(
                    f"**{home_team}** ({home_formation})  vs  "
                    f"**{away_team}** ({away_formation})  |  📅 {match_date}"
                )

                # ── xG ──
                st.markdown("### ⚽ Expected Goals (xG)")
                xc1, xc2, xc3 = st.columns(3)
                xc1.metric(f"🏠 {home_team}", result["xg_home"])
                xc2.metric("VS", "—")
                xc3.metric(f"✈️ {away_team}", result["xg_away"])

                st.divider()

                # ── Win/Draw/Loss ──
                st.markdown("### 📈 Match Probabilities")
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric(f"🏠 {home_team} Win", f"{result['home_win']}%")
                pc2.metric("🤝 Draw",              f"{result['draw']}%")
                pc3.metric(f"✈️ {away_team} Win",  f"{result['away_win']}%")

                prob_df = pd.DataFrame({
                    "Outcome": [f"{home_team} Win", "Draw", f"{away_team} Win"],
                    "Probability (%)": [result["home_win"], result["draw"], result["away_win"]]
                })
                st.bar_chart(prob_df.set_index("Outcome"))

                st.divider()

                # ── Top 5 Scorelines ──
                st.markdown(f"### 🎯 Top 5 Most Likely Scorelines")
                st.caption(f"Home: {home_team} — Away: {away_team}")
                top5_df = pd.DataFrame(result["top5"], columns=["Scoreline","Probability (%)"])
                st.dataframe(top5_df, use_container_width=True, hide_index=True)

                st.divider()

                # ── Heatmap ──
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

                st.divider()

                # ── XI Summary ──
                st.markdown("### 📋 Playing XI Used")
                xi_c1, xi_c2 = st.columns(2)
                with xi_c1:
                    st.markdown(f"**🏠 {home_team} ({home_formation})**")
                    for slot, player in zip(home_slots, home_xi):
                        st.write(f"**{slot}** — {player}")
                with xi_c2:
                    st.markdown(f"**✈️ {away_team} ({away_formation})**")
                    for slot, player in zip(away_slots, away_xi):
                        st.write(f"**{slot}** — {player}")

                st.divider()

                # ── Data info ──
                st.markdown("### ℹ️ Data Used")
                st.info(f"Only matches before **{match_date}** were used for prediction.")
                mdt = pd.to_datetime(str(match_date))
                info_rows = []
                for name, df in PLAYERS.items():
                    past = df[df["Date"] < mdt]
                    if len(past) > 0:
                        info_rows.append({
                            "Player": name,
                            "Matches Used": len(past),
                            "Latest Match Used": str(past["Date"].max().date())
                        })
                if info_rows:
                    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.exception(e)

# ── Footer ──
st.divider()
st.markdown(
    "📌 **To add more players:** Upload their CSV to `player_data/` in your GitHub repo. "
    "They will automatically appear in the squad dropdowns."
)
