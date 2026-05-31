"""
app.py — Streamlit Dashboard with proper XI selection:
- Each slot shows full squad
- Selected players removed from remaining slots
- Duplicate player error shown before prediction
"""

import os, sys
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import (
    load_all_players, get_all_clubs, get_squad,
    predict_match, FORMATIONS, SLOT_TO_GROUP
)

st.set_page_config(
    page_title="⚽ La Liga Predictor 2024-25",
    page_icon="⚽",
    layout="wide"
)

@st.cache_data
def load_data():
    return load_all_players()

PLAYERS   = load_data()
SEASON    = "2024-25"
ALL_CLUBS = get_all_clubs(PLAYERS, SEASON)

st.title("⚽ La Liga Match Predictor — 2024-25")
st.markdown(
    "Select teams, pick a match date, choose your **formation**, "
    "then assign players to each position slot. "
    "The model uses **only data before the selected date**."
)

if not ALL_CLUBS:
    st.error("❌ No player data found. Make sure CSV files are inside the `PLAYER DATA/` folder in your repo.")
    st.stop()

st.divider()

# ── SECTION 1: TEAM & DATE ──
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

# ── SECTION 2: FORMATION ──
st.markdown("### 🗂️ Select Formations")
fc1, fc2 = st.columns(2)
with fc1:
    home_formation = st.selectbox(f"🏠 {home_team} Formation", list(FORMATIONS.keys()), index=0, key="home_form")
with fc2:
    away_formation = st.selectbox(f"✈️ {away_team} Formation", list(FORMATIONS.keys()), index=0, key="away_form")

home_slots = FORMATIONS[home_formation]
away_slots = FORMATIONS[away_formation]

st.divider()

# ── SECTION 3: PLAYING XI ──
st.markdown("### 👕 Select Playing XI")
st.caption("Once a player is selected in one slot, they are removed from all remaining slots.")

home_squad = get_squad(PLAYERS, home_team, SEASON)
away_squad = get_squad(PLAYERS, away_team, SEASON)

home_xi = []
away_xi = []

col_home, col_away = st.columns(2)

# ── HOME XI ──
with col_home:
    st.markdown(f"#### 🏠 {home_team} — {home_formation}")
    if not home_squad:
        st.warning(f"No players found for {home_team}. Add their CSVs to PLAYER DATA/")
    else:
        for i, slot in enumerate(home_slots):
            # Build available list: full squad minus already picked
            picked_so_far = [p for p in home_xi if p]
            remaining     = [p for p in home_squad if p not in picked_so_far]
            if not remaining:
                remaining = home_squad  # all used, allow repeats as last resort

            selected = st.selectbox(
                f"**{slot}** — Slot {i+1}",
                options=remaining,
                index=0,
                key=f"home_{i}"
            )
            home_xi.append(selected)

# ── AWAY XI ──
with col_away:
    st.markdown(f"#### ✈️ {away_team} — {away_formation}")
    if not away_squad:
        st.warning(f"No players found for {away_team}. Add their CSVs to PLAYER DATA/")
    else:
        for i, slot in enumerate(away_slots):
            picked_so_far = [p for p in away_xi if p]
            remaining     = [p for p in away_squad if p not in picked_so_far]
            if not remaining:
                remaining = away_squad

            selected = st.selectbox(
                f"**{slot}** — Slot {i+1}",
                options=remaining,
                index=0,
                key=f"away_{i}"
            )
            away_xi.append(selected)

st.divider()

# ── VALIDATION ──
home_duplicates = [p for p in set(home_xi) if home_xi.count(p) > 1]
away_duplicates = [p for p in set(away_xi)  if away_xi.count(p)  > 1]

if home_duplicates:
    st.error(f"❌ {home_team}: **{', '.join(home_duplicates)}** selected more than once. Each player can only play one position.")
if away_duplicates:
    st.error(f"❌ {away_team}: **{', '.join(away_duplicates)}** selected more than once. Each player can only play one position.")

# ── PREDICT BUTTON ──
predict_btn = st.button(
    "🔮 PREDICT MATCH",
    type="primary",
    use_container_width=True,
    disabled=bool(home_duplicates or away_duplicates)
)

if predict_btn:
    valid_home = [p for p in home_xi if p]
    valid_away = [p for p in away_xi  if p]

    if not valid_home or not valid_away:
        st.error("⚠️ Please select players for both teams.")
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
                    f"**{home_team}** ({home_formation}) vs "
                    f"**{away_team}** ({away_formation}) | 📅 {match_date}"
                )

                # xG
                st.markdown("### ⚽ Expected Goals (xG)")
                xc1, xc2, xc3 = st.columns(3)
                xc1.metric(f"🏠 {home_team}", result["xg_home"])
                xc2.metric("VS", "—")
                xc3.metric(f"✈️ {away_team}", result["xg_away"])
                st.divider()

                # Probabilities
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

                # Top 5
                st.markdown("### 🎯 Top 5 Most Likely Scorelines")
                st.caption(f"Home: {home_team} — Away: {away_team}")
                top5_df = pd.DataFrame(result["top5"], columns=["Scoreline","Probability (%)"])
                st.dataframe(top5_df, use_container_width=True, hide_index=True)
                st.divider()

                # Heatmap
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

                # XI Summary
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

                # Data info
                st.markdown("### ℹ️ Data Used")
                st.info(f"Only matches strictly before **{match_date}** were used.")
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

st.divider()
st.markdown("📌 **Add more players:** Upload CSV to `PLAYER DATA/` folder in GitHub repo.")
