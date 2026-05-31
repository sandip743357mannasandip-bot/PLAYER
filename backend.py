"""
backend.py — Fixed path handling + position support
"""

import os, math, glob, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# ── Path: always relative to this file ──
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PLAYER_DATA_DIR = os.path.join(BASE_DIR, "player_data")

# Position groups for mapping
POSITION_MAP = {
    "GK":  "GK",
    "CB":  "DEF", "LB": "DEF", "RB": "DEF", "WB": "DEF",
    "DM":  "MID", "CM": "MID", "LM": "MID", "RM": "MID", "AM": "MID",
    "LW":  "FWD", "RW": "FWD", "FW": "FWD", "ST": "FWD",
    "-":   "MID",
}

# ─────────────────────────────────────────
# 1. LOAD ALL PLAYER CSVs
# ─────────────────────────────────────────
def load_all_players():
    players = {}
    csv_files = glob.glob(os.path.join(PLAYER_DATA_DIR, "*.csv"))
    for path in csv_files:
        name = os.path.splitext(os.path.basename(path))[0].replace("_", " ").strip()
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        stat_cols = ["Goals","Assists","Shots","SoT","Minutes",
                     "TacklesWon","Interceptions","Crosses","Fouls"]
        for col in stat_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        players[name] = df
    return players

# ─────────────────────────────────────────
# 2. GET ALL CLUBS
# ─────────────────────────────────────────
def get_all_clubs(players_dict, season="2024-25"):
    clubs = set()
    for df in players_dict.values():
        season_df = df[df["Season"] == season]
        clubs.update(season_df["Team"].dropna().unique())
    return sorted(clubs)

# ─────────────────────────────────────────
# 3. GET SQUAD WITH POSITIONS
# ─────────────────────────────────────────
def get_squad_with_positions(players_dict, club, season="2024-25"):
    """
    Returns dict: { player_name: primary_position }
    Only players who played for club in that season.
    """
    squad = {}
    for name, df in players_dict.items():
        mask = (df["Team"] == club) & (df["Season"] == season)
        club_df = df[mask]
        if len(club_df) > 0:
            # Most frequent position
            pos = club_df["Position"].mode()
            primary_pos = pos[0] if len(pos) > 0 else "-"
            squad[name] = primary_pos
    return squad  # { "Vinicius Junior": "FW", "Thibaut Courtois": "GK", ... }

def get_squad(players_dict, club, season="2024-25"):
    return sorted(get_squad_with_positions(players_dict, club, season).keys())

# ─────────────────────────────────────────
# 4. GET PLAYERS BY POSITION GROUP
# ─────────────────────────────────────────
def get_players_by_group(players_dict, club, group, season="2024-25"):
    """
    group: 'GK', 'DEF', 'MID', 'FWD'
    Returns list of player names matching that group.
    """
    squad = get_squad_with_positions(players_dict, club, season)
    result = []
    for name, pos in squad.items():
        mapped = POSITION_MAP.get(pos.upper(), "MID")
        if mapped == group:
            result.append(name)
    # fallback: include all if none found for group
    if not result:
        result = list(squad.keys())
    return sorted(result)

# ─────────────────────────────────────────
# 5. FORMATION SLOT DEFINITIONS
# ─────────────────────────────────────────
FORMATIONS = {
    "4-3-3":  ["GK","RB","CB","CB","LB","CM","CM","CM","RW","ST","LW"],
    "4-4-2":  ["GK","RB","CB","CB","LB","RM","CM","CM","LM","ST","ST"],
    "4-2-3-1":["GK","RB","CB","CB","LB","DM","DM","AM","RW","LW","ST"],
    "3-5-2":  ["GK","CB","CB","CB","RM","CM","CM","CM","LM","ST","ST"],
    "3-4-3":  ["GK","CB","CB","CB","RM","CM","CM","LM","RW","ST","LW"],
    "5-3-2":  ["GK","RB","CB","CB","CB","LB","CM","CM","CM","ST","ST"],
    "4-1-4-1":["GK","RB","CB","CB","LB","DM","RM","CM","CM","LM","ST"],
}

SLOT_TO_GROUP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF",
    "DM": "MID", "CM": "MID", "LM": "MID", "RM": "MID", "AM": "MID",
    "ST": "FWD", "LW": "FWD", "RW": "FWD",
}

# ─────────────────────────────────────────
# 6. AGGREGATE XI FEATURES
# ─────────────────────────────────────────
def aggregate_xi_features(players_dict, xi_players, match_date):
    rows = []
    for player in xi_players:
        if not player or player not in players_dict:
            continue
        df = players_dict[player]
        past = df[df["Date"] < match_date].sort_values("Date")
        if len(past) == 0:
            continue
        last5  = past.tail(5)
        per90  = max(last5["Minutes"].sum() / 90, 0.1)
        rows.append({
            "goals_per90":     last5["Goals"].sum() / per90,
            "assists_per90":   last5["Assists"].sum() / per90,
            "shots_per90":     last5["Shots"].sum() / per90,
            "sot_per90":       last5["SoT"].sum() / per90,
            "tackles_per90":   last5["TacklesWon"].sum() / per90,
            "intercept_per90": last5["Interceptions"].sum() / per90,
            "avg_minutes":     last5["Minutes"].mean(),
        })
    if not rows:
        return pd.Series({k: 0 for k in
            ["goals_per90","assists_per90","shots_per90","sot_per90",
             "tackles_per90","intercept_per90","avg_minutes"]})
    return pd.DataFrame(rows).mean()

# ─────────────────────────────────────────
# 7. BUILD TRAINING DATA
# ─────────────────────────────────────────
def build_training_data(players_dict, club, season="2024-25"):
    club_matches = []
    for name, df in players_dict.items():
        mask = (df["Team"] == club) & (df["Season"] == season)
        cdf  = df[mask][["Date","TeamGoals","OppGoals","Venue"]].drop_duplicates()
        if len(cdf) > 0:
            club_matches.append(cdf)
    if not club_matches:
        return pd.DataFrame(), pd.Series(dtype=float)

    matches = pd.concat(club_matches).drop_duplicates("Date").sort_values("Date")
    features, targets = [], []

    for _, row in matches.iterrows():
        mdate = row["Date"]
        squad = get_squad(players_dict, club, season)
        xi_feats = aggregate_xi_features(players_dict, squad, mdate)

        team_past = []
        for nm, df in players_dict.items():
            mask = (df["Team"] == club) & (df["Date"] < mdate)
            team_past.append(df[mask][["Date","TeamGoals","OppGoals"]].drop_duplicates())
        if team_past:
            hist = pd.concat(team_past).drop_duplicates("Date").sort_values("Date").tail(5)
            avg_scored   = hist["TeamGoals"].mean() if len(hist) > 0 else 1.2
            avg_conceded = hist["OppGoals"].mean()  if len(hist) > 0 else 1.0
        else:
            avg_scored, avg_conceded = 1.2, 1.0

        features.append({
            "avg_scored_last5":   avg_scored,
            "avg_conceded_last5": avg_conceded,
            "goals_per90":        xi_feats["goals_per90"],
            "shots_per90":        xi_feats["shots_per90"],
            "sot_per90":          xi_feats["sot_per90"],
            "assists_per90":      xi_feats["assists_per90"],
            "tackles_per90":      xi_feats["tackles_per90"],
            "venue_home":         1 if row["Venue"] == "Home" else 0,
        })
        targets.append(row["TeamGoals"])

    return pd.DataFrame(features), pd.Series(targets)

# ─────────────────────────────────────────
# 8. TRAIN MODELS
# ─────────────────────────────────────────
def train_models(X, y):
    if len(X) < 5:
        return None, None, 0.5, 0.5
    split = max(1, int(len(X) * 0.8))
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    lr = LinearRegression()
    rf.fit(Xtr, ytr)
    lr.fit(Xtr, ytr)
    if len(Xte) > 0:
        mae_rf = mean_absolute_error(yte, rf.predict(Xte))
        mae_lr = mean_absolute_error(yte, lr.predict(Xte))
    else:
        mae_rf = mae_lr = 1.0
    w_rf = 1 / (mae_rf + 1e-5)
    w_lr = 1 / (mae_lr + 1e-5)
    total = w_rf + w_lr
    return rf, lr, w_rf / total, w_lr / total

# ─────────────────────────────────────────
# 9. TRADITIONAL xG
# ─────────────────────────────────────────
def traditional_xg(xi_feats, avg_scored):
    shots = xi_feats.get("shots_per90", 0) * 90
    g_per_shot = avg_scored / max(shots, 1)
    return max(shots * g_per_shot, 0.3)

# ─────────────────────────────────────────
# 10. POISSON
# ─────────────────────────────────────────
def poisson_prob(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def scoreline_matrix(lam_h, lam_a, max_goals=6):
    m = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            m[i][j] = poisson_prob(lam_h, i) * poisson_prob(lam_a, j)
    return m

# ─────────────────────────────────────────
# 11. MAIN PREDICT
# ─────────────────────────────────────────
def predict_match(players_dict, home_team, away_team,
                  home_xi, away_xi, match_date, season="2024-25"):
    match_date = pd.to_datetime(match_date)
    xg_results = {}

    for team, xi, venue_val in [(home_team, home_xi, 1), (away_team, away_xi, 0)]:
        xi_feats = aggregate_xi_features(players_dict, xi, match_date)

        team_past = []
        for nm, df in players_dict.items():
            mask = (df["Team"] == team) & (df["Date"] < match_date)
            team_past.append(df[mask][["Date","TeamGoals","OppGoals"]].drop_duplicates())
        if team_past:
            hist = pd.concat(team_past).drop_duplicates("Date").sort_values("Date").tail(5)
            avg_scored   = hist["TeamGoals"].mean() if len(hist) > 0 else 1.2
            avg_conceded = hist["OppGoals"].mean()  if len(hist) > 0 else 1.0
        else:
            avg_scored, avg_conceded = 1.2, 1.0

        xg_trad = traditional_xg(xi_feats, avg_scored)

        feat_row = pd.DataFrame([{
            "avg_scored_last5":   avg_scored,
            "avg_conceded_last5": avg_conceded,
            "goals_per90":        xi_feats["goals_per90"],
            "shots_per90":        xi_feats["shots_per90"],
            "sot_per90":          xi_feats["sot_per90"],
            "assists_per90":      xi_feats["assists_per90"],
            "tackles_per90":      xi_feats["tackles_per90"],
            "venue_home":         venue_val,
        }])

        X, y = build_training_data(players_dict, team, season)
        if len(X) >= 5:
            rf, lr, w_rf, w_lr = train_models(X, y)
            xg_rf = max(rf.predict(feat_row)[0], 0)
            xg_lr = max(lr.predict(feat_row)[0], 0)
            xg_ml = w_rf * xg_rf + w_lr * xg_lr
            xg_final = 0.6 * xg_ml + 0.4 * xg_trad
        else:
            xg_final = xg_trad

        xg_results[team] = max(round(xg_final, 3), 0.1)

    lam_h = xg_results[home_team]
    lam_a = xg_results[away_team]
    matrix = scoreline_matrix(lam_h, lam_a)

    win  = float(np.sum(np.tril(matrix, -1)))
    draw = float(np.sum(np.diag(matrix)))
    loss = float(np.sum(np.triu(matrix, 1)))

    scores = [(i, j, matrix[i][j]) for i in range(7) for j in range(7)]
    top5   = sorted(scores, key=lambda x: -x[2])[:5]

    return {
        "xg_home":  lam_h,
        "xg_away":  lam_a,
        "home_win": round(win * 100, 1),
        "draw":     round(draw * 100, 1),
        "away_win": round(loss * 100, 1),
        "top5":     [(f"{h}-{a}", round(p*100,2)) for h,a,p in top5],
        "matrix":   matrix,
        "home_team":home_team,
        "away_team":away_team,
    }
