"""
backend.py — Robust version:
- Finds player_data folder regardless of name/case
- Handles all filename formats (with/without - Sheet1)
- Detects 2024-25 season by DATE RANGE (not Season string)
  so all players are found regardless of how season is labeled
- Falls back gracefully when position has no players
"""

import os, math, glob, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Season date range — any match between these dates = 2024-25
SEASON_START = pd.Timestamp("2024-07-01")
SEASON_END   = pd.Timestamp("2025-06-30")

# ── Find player_data folder automatically ──
def find_player_data_dir():
    candidates = [
        "PLAYER DATA", "player_data", "Player Data",
        "PLAYER_DATA", "PlayerData", "data", "DATA",
    ]
    for c in candidates:
        p = os.path.join(BASE_DIR, c)
        if os.path.isdir(p) and glob.glob(os.path.join(p, "*.csv")):
            return p
    # fallback: any subfolder with CSVs
    for item in os.listdir(BASE_DIR):
        full = os.path.join(BASE_DIR, item)
        if os.path.isdir(full) and glob.glob(os.path.join(full, "*.csv")):
            return full
    return BASE_DIR

PLAYER_DATA_DIR = find_player_data_dir()

POSITION_MAP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "WB": "DEF", "SW": "DEF",
    "DM": "MID", "CM": "MID", "LM": "MID", "RM": "MID", "AM": "MID",
    "LW": "FWD", "RW": "FWD", "FW": "FWD", "ST": "FWD", "CF": "FWD",
    "-":  "MID",
}

FORMATIONS = {
    "4-3-3":   ["GK","RB","CB","CB","LB","CM","CM","CM","RW","ST","LW"],
    "4-4-2":   ["GK","RB","CB","CB","LB","RM","CM","CM","LM","ST","ST"],
    "4-2-3-1": ["GK","RB","CB","CB","LB","DM","DM","AM","RW","LW","ST"],
    "3-5-2":   ["GK","CB","CB","CB","RM","CM","CM","CM","LM","ST","ST"],
    "3-4-3":   ["GK","CB","CB","CB","RM","CM","CM","LM","RW","ST","LW"],
    "5-3-2":   ["GK","RB","CB","CB","CB","LB","CM","CM","CM","ST","ST"],
    "4-1-4-1": ["GK","RB","CB","CB","LB","DM","RM","CM","CM","LM","ST"],
}

SLOT_TO_GROUP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF",
    "DM": "MID", "CM": "MID", "LM": "MID", "RM": "MID", "AM": "MID",
    "ST": "FWD", "LW": "FWD", "RW": "FWD",
}

# ─────────────────────────────────────────
# Clean filename → player name
# "Vinicius Junior - Sheet1.csv" → "Vinicius Junior"
# ─────────────────────────────────────────
def clean_player_name(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0]
    if " - " in name:
        name = name.split(" - ")[0]
    return name.strip()

# ─────────────────────────────────────────
# 1. LOAD ALL PLAYER CSVs
# ─────────────────────────────────────────
def load_all_players():
    players = {}
    for path in glob.glob(os.path.join(PLAYER_DATA_DIR, "*.csv")):
        name = clean_player_name(path)
        try:
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            # Add a clean season flag based on date range
            df["is_2024_25"] = (df["Date"] >= SEASON_START) & (df["Date"] <= SEASON_END)
            stat_cols = ["Goals","Assists","Shots","SoT","Minutes",
                         "TacklesWon","Interceptions","Crosses","Fouls",
                         "TeamGoals","OppGoals"]
            for col in stat_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            players[name] = df
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
            continue
    return players

# ─────────────────────────────────────────
# 2. GET ALL CLUBS — using date range
# ─────────────────────────────────────────
def get_all_clubs(players_dict, season="2024-25"):
    clubs = set()
    for df in players_dict.values():
        clubs.update(df[df["is_2024_25"]]["Team"].dropna().unique())
    return sorted(clubs)

# ─────────────────────────────────────────
# 3. GET SQUAD WITH POSITIONS — using date range
# ─────────────────────────────────────────
def get_squad_with_positions(players_dict, club, season="2024-25"):
    squad = {}
    for name, df in players_dict.items():
        club_df = df[df["is_2024_25"] & (df["Team"] == club)]
        if len(club_df) > 0:
            pos = club_df["Position"].mode()
            squad[name] = pos[0] if len(pos) > 0 else "-"
    return squad

def get_squad(players_dict, club, season="2024-25"):
    return sorted(get_squad_with_positions(players_dict, club, season).keys())

# ─────────────────────────────────────────
# 4. GET PLAYERS BY POSITION GROUP
# Falls back to full squad if no players found for group
# ─────────────────────────────────────────
def get_players_by_group(players_dict, club, group, season="2024-25"):
    squad = get_squad_with_positions(players_dict, club, season)
    filtered = [n for n, pos in squad.items()
                if POSITION_MAP.get(str(pos).upper(), "MID") == group]
    if filtered:
        return sorted(filtered)
    # Fallback: if no players match the position, return all squad members
    # so the slot is never empty
    return sorted(squad.keys())

# ─────────────────────────────────────────
# 5. AGGREGATE XI FEATURES
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
        last5 = past.tail(5)
        per90 = max(last5["Minutes"].sum() / 90, 0.1)
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
# 6. BUILD TRAINING DATA — using date range
# ─────────────────────────────────────────
def build_training_data(players_dict, club, season="2024-25"):
    club_matches = []
    for name, df in players_dict.items():
        cdf = df[df["is_2024_25"] & (df["Team"] == club)]
        cdf = cdf[["Date","TeamGoals","OppGoals","Venue"]].drop_duplicates()
        if len(cdf) > 0:
            club_matches.append(cdf)
    if not club_matches:
        return pd.DataFrame(), pd.Series(dtype=float)

    matches  = pd.concat(club_matches).drop_duplicates("Date").sort_values("Date")
    features, targets = [], []

    for _, row in matches.iterrows():
        mdate    = row["Date"]
        squad    = get_squad(players_dict, club)
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
# 7. TRAIN MODELS
# ─────────────────────────────────────────
def train_models(X, y):
    if len(X) < 5:
        return None, None, 0.5, 0.5
    split = max(1, int(len(X) * 0.8))
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    lr = LinearRegression()
    rf.fit(Xtr, ytr); lr.fit(Xtr, ytr)
    mae_rf = mean_absolute_error(yte, rf.predict(Xte)) if len(Xte) > 0 else 1.0
    mae_lr = mean_absolute_error(yte, lr.predict(Xte)) if len(Xte) > 0 else 1.0
    w_rf = 1/(mae_rf+1e-5); w_lr = 1/(mae_lr+1e-5)
    t = w_rf + w_lr
    return rf, lr, w_rf/t, w_lr/t

# ─────────────────────────────────────────
# 8. TRADITIONAL xG
# ─────────────────────────────────────────
def traditional_xg(xi_feats, avg_scored):
    shots = xi_feats.get("shots_per90", 0) * 90
    g_per_shot = avg_scored / max(shots, 1)
    return max(shots * g_per_shot, 0.3)

# ─────────────────────────────────────────
# 9. POISSON
# ─────────────────────────────────────────
def poisson_prob(lam, k):
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def scoreline_matrix(lam_h, lam_a, max_goals=6):
    m = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            m[i][j] = poisson_prob(lam_h, i) * poisson_prob(lam_a, j)
    return m

# ─────────────────────────────────────────
# 10. MAIN PREDICT
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

        xg_trad  = traditional_xg(xi_feats, avg_scored)
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

        X, y = build_training_data(players_dict, team)
        if len(X) >= 5:
            rf, lr, w_rf, w_lr = train_models(X, y)
            xg_ml    = w_rf * max(rf.predict(feat_row)[0], 0) + \
                       w_lr * max(lr.predict(feat_row)[0], 0)
            xg_final = 0.6 * xg_ml + 0.4 * xg_trad
        else:
            xg_final = xg_trad

        xg_results[team] = max(round(xg_final, 3), 0.1)

    lam_h  = xg_results[home_team]
    lam_a  = xg_results[away_team]
    matrix = scoreline_matrix(lam_h, lam_a)
    win    = float(np.sum(np.tril(matrix, -1)))
    draw   = float(np.sum(np.diag(matrix)))
    loss   = float(np.sum(np.triu(matrix, 1)))
    scores = [(i, j, matrix[i][j]) for i in range(7) for j in range(7)]
    top5   = sorted(scores, key=lambda x: -x[2])[:5]

    return {
        "xg_home":  lam_h, "xg_away":  lam_a,
        "home_win": round(win*100,1), "draw": round(draw*100,1),
        "away_win": round(loss*100,1),
        "top5":     [(f"{h}-{a}", round(p*100,2)) for h,a,p in top5],
        "matrix":   matrix,
        "home_team":home_team, "away_team":away_team,
    }
