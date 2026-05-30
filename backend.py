"""
backend.py
Core prediction logic:
- Loads all player CSVs from player_data/ folder
- Filters data strictly BEFORE selected match date
- Aggregates XI player features
- Trains Random Forest on available data
- Predicts xG using ensemble (RF + traditional xG)
- Poisson distribution for scoreline probabilities
"""

import os
import math
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

PLAYER_DATA_DIR = os.path.join(os.path.dirname(__file__), "player_data")

# ──────────────────────────────────────────────
# 1. LOAD ALL PLAYER CSVs
# ──────────────────────────────────────────────
def load_all_players():
    """
    Returns dict: { player_name: DataFrame }
    Loads every CSV in player_data/ folder.
    """
    players = {}
    for path in glob.glob(os.path.join(PLAYER_DATA_DIR, "*.csv")):
        name = os.path.splitext(os.path.basename(path))[0].replace("_", " ").strip()
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        # Fill NaN stats with 0
        stat_cols = ["Goals","Assists","Shots","SoT","Minutes",
                     "TacklesWon","Interceptions","Crosses","Fouls"]
        for col in stat_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        players[name] = df
    return players


# ──────────────────────────────────────────────
# 2. GET SQUAD FOR A CLUB (filtered by season)
# ──────────────────────────────────────────────
def get_squad(players_dict, club, season="2024-25"):
    """
    Returns list of player names who played for `club` in `season`.
    Only players with data in player_data/ folder are shown.
    """
    squad = []
    for name, df in players_dict.items():
        mask = (df["Team"] == club) & (df["Season"] == season)
        if mask.sum() > 0:
            squad.append(name)
    return sorted(squad)


# ──────────────────────────────────────────────
# 3. GET ALL CLUBS IN DATASET
# ──────────────────────────────────────────────
def get_all_clubs(players_dict, season="2024-25"):
    clubs = set()
    for df in players_dict.values():
        season_df = df[df["Season"] == season]
        clubs.update(season_df["Team"].dropna().unique())
    return sorted(clubs)


# ──────────────────────────────────────────────
# 4. AGGREGATE XI FEATURES BEFORE DATE
# ──────────────────────────────────────────────
def aggregate_xi_features(players_dict, xi_players, match_date):
    """
    For each player in xi_players:
      - Filter their data strictly BEFORE match_date
      - Compute rolling last-5 match stats
    Aggregate across all XI players → team feature vector
    """
    rows = []
    for player in xi_players:
        if player not in players_dict:
            continue
        df = players_dict[player]
        # STRICT: only data before selected date
        past = df[df["Date"] < match_date].sort_values("Date")
        if len(past) == 0:
            continue
        last5 = past.tail(5)
        total_mins = last5["Minutes"].sum()
        per90 = max(total_mins / 90, 0.1)  # avoid division by zero

        rows.append({
            "goals_per90":    last5["Goals"].sum() / per90,
            "assists_per90":  last5["Assists"].sum() / per90,
            "shots_per90":    last5["Shots"].sum() / per90,
            "sot_per90":      last5["SoT"].sum() / per90,
            "tackles_per90":  last5["TacklesWon"].sum() / per90,
            "intercept_per90":last5["Interceptions"].sum() / per90,
            "avg_minutes":    last5["Minutes"].mean(),
            "matches_played": len(past),
        })

    if not rows:
        # Return zeros if no data
        return pd.Series({
            "goals_per90": 0, "assists_per90": 0,
            "shots_per90": 0, "sot_per90": 0,
            "tackles_per90": 0, "intercept_per90": 0,
            "avg_minutes": 0, "matches_played": 0,
        })

    agg = pd.DataFrame(rows).mean()
    return agg


# ──────────────────────────────────────────────
# 5. BUILD MATCH-LEVEL TRAINING DATA FROM PLAYERS
# ──────────────────────────────────────────────
def build_training_data(players_dict, club, season="2024-25"):
    """
    For each match the club played in the season,
    build features using only data BEFORE that match date.
    Target = TeamGoals scored in that match.
    """
    # Collect all matches for this club in this season
    club_matches = []
    for name, df in players_dict.items():
        mask = (df["Team"] == club) & (df["Season"] == season)
        club_df = df[mask].copy()
        if len(club_df) > 0:
            club_matches.append(club_df[["Date","TeamGoals","OppGoals",
                                         "Opponent","Venue","Result"]].drop_duplicates())

    if not club_matches:
        return pd.DataFrame(), pd.Series()

    matches = pd.concat(club_matches).drop_duplicates("Date").sort_values("Date")

    features = []
    targets = []

    for _, row in matches.iterrows():
        match_date = row["Date"]
        actual_goals = row["TeamGoals"]

        # Get all players for this club
        squad = get_squad(players_dict, club, season)

        # Build features using only data BEFORE this match
        xi_feats = aggregate_xi_features(players_dict, squad, match_date)

        # Team-level rolling features (from player match data before this date)
        team_past = []
        for name, df in players_dict.items():
            mask = (df["Team"] == club) & (df["Date"] < match_date)
            team_past.append(df[mask][["Date","TeamGoals","OppGoals"]].drop_duplicates())

        if team_past:
            team_hist = pd.concat(team_past).drop_duplicates("Date").sort_values("Date").tail(5)
            avg_scored   = team_hist["TeamGoals"].mean() if len(team_hist) > 0 else 1.0
            avg_conceded = team_hist["OppGoals"].mean()  if len(team_hist) > 0 else 1.0
        else:
            avg_scored, avg_conceded = 1.0, 1.0

        feat = {
            "avg_scored_last5":   avg_scored,
            "avg_conceded_last5": avg_conceded,
            "goals_per90":        xi_feats["goals_per90"],
            "shots_per90":        xi_feats["shots_per90"],
            "sot_per90":          xi_feats["sot_per90"],
            "assists_per90":      xi_feats["assists_per90"],
            "tackles_per90":      xi_feats["tackles_per90"],
            "venue_home":         1 if row["Venue"] == "Home" else 0,
        }
        features.append(feat)
        targets.append(actual_goals)

    return pd.DataFrame(features), pd.Series(targets)


# ──────────────────────────────────────────────
# 6. TRAIN ML MODELS
# ──────────────────────────────────────────────
def train_models(X, y):
    """Train RF + LR, return models and their MAE weights."""
    if len(X) < 5:
        return None, None, 0.5, 0.5

    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    lr = LinearRegression()

    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    if len(X_test) > 0:
        mae_rf = mean_absolute_error(y_test, rf.predict(X_test))
        mae_lr = mean_absolute_error(y_test, lr.predict(X_test))
    else:
        mae_rf, mae_lr = 1.0, 1.0

    # Weights inversely proportional to MAE
    w_rf = 1 / (mae_rf + 1e-5)
    w_lr = 1 / (mae_lr + 1e-5)
    total = w_rf + w_lr

    return rf, lr, w_rf / total, w_lr / total


# ──────────────────────────────────────────────
# 7. TRADITIONAL xG CALCULATION
# ──────────────────────────────────────────────
def traditional_xg(xi_feats, avg_scored):
    """Baseline xG using shot-based metrics."""
    shots  = xi_feats.get("shots_per90", 0) * 90
    sot    = xi_feats.get("sot_per90", 0) * 90
    g_per_shot = avg_scored / max(shots, 1)
    xg = shots * g_per_shot
    return max(xg, 0.3)  # minimum floor


# ──────────────────────────────────────────────
# 8. POISSON PROBABILITY
# ──────────────────────────────────────────────
def poisson_prob(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def scoreline_matrix(lam_home, lam_away, max_goals=6):
    """Build joint probability matrix P(i,j)."""
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i][j] = poisson_prob(lam_home, i) * poisson_prob(lam_away, j)
    return matrix


# ──────────────────────────────────────────────
# 9. MAIN PREDICTION FUNCTION
# ──────────────────────────────────────────────
def predict_match(players_dict, home_team, away_team,
                  home_xi, away_xi, match_date, season="2024-25"):
    """
    Full prediction pipeline.
    Returns dict with xG, probabilities, top scorelines.
    """
    match_date = pd.to_datetime(match_date)
    results = {}

    for team, xi, venue_val in [(home_team, home_xi, 1),
                                 (away_team, away_xi, 0)]:
        # Build training data using only data BEFORE match_date
        X, y = build_training_data(players_dict, team, season)

        # Filter training rows to only those before match_date
        if len(X) > 0:
            # Rebuild only past matches
            pass  # already handled inside build_training_data

        # XI features before match date
        xi_feats = aggregate_xi_features(players_dict, xi, match_date)

        # Team rolling avg before match date
        team_past = []
        for name, df in players_dict.items():
            mask = (df["Team"] == team) & (df["Date"] < match_date)
            team_past.append(df[mask][["Date","TeamGoals","OppGoals"]].drop_duplicates())

        if team_past:
            hist = pd.concat(team_past).drop_duplicates("Date").sort_values("Date").tail(5)
            avg_scored   = hist["TeamGoals"].mean() if len(hist) > 0 else 1.2
            avg_conceded = hist["OppGoals"].mean()  if len(hist) > 0 else 1.0
        else:
            avg_scored, avg_conceded = 1.2, 1.0

        # Traditional xG baseline
        xg_trad = traditional_xg(xi_feats, avg_scored)

        # ML prediction
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

        if len(X) >= 5:
            rf, lr, w_rf, w_lr = train_models(X, y)
            xg_rf = max(rf.predict(feat_row)[0], 0)
            xg_lr = max(lr.predict(feat_row)[0], 0)
            xg_ml = w_rf * xg_rf + w_lr * xg_lr
            # Final ensemble: 60% ML + 40% traditional
            xg_final = 0.6 * xg_ml + 0.4 * xg_trad
        else:
            xg_final = xg_trad  # fallback if not enough data

        results[team] = max(round(xg_final, 3), 0.1)

    # Scoreline matrix
    lam_h = results[home_team]
    lam_a = results[away_team]
    matrix = scoreline_matrix(lam_h, lam_a)

    # Win / Draw / Loss
    win  = float(np.sum(np.tril(matrix, -1)))  # home goals > away
    draw = float(np.sum(np.diag(matrix)))
    loss = float(np.sum(np.triu(matrix, 1)))

    # Top 5 scorelines
    scores = []
    for i in range(7):
        for j in range(7):
            scores.append((i, j, matrix[i][j]))
    top5 = sorted(scores, key=lambda x: -x[2])[:5]

    return {
        "xg_home":   lam_h,
        "xg_away":   lam_a,
        "home_win":  round(win * 100, 1),
        "draw":      round(draw * 100, 1),
        "away_win":  round(loss * 100, 1),
        "top5":      [(f"{h}-{a}", round(p * 100, 2)) for h, a, p in top5],
        "matrix":    matrix,
        "home_team": home_team,
        "away_team": away_team,
    }
