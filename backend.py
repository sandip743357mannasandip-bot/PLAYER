"""
backend.py — Season-aware version
Uses SEASON_DATA.csv to know which player played for which club in which season
Season date ranges defined per season string
"""

import os, math, glob, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Find player data folder ──
def find_player_data_dir():
    for name in ["PLAYER DATA","player_data","Player Data","PLAYER_DATA","data","DATA"]:
        p = os.path.join(BASE_DIR, name)
        if os.path.isdir(p) and glob.glob(os.path.join(p,"*.csv")):
            return p
    for item in os.listdir(BASE_DIR):
        full = os.path.join(BASE_DIR, item)
        if os.path.isdir(full) and glob.glob(os.path.join(full,"*.csv")):
            return full
    return BASE_DIR

PLAYER_DATA_DIR = find_player_data_dir()

# ── Season date ranges ──
SEASON_DATES = {
    "2024-2025": ("2024-07-01","2025-06-30"),
    "2023-2024": ("2023-07-01","2024-06-30"),
    "2022-2023": ("2022-07-01","2023-06-30"),
    "2021-2022": ("2021-07-01","2022-06-30"),
    "2020-2021": ("2020-07-01","2021-06-30"),
    "2019-2020": ("2019-07-01","2020-06-30"),
    "2018-2019": ("2018-07-01","2019-06-30"),
    "2017-2018": ("2017-07-01","2018-06-30"),
    "2016-2017": ("2016-07-01","2017-06-30"),
    "2015-2016": ("2015-07-01","2016-06-30"),
    "2014-2015": ("2014-07-01","2015-06-30"),
    "2013-2014": ("2013-07-01","2014-06-30"),
    "2012-2013": ("2012-07-01","2013-06-30"),
    "2011-2012": ("2011-07-01","2012-06-30"),
    "2010-2011": ("2010-07-01","2011-06-30"),
    "2009-2010": ("2009-07-01","2010-06-30"),
    "2008-2009": ("2008-07-01","2009-06-30"),
}

def get_season_range(season):
    if season in SEASON_DATES:
        s, e = SEASON_DATES[season]
        return pd.Timestamp(s), pd.Timestamp(e)
    # fallback: parse from string like "2024-2025"
    try:
        start_yr = int(season.split("-")[0])
        return pd.Timestamp(f"{start_yr}-07-01"), pd.Timestamp(f"{start_yr+1}-06-30")
    except:
        return pd.Timestamp("2024-07-01"), pd.Timestamp("2025-06-30")

def season_mask(df, season):
    s, e = get_season_range(season)
    return (df["Date"] >= s) & (df["Date"] <= e)

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
    "GK":"GK","CB":"DEF","LB":"DEF","RB":"DEF",
    "DM":"MID","CM":"MID","LM":"MID","RM":"MID","AM":"MID",
    "ST":"FWD","LW":"FWD","RW":"FWD",
}

def clean_player_name(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0]
    if " - " in name:
        name = name.split(" - ")[0]
    return name.strip()

# ─────────────────────────────────────────
# 1. LOAD SEASON DATA (PLAYER → TEAM mapping)
# ─────────────────────────────────────────
def load_season_data():
    """
    Loads SEASON_DATA.csv → { season: { club: [player_names] } }
    Normalises player names to lowercase for fuzzy matching
    """
    path = os.path.join(PLAYER_DATA_DIR, "SEASON_DATA.csv")
    if not os.path.exists(path):
        # also try root
        path = os.path.join(BASE_DIR, "SEASON_DATA.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    result = {}
    for _, row in df.iterrows():
        season = str(row["SEASON"]).strip()
        club   = str(row["TEAM"]).strip()
        player = str(row["PLAYER"]).strip()
        if season not in result:
            result[season] = {}
        if club not in result[season]:
            result[season][club] = []
        result[season][club].append(player)
    return result

# ─────────────────────────────────────────
# 2. LOAD ALL PLAYER CSVs
# ─────────────────────────────────────────
def load_all_players():
    players = {}
    for path in glob.glob(os.path.join(PLAYER_DATA_DIR, "*.csv")):
        if "SEASON_DATA" in os.path.basename(path).upper():
            continue  # skip the season mapping file
        name = clean_player_name(path)
        try:
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).copy()
            for col in ["Goals","Assists","Shots","SoT","Minutes",
                        "TacklesWon","Interceptions","Crosses","Fouls",
                        "TeamGoals","OppGoals"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            players[name] = df
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return players

# ─────────────────────────────────────────
# 3. GET ALL SEASONS available in season data
# ─────────────────────────────────────────
def get_all_seasons(season_data):
    return sorted(season_data.keys(), reverse=True)

# ─────────────────────────────────────────
# 4. GET ALL CLUBS for a season
# ─────────────────────────────────────────
def get_clubs_for_season(season_data, season):
    return sorted(season_data.get(season, {}).keys())

# ─────────────────────────────────────────
# 5. GET SQUAD for a club in a season
#    Matches season_data names → player CSV names (fuzzy)
# ─────────────────────────────────────────
def normalize(name):
    """Lowercase, remove accents roughly, strip spaces"""
    import unicodedata
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.lower().strip()

def get_squad_for_season(season_data, players_dict, club, season):
    """
    Returns list of player CSV names who played for club in season.
    Matches by normalised name comparison.
    """
    season_players = season_data.get(season, {}).get(club, [])
    if not season_players:
        return []

    # Build normalised lookup: norm_name → csv_name
    csv_norm = {normalize(n): n for n in players_dict.keys()}

    matched = []
    for sp in season_players:
        norm_sp = normalize(sp)
        if norm_sp in csv_norm:
            matched.append(csv_norm[norm_sp])
        else:
            # partial match fallback
            for norm_csv, csv_name in csv_norm.items():
                if norm_sp in norm_csv or norm_csv in norm_sp:
                    matched.append(csv_name)
                    break
    return sorted(set(matched))

# ─────────────────────────────────────────
# 6. AGGREGATE XI FEATURES
# ─────────────────────────────────────────
def aggregate_xi_features(players_dict, xi_players, match_date):
    rows = []
    for player in xi_players:
        if not player or player not in players_dict:
            continue
        df   = players_dict[player]
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
        return pd.Series({k:0 for k in ["goals_per90","assists_per90",
            "shots_per90","sot_per90","tackles_per90","intercept_per90","avg_minutes"]})
    return pd.DataFrame(rows).mean()

# ─────────────────────────────────────────
# 7. BUILD TRAINING DATA
# ─────────────────────────────────────────
def build_training_data(players_dict, club, season):
    s_start, s_end = get_season_range(season)
    club_matches = []
    for name, df in players_dict.items():
        cdf = df[season_mask(df, season) & (df["Team"] == club)]
        cdf = cdf[["Date","TeamGoals","OppGoals","Venue"]].drop_duplicates()
        if len(cdf) > 0:
            club_matches.append(cdf)
    if not club_matches:
        return pd.DataFrame(), pd.Series(dtype=float)

    matches = pd.concat(club_matches).drop_duplicates("Date").sort_values("Date")
    features, targets = [], []

    for _, row in matches.iterrows():
        mdate    = row["Date"]
        squad    = [n for n in players_dict if True]  # all players
        xi_feats = aggregate_xi_features(players_dict, list(players_dict.keys()), mdate)

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
# 8. TRAIN / PREDICT HELPERS
# ─────────────────────────────────────────
def train_models(X, y):
    if len(X) < 5:
        return None, None, 0.5, 0.5
    split = max(1, int(len(X)*0.8))
    Xtr,Xte = X.iloc[:split], X.iloc[split:]
    ytr,yte = y.iloc[:split], y.iloc[split:]
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    lr = LinearRegression()
    rf.fit(Xtr,ytr); lr.fit(Xtr,ytr)
    mae_rf = mean_absolute_error(yte, rf.predict(Xte)) if len(Xte)>0 else 1.0
    mae_lr = mean_absolute_error(yte, lr.predict(Xte)) if len(Xte)>0 else 1.0
    w_rf=1/(mae_rf+1e-5); w_lr=1/(mae_lr+1e-5); t=w_rf+w_lr
    return rf, lr, w_rf/t, w_lr/t

def traditional_xg(xi_feats, avg_scored):
    shots = xi_feats.get("shots_per90",0)*90
    return max(shots*(avg_scored/max(shots,1)), 0.3)

def poisson_prob(lam, k):
    return math.exp(-lam)*(lam**k)/math.factorial(k)

def scoreline_matrix(lam_h, lam_a, max_goals=6):
    m = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            m[i][j] = poisson_prob(lam_h,i)*poisson_prob(lam_a,j)
    return m

# ─────────────────────────────────────────
# 9. MAIN PREDICT
# ─────────────────────────────────────────
def predict_match(players_dict, home_team, away_team,
                  home_xi, away_xi, match_date, season):
    match_date = pd.to_datetime(match_date)
    xg_results = {}

    for team, xi, venue_val in [(home_team,home_xi,1),(away_team,away_xi,0)]:
        xi_feats = aggregate_xi_features(players_dict, xi, match_date)
        team_past = []
        for nm, df in players_dict.items():
            mask = (df["Team"]==team) & (df["Date"]<match_date)
            team_past.append(df[mask][["Date","TeamGoals","OppGoals"]].drop_duplicates())
        if team_past:
            hist = pd.concat(team_past).drop_duplicates("Date").sort_values("Date").tail(5)
            avg_scored   = hist["TeamGoals"].mean() if len(hist)>0 else 1.2
            avg_conceded = hist["OppGoals"].mean()  if len(hist)>0 else 1.0
        else:
            avg_scored, avg_conceded = 1.2, 1.0

        xg_trad  = traditional_xg(xi_feats, avg_scored)
        feat_row = pd.DataFrame([{
            "avg_scored_last5":avg_scored,"avg_conceded_last5":avg_conceded,
            "goals_per90":xi_feats["goals_per90"],"shots_per90":xi_feats["shots_per90"],
            "sot_per90":xi_feats["sot_per90"],"assists_per90":xi_feats["assists_per90"],
            "tackles_per90":xi_feats["tackles_per90"],"venue_home":venue_val,
        }])
        X, y = build_training_data(players_dict, team, season)
        if len(X) >= 5:
            rf,lr,w_rf,w_lr = train_models(X,y)
            xg_ml    = w_rf*max(rf.predict(feat_row)[0],0)+w_lr*max(lr.predict(feat_row)[0],0)
            xg_final = 0.6*xg_ml + 0.4*xg_trad
        else:
            xg_final = xg_trad
        xg_results[team] = max(round(xg_final,3), 0.1)

    lam_h=xg_results[home_team]; lam_a=xg_results[away_team]
    matrix=scoreline_matrix(lam_h,lam_a)
    win=float(np.sum(np.tril(matrix,-1))); draw=float(np.sum(np.diag(matrix)))
    loss=float(np.sum(np.triu(matrix,1)))
    scores=[(i,j,matrix[i][j]) for i in range(7) for j in range(7)]
    top5=sorted(scores,key=lambda x:-x[2])[:5]
    return {
        "xg_home":lam_h,"xg_away":lam_a,
        "home_win":round(win*100,1),"draw":round(draw*100,1),"away_win":round(loss*100,1),
        "top5":[(f"{h}-{a}",round(p*100,2)) for h,a,p in top5],
        "matrix":matrix,"home_team":home_team,"away_team":away_team,
    }
