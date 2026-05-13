# =========================================================
# IPL PREDICTOR — STREAMLIT FINAL VERSION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import warnings

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IPL Predictor",
    page_icon="🏏",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

html, body, .stApp {
    background-color: white;
}

h1,h2,h3,h4,h5,h6,p,div,span,label {
    color: black !important;
}

.ipl-title{
    text-align:center;
    font-size:60px;
    font-weight:bold;
    color:#ff6b00 !important;
}

.result-box{
    padding:25px;
    border-radius:15px;
    background:#fff3e6;
    border:2px solid #ff6b00;
    text-align:center;
    margin-top:20px;
}

.big-score{
    font-size:70px;
    font-weight:bold;
    color:#ff6b00;
}

.win-team{
    font-size:45px;
    font-weight:bold;
    color:green;
}

.stButton button{
    width:100%;
    background:#ff6b00;
    color:white;
    border:none;
    border-radius:10px;
    padding:12px;
    font-size:20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<p class='ipl-title'>🏏 IPL Predictor</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# EXTRACT DATASET
# =========================================================

ZIP_FILE = "ipl_dataset.zip"
CSV_FILE = "ipl_dataset.csv"

if not os.path.exists(CSV_FILE):

    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall()

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(CSV_FILE)

# =========================================================
# TEAM FIXES
# =========================================================

TEAM_NAME_MAPPING = {

    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Gujarat Lions": "Gujarat Titans"

}

df["batting_team"] = df["batting_team"].replace(TEAM_NAME_MAPPING)
df["bowling_team"] = df["bowling_team"].replace(TEAM_NAME_MAPPING)

# =========================================================
# VENUE FIXES
# =========================================================

VENUE_MAPPING = {

    "Arun Jaitley Stadium, Delhi":
        "Arun Jaitley Stadium",

    "Arun Jaitley Stadium":
        "Arun Jaitley Stadium",

    "M Chinnaswamy Stadium":
        "M. Chinnaswamy Stadium",

    "M.Chinnaswamy Stadium":
        "M. Chinnaswamy Stadium",

    "MA Chidambaram Stadium":
        "M. A. Chidambaram Stadium",

    "MA Chidambaram Stadium, Chepauk":
        "M. A. Chidambaram Stadium",

    "Punjab Cricket Association Stadium":
        "PCA Stadium",

    "Punjab Cricket Association IS Bindra Stadium":
        "PCA Stadium",

    "Rajiv Gandhi International Stadium":
        "Rajiv Gandhi Intl. Cricket Stadium"

}

df["venue"] = (
    df["venue"]
    .replace(VENUE_MAPPING)
    .str.strip()
)

# =========================================================
# CURRENT IPL TEAMS
# =========================================================

IPL_TEAMS = [

    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Sunrisers Hyderabad",
    "Rajasthan Royals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans"

]

# =========================================================
# FILTER DATA
# =========================================================

df = df[
    (df["batting_team"].isin(IPL_TEAMS)) &
    (df["bowling_team"].isin(IPL_TEAMS))
]

# =========================================================
# VENUES
# =========================================================

IPL_VENUES = sorted(df["venue"].dropna().unique())

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["overs_completed"] = (
    df["over"] + (df["ball"] / 6)
)

df["overs_completed"] = df["overs_completed"].clip(0.1, 20)

df["team_score"] = df["team_runs"]

df["wickets"] = df["team_wicket"]

df["current_run_rate"] = (
    df["team_score"] / df["overs_completed"]
)

df["final_score"] = df.groupby(
    ["match_id", "innings"]
)["team_score"].transform("max")

# =========================================================
# ENCODING
# =========================================================

TEAM_ENC = {
    team: idx for idx, team in enumerate(IPL_TEAMS)
}

VENUE_ENC = {
    venue: idx for idx, venue in enumerate(IPL_VENUES)
}

# =========================================================
# SCORE DATASET
# =========================================================

score_df = df[[
    "batting_team",
    "bowling_team",
    "venue",
    "team_score",
    "wickets",
    "overs_completed",
    "current_run_rate",
    "final_score"
]].dropna()

score_df.columns = [
    "batting_team",
    "bowling_team",
    "venue",
    "current_runs",
    "wickets",
    "overs",
    "crr",
    "final_score"
]

score_df["batting_team"] = score_df["batting_team"].map(TEAM_ENC)
score_df["bowling_team"] = score_df["bowling_team"].map(TEAM_ENC)
score_df["venue"] = score_df["venue"].map(VENUE_ENC)

# =========================================================
# WIN DATASET
# =========================================================

innings1 = df[df["innings"] == 1]

targets = innings1.groupby(
    "match_id"
)["team_score"].max().reset_index()

targets.columns = ["match_id", "target"]

win_df = df[df["innings"] == 2].copy()

win_df = win_df.merge(
    targets,
    on="match_id"
)

win_df["runs_left"] = (
    win_df["target"] + 1 - win_df["team_score"]
)

win_df["balls_left"] = (
    120 - (
        (win_df["over"] * 6) + win_df["ball"]
    )
)

win_df["balls_left"] = win_df["balls_left"].clip(lower=1)

win_df["required_run_rate"] = (
    win_df["runs_left"] * 6 / win_df["balls_left"]
)

win_df["winner"] = np.where(
    win_df["runs_left"] <= 0,
    1,
    0
)

win_df = win_df[[
    "batting_team",
    "bowling_team",
    "venue",
    "target",
    "team_score",
    "wickets",
    "overs_completed",
    "current_run_rate",
    "required_run_rate",
    "winner"
]].dropna()

win_df["batting_team"] = win_df["batting_team"].map(TEAM_ENC)
win_df["bowling_team"] = win_df["bowling_team"].map(TEAM_ENC)
win_df["venue"] = win_df["venue"].map(VENUE_ENC)

# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models():

    Xr = score_df.drop("final_score", axis=1)
    yr = score_df["final_score"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr,
        test_size=0.2,
        random_state=42
    )

    rf_r = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=15,
        random_state=42
    )

    gb_r = GradientBoostingRegressor(
        n_estimators=60,
        max_depth=3,
        random_state=42
    )

    lr_r = LinearRegression()

    dt_r = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=20,
        random_state=42
    )

    REG_MODELS = {
        "Random Forest": rf_r,
        "Gradient Boosting": gb_r,
        "Linear Regression": lr_r,
        "Decision Tree": dt_r
    }

    for model in REG_MODELS.values():
        model.fit(Xr_train, yr_train)

    return REG_MODELS

REG_MODELS = train_models()

st.success("Models Trained Successfully ✅")
