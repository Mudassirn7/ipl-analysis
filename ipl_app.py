import streamlit as st
import pandas as pd
import numpy as np
import gdown
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

.ipl-title {
    text-align:center;
    font-size:55px;
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
    font-size:20px;
    border:none;
    border-radius:10px;
    padding:12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<p class='ipl-title'>🏏 IPL PREDICTOR</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# DOWNLOAD DATASET
# =========================================================

FILE_ID = "1mr2IIjhMOtRp0ZDlVLw_IFxmAY_ExGUL"

DATA_FILE = "ipl_dataset.csv"

if not os.path.exists(DATA_FILE):

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(url, DATA_FILE, quiet=False)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(DATA_FILE)

# =========================================================
# FIX TEAM NAMES
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
# FILTER CURRENT TEAMS
# =========================================================

df = df[
    (df["batting_team"].isin(IPL_TEAMS)) &
    (df["bowling_team"].isin(IPL_TEAMS))
]

# =========================================================
# VENUES
# =========================================================

if "venue" in df.columns:
    IPL_VENUES = sorted(df["venue"].dropna().unique().tolist())
else:
    IPL_VENUES = ["Unknown Stadium"]
    df["venue"] = "Unknown Stadium"

# =========================================================
# CREATE OVERS COLUMN
# =========================================================

df["overs"] = df["over"] + (df["ball"] / 6)

# =========================================================
# CREATE TEAM SCORE
# =========================================================

df["team_score"] = df.groupby(
    ["match_id", "innings"]
)["total_runs"].cumsum()

# =========================================================
# CREATE WICKETS
# =========================================================

# Detect wicket column automatically

if "is_wicket" in df.columns:

    wicket_col = "is_wicket"

elif "player_dismissed" in df.columns:

    df["is_wicket"] = df["player_dismissed"].notnull().astype(int)

    wicket_col = "is_wicket"

else:

    df["is_wicket"] = 0

    wicket_col = "is_wicket"

df["team_wicket"] = df.groupby(
    ["match_id", "innings"]
)[wicket_col].cumsum()

# =========================================================
# CURRENT RUN RATE
# =========================================================

df["crr"] = (
    df["team_score"] /
    df["overs"].replace(0, 0.1)
)

# =========================================================
# FINAL SCORE
# =========================================================

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
    "team_wicket",
    "overs",
    "crr",
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

targets = innings1.groupby("match_id")["team_score"].max().reset_index()

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
    120 - ((win_df["over"] * 6) + win_df["ball"])
)

win_df["balls_left"] = win_df["balls_left"].replace(0, 1)

win_df["rrr"] = (
    win_df["runs_left"] * 6 /
    win_df["balls_left"]
)

win_df["winner"] = (
    win_df["runs_left"] <= 0
).astype(int)

win_df = win_df[[
    "batting_team",
    "bowling_team",
    "venue",
    "target",
    "team_score",
    "team_wicket",
    "overs",
    "crr",
    "rrr",
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

    # ==========================
    # SCORE PREDICTION
    # ==========================

    Xr = score_df[[
        "batting_team",
        "bowling_team",
        "venue",
        "current_runs",
        "wickets",
        "overs",
        "crr"
    ]]

    yr = score_df["final_score"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr,
        yr,
        test_size=0.2,
        random_state=42
    )

    rf_r = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    gb_r = GradientBoostingRegressor(
        random_state=42
    )

    lr_r = LinearRegression()

    dt_r = DecisionTreeRegressor(
        random_state=42
    )

    reg_models = {
        "Random Forest": rf_r,
        "Gradient Boosting": gb_r,
        "Linear Regression": lr_r,
        "Decision Tree": dt_r
    }

    for model in reg_models.values():
        model.fit(Xr_train, yr_train)

    # ==========================
    # WIN PREDICTION
    # ==========================

    Xc = win_df[[
        "batting_team",
        "bowling_team",
        "venue",
        "target",
        "team_score",
        "team_wicket",
        "overs",
        "crr",
        "rrr"
    ]]

    yc = win_df["winner"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc,
        yc,
        test_size=0.2,
        random_state=42
    )

    rf_c = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    gb_c = GradientBoostingClassifier(
        random_state=42
    )

    lr_c = LogisticRegression(
        max_iter=1000
    )

    dt_c = DecisionTreeClassifier(
        random_state=42
    )

    cls_models = {
        "Random Forest": rf_c,
        "Gradient Boosting": gb_c,
        "Logistic Regression": lr_c,
        "Decision Tree": dt_c
    }

    for model in cls_models.values():
        model.fit(Xc_train, yc_train)

    return (
        reg_models,
        cls_models
    )

# =========================================================
# TRAIN
# =========================================================

with st.spinner("Training models on real IPL data..."):

    REG_MODELS, CLS_MODELS = train_models()

st.success("Models Trained Successfully ✅")

# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs([
    "🎯 Score Predictor",
    "🏆 Win Predictor"
])

# =========================================================
# SCORE PREDICTOR
# =========================================================

with tab1:

    col1, col2 = st.columns(2)

    with col1:

        batting_team = st.selectbox(
            "Batting Team",
            IPL_TEAMS
        )

        bowling_team = st.selectbox(
            "Bowling Team",
            [x for x in IPL_TEAMS if x != batting_team]
        )

        venue = st.selectbox(
            "Venue",
            IPL_VENUES
        )

        model_name = st.selectbox(
            "Model",
            list(REG_MODELS.keys())
        )

    with col2:

        current_runs = st.number_input(
            "Current Runs",
            0,
            300,
            80
        )

        wickets = st.slider(
            "Wickets Fallen",
            0,
            9,
            2
        )

        overs = st.slider(
            "Overs",
            0.1,
            20.0,
            10.0
        )

    if st.button("🎯 Predict Score"):

        crr = current_runs / max(overs, 0.1)

        X = np.array([[
            TEAM_ENC[batting_team],
            TEAM_ENC[bowling_team],
            VENUE_ENC[venue],
            current_runs,
            wickets,
            overs,
            crr
        ]])

        prediction = int(
            REG_MODELS[model_name].predict(X)[0]
        )

        st.markdown(f"""
        <div class='result-box'>
            <p>Predicted Final Score</p>
            <p class='big-score'>{prediction}</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# WIN PREDICTOR
# =========================================================

with tab2:

    col1, col2 = st.columns(2)

    with col1:

        chase_team = st.selectbox(
            "Chasing Team",
            IPL_TEAMS
        )

        defend_team = st.selectbox(
            "Defending Team",
            [x for x in IPL_TEAMS if x != chase_team]
        )

        venue2 = st.selectbox(
            "Venue",
            IPL_VENUES,
            key="v2"
        )

        model_name2 = st.selectbox(
            "Model",
            list(CLS_MODELS.keys())
        )

    with col2:

        target = st.number_input(
            "Target",
            50,
            300,
            180
        )

        score = st.number_input(
            "Current Score",
            0,
            300,
            90
        )

        wickets2 = st.slider(
            "Wickets",
            0,
            9,
            3
        )

        overs2 = st.slider(
            "Overs Completed",
            0.1,
            20.0,
            10.0
        )

    if st.button("🏆 Predict Winner"):

        crr = score / max(overs2, 0.1)

        rrr = (
            (target - score) /
            max(20 - overs2, 0.1)
        )

        X2 = np.array([[
            TEAM_ENC[chase_team],
            TEAM_ENC[defend_team],
            VENUE_ENC[venue2],
            target,
            score,
            wickets2,
            overs2,
            crr,
            rrr
        ]])

        model = CLS_MODELS[model_name2]

        pred = model.predict(X2)[0]

        prob = model.predict_proba(X2)[0]

        winner = chase_team if pred == 1 else defend_team

        confidence = round(max(prob) * 100, 2)

        st.markdown(f"""
        <div class='result-box'>
            <p>Predicted Winner</p>
            <p class='win-team'>{winner}</p>
            <p>Confidence: {confidence}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(prob[1] * 100))

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "<center>IPL Predictor using Real IPL Dataset + Machine Learning</center>",
    unsafe_allow_html=True
)
