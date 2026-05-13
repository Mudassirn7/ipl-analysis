import streamlit as st
import numpy as np
import pandas as pd
import gdown
import os
import warnings

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

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

warnings.filterwarnings('ignore')

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
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@400;600;700&display=swap');

html, body {
    background-color: #ffffff !important;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.block-container {
    background-color: #ffffff !important;
}

p, span, div, li, td, th,
.stMarkdown p,
[data-testid="stText"] {
    color: #111111 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

h1, h2, h3, h4 {
    color: #111111 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

label,
.stSelectbox label,
.stSlider label,
.stNumberInput label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
    color: #222222 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    font-family: 'Rajdhani', sans-serif !important;
}

.stSelectbox > div > div,
input[type="number"] {
    background: #f9f9f9 !important;
    color: #111111 !important;
    border: 1px solid #cccccc !important;
}

button[data-baseweb="tab"] p,
button[data-baseweb="tab"] span {
    color: #333333 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

button[data-baseweb="tab"][aria-selected="true"] p {
    color: #d44000 !important;
}

[data-testid="stMetricValue"] {
    color: #d44000 !important;
    font-family: 'Bebas Neue', sans-serif !important;
}

[data-testid="stMetricLabel"] p {
    color: #444444 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #d44000, #ff8c00) !important;
    color: #ffffff !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.3rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    width: 100% !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #d44000, #ff8c00) !important;
}

[data-testid="stDataFrame"] * {
    color: #111111 !important;
}

.stCaption, small {
    color: #555555 !important;
}

.ipl-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    text-align: center;
    background: linear-gradient(90deg, #d44000, #ff8c00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 4px;
    margin-bottom: 0;
}

.ipl-subtitle {
    text-align: center;
    color: #555555 !important;
    font-size: 0.95rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.tab-info {
    background: #fff3e0;
    border-left: 4px solid #ff6d00;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 20px;
    color: #333333 !important;
    font-size: 1rem;
}

.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    color: #d44000 !important;
    letter-spacing: 2px;
    border-bottom: 2px solid #ffcc99;
    padding-bottom: 4px;
    margin-bottom: 14px;
}

.result-box {
    background: linear-gradient(135deg, #fff3e0, #ffe0b2);
    border: 2px solid #ff6d00;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    margin-top: 16px;
}

.big-score {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    color: #d44000 !important;
    line-height: 1;
}

.winner-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: #1a7a1a !important;
    letter-spacing: 2px;
}

.res-label {
    color: #666666 !important;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.team-sub {
    color: #d44000 !important;
    font-size: 1.1rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# GOOGLE DRIVE DATASET DOWNLOAD
# =========================================================

FILE_ID = "1mr2IIjhMOtRp0ZDlVLw_IFxmAY_ExGUL"
DATA_PATH = "ipl_dataset.csv"

if not os.path.exists(DATA_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, DATA_PATH, quiet=False)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(DATA_PATH)

# =========================================================
# TEAM NAME FIXING
# =========================================================

TEAM_NAME_MAPPING = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Gujarat Lions": "Gujarat Titans",
    "Deccan Chargers": "Sunrisers Hyderabad"
}

df["batting_team"] = df["batting_team"].replace(TEAM_NAME_MAPPING)
df["bowling_team"] = df["bowling_team"].replace(TEAM_NAME_MAPPING)

# =========================================================
# FILTER ONLY CURRENT IPL TEAMS
# =========================================================

CURRENT_TEAMS = [
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

df = df[
    (df["batting_team"].isin(CURRENT_TEAMS)) &
    (df["bowling_team"].isin(CURRENT_TEAMS))
]

# =========================================================
# CREATE VENUE LIST
# =========================================================

IPL_VENUES = sorted(df["venue"].dropna().unique().tolist())

# =========================================================
# ENCODING
# =========================================================

TEAM_ENC = {
    team: i for i, team in enumerate(CURRENT_TEAMS)
}

VENUE_ENC = {
    venue: i for i, venue in enumerate(IPL_VENUES)
}

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["overs_completed"] = df["over"] + (df["ball"] / 6)

df["current_run_rate"] = (
    df["team_score"] /
    df["overs_completed"].replace(0, 0.1)
)

# =========================================================
# SCORE PREDICTION DATASET
# =========================================================

score_df = df.copy()

score_df["final_score"] = score_df.groupby(
    ["match_id", "innings"]
)["team_score"].transform("max")

score_df = score_df[
    [
        "batting_team",
        "bowling_team",
        "venue",
        "team_score",
        "team_wicket",
        "overs_completed",
        "current_run_rate",
        "final_score"
    ]
].dropna()

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

# =========================================================
# WIN PREDICTION DATASET
# =========================================================

win_df = df[df["innings"] == 2].copy()

match_target = (
    df[df["innings"] == 1]
    .groupby("match_id")["team_score"]
    .max()
    .reset_index()
)

match_target.columns = ["match_id", "target"]

win_df = win_df.merge(
    match_target,
    on="match_id",
    how="left"
)

win_df["runs_left"] = (
    win_df["target"] + 1 - win_df["team_score"]
)

win_df["balls_left"] = (
    120 - ((win_df["over"] * 6) + win_df["ball"])
)

win_df["balls_left"] = win_df["balls_left"].clip(lower=1)

win_df["required_run_rate"] = (
    win_df["runs_left"] * 6 /
    win_df["balls_left"]
)

win_df["winner"] = (
    win_df["runs_left"] <= 0
).astype(int)

win_df = win_df[
    [
        "batting_team",
        "bowling_team",
        "venue",
        "target",
        "team_score",
        "team_wicket",
        "overs_completed",
        "current_run_rate",
        "required_run_rate",
        "winner"
    ]
].dropna()

# =========================================================
# ENCODING APPLY
# =========================================================

def encode_team(x):
    return TEAM_ENC.get(x, 0)

def encode_venue(x):
    return VENUE_ENC.get(x, 0)

# SCORE DATA
score_df["batting_team"] = score_df["batting_team"].apply(encode_team)
score_df["bowling_team"] = score_df["bowling_team"].apply(encode_team)
score_df["venue"] = score_df["venue"].apply(encode_venue)

# WIN DATA
win_df["batting_team"] = win_df["batting_team"].apply(encode_team)
win_df["bowling_team"] = win_df["bowling_team"].apply(encode_team)
win_df["venue"] = win_df["venue"].apply(encode_venue)

# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models():

    # ================= SCORE MODELS =================

    Xr = score_df[
        [
            "batting_team",
            "bowling_team",
            "venue",
            "current_runs",
            "wickets",
            "overs",
            "crr"
        ]
    ]

    yr = score_df["final_score"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr,
        yr,
        test_size=0.2,
        random_state=42
    )

    rf_r = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    gb_r = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )

    lr_r = LinearRegression()

    dt_r = DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    )

    reg_models = {
        "rf_r": rf_r,
        "gb_r": gb_r,
        "lr_r": lr_r,
        "dt_r": dt_r
    }

    for model in reg_models.values():
        model.fit(Xr_train, yr_train)

    # ================= WIN MODELS =================

    Xc = win_df[
        [
            "batting_team",
            "bowling_team",
            "venue",
            "target",
            "team_score",
            "team_wicket",
            "overs_completed",
            "current_run_rate",
            "required_run_rate"
        ]
    ]

    yc = win_df["winner"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc,
        yc,
        test_size=0.2,
        random_state=42
    )

    rf_c = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    gb_c = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )

    lr_c = LogisticRegression(max_iter=500)

    dt_c = DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    )

    cls_models = {
        "rf_c": rf_c,
        "gb_c": gb_c,
        "lr_c": lr_c,
        "dt_c": dt_c
    }

    for model in cls_models.values():
        model.fit(Xc_train, yc_train)

    # =================================================
    # METRICS
    # =================================================

    reg_metrics = {}

    for name, model in reg_models.items():

        train_pred = model.predict(Xr_train)
        test_pred = model.predict(Xr_test)

        reg_metrics[name] = {
            "train_r2": round(r2_score(yr_train, train_pred), 4),
            "test_r2": round(r2_score(yr_test, test_pred), 4),
            "train_rmse": round(np.sqrt(mean_squared_error(yr_train, train_pred)), 2),
            "test_rmse": round(np.sqrt(mean_squared_error(yr_test, test_pred)), 2),
            "test_mae": round(mean_absolute_error(yr_test, test_pred), 2)
        }

    cls_metrics = {}

    for name, model in cls_models.items():

        train_pred = model.predict(Xc_train)
        test_pred = model.predict(Xc_test)

        cls_metrics[name] = {
            "train_acc": round(accuracy_score(yc_train, train_pred) * 100, 2),
            "test_acc": round(accuracy_score(yc_test, test_pred) * 100, 2),
            "f1": round(f1_score(yc_test, test_pred) * 100, 2),
            "precision": round(precision_score(yc_test, test_pred) * 100, 2),
            "recall": round(recall_score(yc_test, test_pred) * 100, 2)
        }

    return {
        "rf_r": rf_r,
        "gb_r": gb_r,
        "lr_r": lr_r,
        "dt_r": dt_r,

        "rf_c": rf_c,
        "gb_c": gb_c,
        "lr_c": lr_c,
        "dt_c": dt_c,

        "reg_metrics": reg_metrics,
        "cls_metrics": cls_metrics
    }

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<p class="ipl-title">🏏 IPL PREDICTOR</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="ipl-subtitle">REAL IPL DATASET + MACHINE LEARNING</p>',
    unsafe_allow_html=True
)

st.markdown("---")

with st.spinner("⚙️ Training ML Models on Real IPL Dataset..."):
    M = train_models()

st.success("✅ Models trained successfully on real IPL dataset!")

# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs(
    [
        "🎯 Score Predictor",
        "🏆 Win Predictor"
    ]
)

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.markdown(
        '<div class="tab-info">Predict final innings score using real IPL data.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        bat_t = st.selectbox(
            "Batting Team",
            CURRENT_TEAMS
        )

        bowl_t = st.selectbox(
            "Bowling Team",
            [t for t in CURRENT_TEAMS if t != bat_t]
        )

        venue = st.selectbox(
            "Venue",
            IPL_VENUES
        )

        model_name = st.selectbox(
            "Model",
            [
                "Random Forest",
                "Gradient Boosting",
                "Linear Regression",
                "Decision Tree"
            ]
        )

    with c2:

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
            "Overs Completed",
            0.1,
            20.0,
            10.0
        )

    if st.button("🎯 Predict Final Score"):

        crr = current_runs / max(overs, 0.1)

        X = np.array([[
            TEAM_ENC[bat_t],
            TEAM_ENC[bowl_t],
            VENUE_ENC[venue],
            current_runs,
            wickets,
            overs,
            crr
        ]])

        model_map = {
            "Random Forest": M["rf_r"],
            "Gradient Boosting": M["gb_r"],
            "Linear Regression": M["lr_r"],
            "Decision Tree": M["dt_r"]
        }

        pred = int(model_map[model_name].predict(X)[0])

        st.markdown(f"""
        <div class="result-box">
            <p class="res-label">Predicted Final Score</p>
            <p class="big-score">{pred}</p>
            <p class="team-sub">{bat_t}</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.markdown(
        '<div class="tab-info">Predict winner during run chase.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        chase_team = st.selectbox(
            "Chasing Team",
            CURRENT_TEAMS
        )

        defend_team = st.selectbox(
            "Defending Team",
            [t for t in CURRENT_TEAMS if t != chase_team]
        )

        venue2 = st.selectbox(
            "Venue",
            IPL_VENUES,
            key="venue2"
        )

        model2 = st.selectbox(
            "Model",
            [
                "Random Forest",
                "Gradient Boosting",
                "Logistic Regression",
                "Decision Tree"
            ]
        )

    with c2:

        target = st.number_input(
            "Target",
            50,
            300,
            180
        )

        current_score = st.number_input(
            "Current Score",
            0,
            300,
            90
        )

        wickets2 = st.slider(
            "Wickets Fallen",
            0,
            9,
            3
        )

        overs2 = st.slider(
            "Overs Completed",
            0.1,
            20.0,
            10.0,
            key="ov2"
        )

    if st.button("🏆 Predict Winner"):

        crr2 = current_score / max(overs2, 0.1)

        rrr = (
            (target - current_score) /
            max(20 - overs2, 0.1)
        )

        X2 = np.array([[
            TEAM_ENC[chase_team],
            TEAM_ENC[defend_team],
            VENUE_ENC[venue2],
            target,
            current_score,
            wickets2,
            overs2,
            crr2,
            rrr
        ]])

        model_map2 = {
            "Random Forest": M["rf_c"],
            "Gradient Boosting": M["gb_c"],
            "Logistic Regression": M["lr_c"],
            "Decision Tree": M["dt_c"]
        }

        model_used = model_map2[model2]

        pred = model_used.predict(X2)[0]
        prob = model_used.predict_proba(X2)[0]

        winner = chase_team if pred == 1 else defend_team

        confidence = round(max(prob) * 100, 2)

        st.markdown(f"""
        <div class="result-box">
            <p class="res-label">Predicted Winner</p>
            <p class="winner-name">{winner}</p>
            <p style="font-size:1rem;">
                Confidence: <b>{confidence}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(prob[1] * 100))

        st.caption(
            f"{chase_team}: {round(prob[1]*100,1)}% | "
            f"{defend_team}: {round(prob[0]*100,1)}%"
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    '''
    <p style="text-align:center;color:#888;font-size:0.85rem;">
    IPL Predictor using REAL IPL DATASET + Streamlit + Machine Learning
    </p>
    ''',
    unsafe_allow_html=True
)
