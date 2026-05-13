import streamlit as st
import pandas as pd
import numpy as np
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
html, body, .stApp { background-color: white; }
h1,h2,h3,h4,h5,h6,p,div,span,label { color: black !important; }
.ipl-title { text-align:center; font-size:60px; font-weight:bold; color:#ff6b00 !important; }
.result-box { padding:25px; border-radius:15px; background:#fff3e6; border:2px solid #ff6b00; text-align:center; margin-top:20px; }
.big-score { font-size:70px; font-weight:bold; color:#ff6b00; }
.win-team { font-size:45px; font-weight:bold; color:green; }
.stButton button { width:100%; background:#ff6b00; color:white; border:none; border-radius:10px; padding:12px; font-size:20px; }
.analysis-card { background:#f8f9fa; border-left:4px solid #ff6b00; padding:15px; margin:10px 0; border-radius:5px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("<p class='ipl-title'>🏏 IPL Predictor</p>", unsafe_allow_html=True)
st.markdown("---")

# =========================================================
# LOAD DATASET
# =========================================================

CSV_FILE = "IPL.csv"

if not os.path.exists(CSV_FILE):
    try:
        os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
        os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]

        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "chaitu20/ipl-dataset2008-2025",
            path=".", unzip=True
        )
    except Exception as e:
        st.error(f"Dataset download failed: {e}")
        st.stop()

if not os.path.exists(CSV_FILE):
    st.error("IPL.csv not found after download.")
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================

raw_df = pd.read_csv(CSV_FILE)

# =========================================================
# TEAM FIXES
# =========================================================

TEAM_NAME_MAPPING = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Gujarat Lions": "Gujarat Titans"
}

raw_df["batting_team"] = raw_df["batting_team"].replace(TEAM_NAME_MAPPING)
raw_df["bowling_team"] = raw_df["bowling_team"].replace(TEAM_NAME_MAPPING)

# =========================================================
# VENUE FIXES
# =========================================================

VENUE_MAPPING = {
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    "M Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "MA Chidambaram Stadium": "M. A. Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk": "M. A. Chidambaram Stadium",
    "Punjab Cricket Association Stadium": "PCA Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "PCA Stadium",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi Intl. Cricket Stadium",
    "Rajiv Gandhi Intl. Cricket Stadium": "Rajiv Gandhi Intl. Cricket Stadium"
}

raw_df["venue"] = raw_df["venue"].replace(VENUE_MAPPING)

# =========================================================
# CURRENT IPL TEAMS
# =========================================================

IPL_TEAMS = [
    "Chennai Super Kings", "Mumbai Indians",
    "Royal Challengers Bangalore", "Kolkata Knight Riders",
    "Delhi Capitals", "Sunrisers Hyderabad",
    "Rajasthan Royals", "Punjab Kings",
    "Lucknow Super Giants", "Gujarat Titans"
]

df = raw_df[
    (raw_df["batting_team"].isin(IPL_TEAMS)) &
    (raw_df["bowling_team"].isin(IPL_TEAMS))
].copy()

# =========================================================
# VENUES
# =========================================================

IPL_VENUES = sorted(df["venue"].dropna().unique().tolist())

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["overs_completed"] = df["over"] + (df["ball"] / 6)
df["current_run_rate"] = df["team_runs"] / df["overs_completed"].replace(0, 0.1)
df["final_score"] = df.groupby(["match_id", "innings"])["team_runs"].transform("max")

# =========================================================
# ENCODING
# =========================================================

TEAM_ENC = {team: idx for idx, team in enumerate(IPL_TEAMS)}
VENUE_ENC = {venue: idx for idx, venue in enumerate(IPL_VENUES)}

# =========================================================
# SCORE DATASET — only use data from overs 6-16
# (mid-game state, avoids end-of-innings leakage)
# =========================================================

score_df = df[
    (df["overs_completed"] >= 6) &
    (df["overs_completed"] <= 16)
][[
    "batting_team", "bowling_team", "venue",
    "team_runs", "team_wicket", "overs_completed",
    "current_run_rate", "final_score"
]].dropna().copy()

score_df.columns = [
    "batting_team", "bowling_team", "venue",
    "current_runs", "wickets", "overs", "crr", "final_score"
]

score_df["batting_team"] = score_df["batting_team"].map(TEAM_ENC)
score_df["bowling_team"] = score_df["bowling_team"].map(TEAM_ENC)
score_df["venue"] = score_df["venue"].map(VENUE_ENC)
score_df = score_df.dropna()

# =========================================================
# WIN DATASET — Fix: remove leaky features
# winner is determined at MATCH END, not mid-game
# Use only mid-game features (overs 6–18), no runs_left
# =========================================================

innings1 = df[df["innings"] == 1]
targets = innings1.groupby("match_id")["team_runs"].max().reset_index()
targets.columns = ["match_id", "target"]

# Get last ball of each match to know actual winner
match_results = df[df["innings"] == 2].copy()
match_results = match_results.merge(targets, on="match_id")

# Winner = 1 if chasing team score >= target at match end
match_end = match_results.sort_values(
    ["match_id", "over", "ball"]
).groupby("match_id").last().reset_index()

match_end["won_chase"] = (match_end["team_runs"] >= match_end["target"]).astype(int)

winner_lookup = match_end[["match_id", "won_chase"]]

# Build win_df using MID-GAME snapshots (overs 6-18)
win_df = df[
    (df["innings"] == 2) &
    (df["overs_completed"] >= 6) &
    (df["overs_completed"] <= 18)
].copy()

win_df = win_df.merge(targets, on="match_id")
win_df = win_df.merge(winner_lookup, on="match_id")

win_df["runs_needed"] = win_df["target"] - win_df["team_runs"]
win_df["balls_left"] = 120 - ((win_df["over"] * 6) + win_df["ball"])
win_df["balls_left"] = win_df["balls_left"].replace(0, 1)
win_df["required_run_rate"] = win_df["runs_needed"] * 6 / win_df["balls_left"]

# Features: NO runs_left (leaky), use %target_achieved instead
win_df["pct_target_done"] = win_df["team_runs"] / win_df["target"].replace(0, 1)
win_df["pct_overs_done"] = win_df["overs_completed"] / 20

win_df = win_df[[
    "batting_team", "bowling_team", "venue",
    "target", "team_wicket", "overs_completed",
    "current_run_rate", "required_run_rate",
    "pct_target_done", "pct_overs_done",
    "won_chase"
]].dropna().copy()

win_df["batting_team"] = win_df["batting_team"].map(TEAM_ENC)
win_df["bowling_team"] = win_df["bowling_team"].map(TEAM_ENC)
win_df["venue"] = win_df["venue"].map(VENUE_ENC)
win_df = win_df.dropna()

# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models():

    # --- REGRESSION ---
    Xr = score_df[[
        "batting_team", "bowling_team", "venue",
        "current_runs", "wickets", "overs", "crr"
    ]]
    yr = score_df["final_score"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )

    REG_MODELS = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=8,
            min_samples_leaf=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=42
        ),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=6, min_samples_leaf=15, random_state=42
        )
    }

    for model in REG_MODELS.values():
        model.fit(Xr_train, yr_train)

    # --- CLASSIFICATION ---
    Xc = win_df[[
        "batting_team", "bowling_team", "venue",
        "target", "team_wicket", "overs_completed",
        "current_run_rate", "required_run_rate",
        "pct_target_done", "pct_overs_done"
    ]]
    yc = win_df["won_chase"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=42
    )

    CLS_MODELS = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6,
            min_samples_leaf=15, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=42
        )
    }

    for model in CLS_MODELS.values():
        model.fit(Xc_train, yc_train)

    # --- METRICS ---
    reg_metrics = {}
    for name, model in REG_MODELS.items():
        train_pred = model.predict(Xr_train)
        test_pred  = model.predict(Xr_test)
        train_r2   = r2_score(yr_train, train_pred)
        test_r2    = r2_score(yr_test, test_pred)
        reg_metrics[name] = {
            "Train R²": round(train_r2, 4),
            "Test R²":  round(test_r2, 4),
            "RMSE":     round(np.sqrt(mean_squared_error(yr_test, test_pred)), 2),
            "MAE":      round(mean_absolute_error(yr_test, test_pred), 2),
            "Overfit":  "Yes" if abs(train_r2 - test_r2) > 0.10 else "No"
        }

    cls_metrics = {}
    for name, model in CLS_MODELS.items():
        train_pred = model.predict(Xc_train)
        test_pred  = model.predict(Xc_test)
        train_acc  = accuracy_score(yc_train, train_pred) * 100
        test_acc   = accuracy_score(yc_test,  test_pred)  * 100
        cls_metrics[name] = {
            "Train Acc %":  round(train_acc, 2),
            "Test Acc %":   round(test_acc, 2),
            "Precision %":  round(precision_score(yc_test, test_pred, zero_division=0) * 100, 2),
            "Recall %":     round(recall_score(yc_test, test_pred, zero_division=0) * 100, 2),
            "F1 Score %":   round(f1_score(yc_test, test_pred, zero_division=0) * 100, 2),
            "Overfit":      "Yes" if abs(train_acc - test_acc) > 8 else "No"
        }

    return (
        REG_MODELS, CLS_MODELS,
        reg_metrics, cls_metrics,
        Xr_train, Xr_test, yr_train, yr_test,
        Xc_train, Xc_test, yc_train, yc_test
    )

# =========================================================
# TRAIN
# =========================================================

with st.spinner("Training Models..."):
    (
        REG_MODELS, CLS_MODELS,
        REG_METRICS, CLS_METRICS,
        Xr_train, Xr_test, yr_train, yr_test,
        Xc_train, Xc_test, yc_train, yc_test
    ) = train_models()

st.success("Models Trained Successfully ✅")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Score Predictor",
    "🏆 Win Predictor",
    "📊 Model Report",
    "🔍 Data Analysis"
])

# =========================================================
# TAB 1 — SCORE PREDICTOR
# =========================================================

with tab1:
    st.subheader("Predict Final Score")

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox("Batting Team", IPL_TEAMS, key="bat1")
        bowling_team = st.selectbox(
            "Bowling Team",
            [x for x in IPL_TEAMS if x != batting_team],
            key="bowl1"
        )
        venue = st.selectbox("Venue", IPL_VENUES, key="venue1")
        model_name = st.selectbox("ML Model", list(REG_MODELS.keys()), key="model1")

    with col2:
        current_runs = st.number_input("Current Runs", 0, 300, 80)
        wickets      = st.slider("Wickets", 0, 9, 2, key="wk1")
        over_num     = st.slider("Overs", 0, 19, 10, key="ov1")
        ball_num     = st.slider("Balls", 0, 5, 0, key="ball1")
        overs        = round(over_num + (ball_num / 6), 2)
        st.caption(f"{over_num}.{ball_num} overs")

    if st.button("Predict Final Score", key="btn1"):
        crr = current_runs / max(overs, 0.1)
        X   = np.array([[
            TEAM_ENC[batting_team], TEAM_ENC[bowling_team],
            VENUE_ENC[venue], current_runs, wickets, overs, crr
        ]])
        prediction = int(REG_MODELS[model_name].predict(X)[0])
        st.markdown(f"""
        <div class="result-box">
        <p>Predicted Final Score</p>
        <p class="big-score">{prediction}</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2 — WIN PREDICTOR
# =========================================================

with tab2:
    st.subheader("Predict Match Winner")

    col1, col2 = st.columns(2)

    with col1:
        chasing_team   = st.selectbox("Chasing Team", IPL_TEAMS, key="ct")
        defending_team = st.selectbox(
            "Defending Team",
            [x for x in IPL_TEAMS if x != chasing_team],
            key="dt"
        )
        venue2      = st.selectbox("Venue", IPL_VENUES, key="venue2")
        model_name2 = st.selectbox("ML Model", list(CLS_MODELS.keys()), key="model2")

    with col2:
        target        = st.number_input("Target", 50, 300, 180)
        current_score = st.number_input("Current Score", 0, 300, 90)
        wickets2      = st.slider("Wickets Fallen", 0, 9, 3, key="wk2")
        over_num2     = st.slider("Overs", 0, 19, 10, key="ov2")
        ball_num2     = st.slider("Balls", 0, 5, 0, key="ball2")
        overs2        = round(over_num2 + (ball_num2 / 6), 2)
        st.caption(f"{over_num2}.{ball_num2} overs")

    if st.button("Predict Winner", key="btn2"):
        crr          = current_score / max(overs2, 0.1)
        rrr          = (target - current_score) * 6 / max((120 - overs2 * 6), 1)
        pct_done     = current_score / max(target, 1)
        pct_overs    = overs2 / 20

        X2 = np.array([[
            TEAM_ENC[chasing_team], TEAM_ENC[defending_team],
            VENUE_ENC[venue2], target, wickets2,
            overs2, crr, rrr, pct_done, pct_overs
        ]])

        model  = CLS_MODELS[model_name2]
        pred   = model.predict(X2)[0]
        prob   = model.predict_proba(X2)[0]
        winner = chasing_team if pred == 1 else defending_team
        conf   = round(max(prob) * 100, 2)

        st.markdown(f"""
        <div class="result-box">
        <p>Predicted Winner</p>
        <p class="win-team">{winner}</p>
        <p>Confidence: {conf}%</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 3 — MODEL REPORT
# =========================================================

with tab3:
    st.subheader("📈 Regression Model Report")
    st.caption("Predicting final innings score from mid-game state (overs 6–16)")
    reg_table = pd.DataFrame(REG_METRICS).T
    st.dataframe(reg_table, use_container_width=True)

    st.markdown("---")

    st.subheader("🏆 Classification Model Report")
    st.caption("Predicting match winner from 2nd innings mid-game state (overs 6–18)")
    cls_table = pd.DataFrame(CLS_METRICS).T
    st.dataframe(cls_table, use_container_width=True)

# =========================================================
# TAB 4 — DATA ANALYSIS
# =========================================================

with tab4:

    st.subheader("🔍 Data Analysis & Pipeline")

    # ---- STEP 1: DATA LOADING ----
    st.markdown("### Step 1: Data Loading")
    st.markdown(f"""
    <div class="analysis-card">
    <b>Source:</b> Kaggle — IPL Ball-by-Ball Dataset (2008–2025)<br>
    <b>File:</b> IPL.csv<br>
    <b>Total Rows Loaded:</b> {len(raw_df):,}<br>
    <b>Total Columns:</b> {raw_df.shape[1]}<br>
    <b>Key Columns:</b> match_id, batting_team, bowling_team, venue, over, ball, batsman_runs, team_runs, team_wicket, innings
    </div>
    """, unsafe_allow_html=True)

    # ---- STEP 2: DATA CLEANING ----
    st.markdown("### Step 2: Data Cleaning")

    null_counts = raw_df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]

    st.markdown(f"""
    <div class="analysis-card">
    <b>Duplicate Rows:</b> {raw_df.duplicated().sum()}<br>
    <b>Columns with Null Values:</b> {len(null_cols)}<br>
    <b>Team Name Fixes Applied:</b> {len(TEAM_NAME_MAPPING)} mappings
    (e.g., "Delhi Daredevils" → "Delhi Capitals", "Kings XI Punjab" → "Punjab Kings")<br>
    <b>Venue Name Fixes Applied:</b> {len(VENUE_MAPPING)} mappings
    (e.g., "MA Chidambaram Stadium, Chepauk" → "M. A. Chidambaram Stadium")<br>
    <b>Teams Filtered:</b> Only current 10 IPL teams kept
    </div>
    """, unsafe_allow_html=True)

    if len(null_cols) > 0:
        st.caption("Null value counts per column:")
        st.dataframe(null_cols.reset_index().rename(
            columns={"index": "Column", 0: "Null Count"}
        ), use_container_width=True)
    else:
        st.caption("✅ No null values found in key columns.")

    # ---- STEP 3: FEATURE ENGINEERING ----
    st.markdown("### Step 3: Feature Engineering")
    st.markdown("""
    <div class="analysis-card">
    <b>New Features Created:</b><br>
    • <b>overs_completed</b> = over + (ball / 6) — exact decimal overs<br>
    • <b>current_run_rate (CRR)</b> = team_runs / overs_completed<br>
    • <b>final_score</b> = max team_runs per match-innings (target for regression)<br>
    • <b>required_run_rate (RRR)</b> = runs_needed × 6 / balls_left (2nd innings)<br>
    • <b>pct_target_done</b> = current_score / target (removes absolute scale bias)<br>
    • <b>pct_overs_done</b> = overs_completed / 20
    </div>
    """, unsafe_allow_html=True)

    # ---- STEP 4: LABEL CREATION ----
    st.markdown("### Step 4: Label / Target Creation")
    st.markdown("""
    <div class="analysis-card">
    <b>Regression Target:</b> final_score — actual total runs scored in that innings<br>
    <b>Classification Target:</b> won_chase (0 or 1)<br>
    &nbsp;&nbsp;&nbsp;→ Determined from the LAST ball of each match (no mid-game leakage)<br>
    &nbsp;&nbsp;&nbsp;→ 1 = chasing team's final score ≥ target &nbsp;|&nbsp; 0 = failed to chase<br>
    <b>Why this matters:</b> Earlier version used runs_left ≤ 0 as label which caused
    100% accuracy (data leakage). Fixed by using actual match-end result.
    </div>
    """, unsafe_allow_html=True)

    # ---- STEP 5: DATASET STATS ----
    st.markdown("### Step 5: Dataset Summary After Processing")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches",   str(df["match_id"].nunique()))
    col2.metric("Regression Rows", str(len(score_df)))
    col3.metric("Classification Rows", str(len(win_df)))
    col4.metric("Unique Venues",   str(len(IPL_VENUES)))

    st.markdown("---")

    # ---- STEP 6: ENCODING ----
    st.markdown("### Step 6: Label Encoding")
    st.markdown("""
    <div class="analysis-card">
    <b>Method:</b> Integer Label Encoding (manual dictionary)<br>
    <b>batting_team / bowling_team:</b> Each team mapped to 0–9<br>
    <b>venue:</b> Each venue mapped to integer index<br>
    <b>Reason:</b> Tree-based and linear models require numeric input
    </div>
    """, unsafe_allow_html=True)

    # ---- STEP 7: TRAIN-TEST SPLIT ----
    st.markdown("### Step 7: Train-Test Split")
    st.markdown("""
    <div class="analysis-card">
    <b>Split Ratio:</b> 80% Train / 20% Test<br>
    <b>random_state:</b> 42 (reproducibility)<br>
    <b>Method:</b> sklearn train_test_split (random shuffle)
    </div>
    """, unsafe_allow_html=True)

    # ---- STEP 8: MODELS USED ----
    st.markdown("### Step 8: Models Used")
    model_info = {
        "Model": [
            "Random Forest", "Gradient Boosting",
            "Linear Regression / Logistic Regression", "Decision Tree"
        ],
        "Type": [
            "Ensemble (Bagging)", "Ensemble (Boosting)",
            "Linear", "Single Tree"
        ],
        "Used For": [
            "Both Regression & Classification",
            "Both Regression & Classification",
            "Regression + Classification",
            "Both Regression & Classification"
        ]
    }
    st.dataframe(pd.DataFrame(model_info), use_container_width=True)

    # ---- STEP 9: EVALUATION METRICS ----
    st.markdown("### Step 9: Evaluation Metrics")
    st.markdown("""
    <div class="analysis-card">
    <b>Regression:</b> R² (Train & Test), RMSE, MAE, Overfit Check (Train R² - Test R² > 0.10)<br>
    <b>Classification:</b> Accuracy (Train & Test), Precision, Recall, F1 Score, Overfit Check (diff > 8%)
    </div>
    """, unsafe_allow_html=True)

    # ---- TEAM STATS ----
    st.markdown("---")
    st.markdown("### 📊 Team Match Count (Filtered Dataset)")
    team_counts = pd.concat([
        df["batting_team"], df["bowling_team"]
    ]).value_counts().reset_index()
    team_counts.columns = ["Team", "Ball-by-Ball Rows"]
    st.dataframe(team_counts, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    "<center>IPL Predictor — Ball-by-Ball Dataset | ML Pipeline with 4 Models</center>",
    unsafe_allow_html=True
)
