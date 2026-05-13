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
# CUSTOM CSS
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
# DOWNLOAD DATASET FROM GOOGLE DRIVE
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
# TEAM NAME FIXING
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
# FILTER ONLY CURRENT TEAMS
# =========================================================

df = df[
    (df["batting_team"].isin(IPL_TEAMS)) &
    (df["bowling_team"].isin(IPL_TEAMS))
]

# =========================================================
# VENUES
# =========================================================

IPL_VENUES = sorted(
    df["venue"].dropna().unique().tolist()
)

# =========================================================
# OVERS COMPLETED
# =========================================================

df["overs_completed"] = (
    df["over"] + (df["ball"] / 6)
)

# =========================================================
# TEAM SCORE + WICKETS
# =========================================================

df["team_score"] = df["team_runs"]

df["wickets"] = df["team_wicket"]

# =========================================================
# CURRENT RUN RATE
# =========================================================

df["current_run_rate"] = (
    df["team_score"] /
    df["overs_completed"].replace(0, 0.1)
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
    win_df["target"] + 1 -
    win_df["team_score"]
)

win_df["balls_left"] = (
    120 - (
        (win_df["over"] * 6) +
        win_df["ball"]
    )
)

win_df["balls_left"] = (
    win_df["balls_left"].replace(0, 1)
)

win_df["required_run_rate"] = (
    win_df["runs_left"] * 6 /
    win_df["balls_left"]
)

# Winner label

win_df["winner"] = (
    win_df["runs_left"] <= 0
).astype(int)

# Keep required columns

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

# Encoding

win_df["batting_team"] = win_df["batting_team"].map(TEAM_ENC)

win_df["bowling_team"] = win_df["bowling_team"].map(TEAM_ENC)

win_df["venue"] = win_df["venue"].map(VENUE_ENC)

# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models():

    # =====================================================
    # SCORE PREDICTION
    # =====================================================

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

    REG_MODELS = {

        "Random Forest": rf_r,
        "Gradient Boosting": gb_r,
        "Linear Regression": lr_r,
        "Decision Tree": dt_r

    }

    for model in REG_MODELS.values():
        model.fit(Xr_train, yr_train)

    # =====================================================
    # WIN PREDICTION
    # =====================================================

    Xc = win_df[[
        "batting_team",
        "bowling_team",
        "venue",
        "target",
        "team_score",
        "wickets",
        "overs_completed",
        "current_run_rate",
        "required_run_rate"
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

    CLS_MODELS = {

        "Random Forest": rf_c,
        "Gradient Boosting": gb_c,
        "Logistic Regression": lr_c,
        "Decision Tree": dt_c

    }

    for model in CLS_MODELS.values():
        model.fit(Xc_train, yc_train)

    # =====================================================
    # METRICS
    # =====================================================

    reg_metrics = {}

    for name, model in REG_MODELS.items():

        pred_train = model.predict(Xr_train)

        pred_test = model.predict(Xr_test)

        reg_metrics[name] = {

            "Train R2": round(
                r2_score(yr_train, pred_train), 4
            ),

            "Test R2": round(
                r2_score(yr_test, pred_test), 4
            ),

            "RMSE": round(
                np.sqrt(
                    mean_squared_error(yr_test, pred_test)
                ), 2
            ),

            "MAE": round(
                mean_absolute_error(yr_test, pred_test), 2
            )

        }

    cls_metrics = {}

    for name, model in CLS_MODELS.items():

        pred_train = model.predict(Xc_train)

        pred_test = model.predict(Xc_test)

        cls_metrics[name] = {

            "Train Accuracy": round(
                accuracy_score(yc_train, pred_train) * 100, 2
            ),

            "Test Accuracy": round(
                accuracy_score(yc_test, pred_test) * 100, 2
            ),

            "Precision": round(
                precision_score(yc_test, pred_test) * 100, 2
            ),

            "Recall": round(
                recall_score(yc_test, pred_test) * 100, 2
            ),

            "F1 Score": round(
                f1_score(yc_test, pred_test) * 100, 2
            )

        }

    return (
        REG_MODELS,
        CLS_MODELS,
        reg_metrics,
        cls_metrics
    )

# =========================================================
# TRAINING
# =========================================================

with st.spinner("Training models on real IPL dataset..."):

    (
        REG_MODELS,
        CLS_MODELS,
        REG_METRICS,
        CLS_METRICS
    ) = train_models()

st.success("Models Trained Successfully ✅")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([

    "🎯 Score Predictor",
    "🏆 Win Predictor",
    "📊 Model Metrics"

])

# =========================================================
# TAB 1 : SCORE PREDICTION
# =========================================================

with tab1:

    st.subheader("Predict Final Innings Score")

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
            "ML Model",
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
            "Overs Completed",
            0.1,
            20.0,
            10.0
        )

    if st.button("🎯 Predict Final Score"):

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
        <div class="result-box">

            <p>Predicted Final Score</p>

            <p class="big-score">
                {prediction}
            </p>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2 : WIN PREDICTION
# =========================================================

with tab2:

    st.subheader("Predict Match Winner")

    col1, col2 = st.columns(2)

    with col1:

        chasing_team = st.selectbox(
            "Chasing Team",
            IPL_TEAMS
        )

        defending_team = st.selectbox(
            "Defending Team",
            [x for x in IPL_TEAMS if x != chasing_team]
        )

        venue2 = st.selectbox(
            "Venue",
            IPL_VENUES,
            key="venue2"
        )

        model_name2 = st.selectbox(
            "ML Model",
            list(CLS_MODELS.keys())
        )

    with col2:

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
            10.0
        )

    if st.button("🏆 Predict Winner"):

        crr = current_score / max(overs2, 0.1)

        rrr = (
            (target - current_score) /
            max((20 - overs2), 0.1)
        )

        X2 = np.array([[
            TEAM_ENC[chasing_team],
            TEAM_ENC[defending_team],
            VENUE_ENC[venue2],
            target,
            current_score,
            wickets2,
            overs2,
            crr,
            rrr
        ]])

        model = CLS_MODELS[model_name2]

        pred = model.predict(X2)[0]

        prob = model.predict_proba(X2)[0]

        winner = (
            chasing_team
            if pred == 1
            else defending_team
        )

        confidence = round(
            max(prob) * 100,
            2
        )

        st.markdown(f"""
        <div class="result-box">

            <p>Predicted Winner</p>

            <p class="win-team">
                {winner}
            </p>

            <p>
                Confidence: {confidence}%
            </p>

        </div>
        """, unsafe_allow_html=True)

        st.progress(int(prob[1] * 100))

# =========================================================
# TAB 3 : METRICS
# =========================================================

with tab3:

    st.subheader("Regression Models")

    reg_table = pd.DataFrame(REG_METRICS).T

    st.dataframe(
        reg_table,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Classification Models")

    cls_table = pd.DataFrame(CLS_METRICS).T

    st.dataframe(
        cls_table,
        use_container_width=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <center>
    IPL Predictor using Real IPL Dataset + Machine Learning
    </center>
    """,
    unsafe_allow_html=True
)
