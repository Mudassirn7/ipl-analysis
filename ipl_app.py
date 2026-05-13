# =========================================================
# IPL PREDICTOR — FINAL ENHANCED VERSION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
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
    font-size:60px;
    font-weight:bold;
    color:#ff6b00 !important;
}

.result-box {
    padding:25px;
    border-radius:15px;
    background:#fff3e6;
    border:2px solid #ff6b00;
    text-align:center;
    margin-top:20px;
}

.big-score {
    font-size:70px;
    font-weight:bold;
    color:#ff6b00;
}

.win-team {
    font-size:45px;
    font-weight:bold;
    color:green;
}

.stButton button {
    width:100%;
    background:#ff6b00;
    color:white;
    border:none;
    border-radius:10px;
    padding:12px;
    font-size:20px;
}

.analysis-card {
    background:#f8f9fa;
    border-left:4px solid #ff6b00;
    padding:15px;
    margin:10px 0;
    border-radius:5px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<p class='ipl-title'>🏏 IPL Predictor Dashboard</p>",
    unsafe_allow_html=True
)

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
            path=".",
            unzip=True
        )

    except Exception as e:
        st.error(f"Dataset download failed: {e}")
        st.stop()

if not os.path.exists(CSV_FILE):
    st.error("IPL.csv not found.")
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
    "Punjab Cricket Association IS Bindra Stadium": "PCA Stadium"
}

raw_df["venue"] = raw_df["venue"].replace(VENUE_MAPPING)

# =========================================================
# IPL TEAMS
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

df["current_run_rate"] = (
    df["team_runs"] /
    df["overs_completed"].replace(0, 0.1)
)

df["final_score"] = df.groupby(
    ["match_id", "innings"]
)["team_runs"].transform("max")

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

score_df = df[
    (df["overs_completed"] >= 6) &
    (df["overs_completed"] <= 16)
][[
    "batting_team",
    "bowling_team",
    "venue",
    "team_runs",
    "team_wicket",
    "overs_completed",
    "current_run_rate",
    "final_score"
]].dropna().copy()

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
)["team_runs"].max().reset_index()

targets.columns = ["match_id", "target"]

match_results = df[df["innings"] == 2].copy()

match_results = match_results.merge(
    targets,
    on="match_id"
)

match_end = match_results.sort_values(
    ["match_id", "over", "ball"]
).groupby("match_id").last().reset_index()

match_end["won_chase"] = (
    match_end["team_runs"] >= match_end["target"]
).astype(int)

winner_lookup = match_end[[
    "match_id",
    "won_chase"
]]

win_df = df[
    (df["innings"] == 2) &
    (df["overs_completed"] >= 6) &
    (df["overs_completed"] <= 18)
].copy()

win_df = win_df.merge(targets, on="match_id")
win_df = win_df.merge(winner_lookup, on="match_id")

win_df["runs_needed"] = (
    win_df["target"] - win_df["team_runs"]
)

win_df["balls_left"] = (
    120 - ((win_df["over"] * 6) + win_df["ball"])
)

win_df["balls_left"] = win_df["balls_left"].replace(0, 1)

win_df["required_run_rate"] = (
    win_df["runs_needed"] * 6 /
    win_df["balls_left"]
)

win_df["pct_target_done"] = (
    win_df["team_runs"] /
    win_df["target"].replace(0, 1)
)

win_df["pct_overs_done"] = (
    win_df["overs_completed"] / 20
)

win_df = win_df[[
    "batting_team",
    "bowling_team",
    "venue",
    "target",
    "team_wicket",
    "overs_completed",
    "current_run_rate",
    "required_run_rate",
    "pct_target_done",
    "pct_overs_done",
    "won_chase"
]].dropna()

win_df["batting_team"] = win_df["batting_team"].map(TEAM_ENC)
win_df["bowling_team"] = win_df["bowling_team"].map(TEAM_ENC)
win_df["venue"] = win_df["venue"].map(VENUE_ENC)

# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models():

    # =====================================================
    # REGRESSION
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

    REG_MODELS = {

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        ),

        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=6,
            min_samples_leaf=15,
            random_state=42
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }

    for model in REG_MODELS.values():
        model.fit(Xr_train, yr_train)

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    Xc = win_df[[
        "batting_team",
        "bowling_team",
        "venue",
        "target",
        "team_wicket",
        "overs_completed",
        "current_run_rate",
        "required_run_rate",
        "pct_target_done",
        "pct_overs_done"
    ]]

    yc = win_df["won_chase"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc,
        yc,
        test_size=0.2,
        random_state=42
    )

    CLS_MODELS = {

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=15,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            random_state=42
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42
        )
    }

    for model in CLS_MODELS.values():
        model.fit(Xc_train, yc_train)

    # =====================================================
    # METRICS
    # =====================================================

    reg_metrics = {}

    for name, model in REG_MODELS.items():

        train_pred = model.predict(Xr_train)
        test_pred = model.predict(Xr_test)

        reg_metrics[name] = {

            "Train R²": round(
                r2_score(yr_train, train_pred), 4
            ),

            "Test R²": round(
                r2_score(yr_test, test_pred), 4
            ),

            "RMSE": round(
                np.sqrt(mean_squared_error(
                    yr_test,
                    test_pred
                )), 2
            ),

            "MAE": round(
                mean_absolute_error(
                    yr_test,
                    test_pred
                ), 2
            )
        }

    cls_metrics = {}

    for name, model in CLS_MODELS.items():

        train_pred = model.predict(Xc_train)
        test_pred = model.predict(Xc_test)

        cls_metrics[name] = {

            "Train Acc": round(
                accuracy_score(
                    yc_train,
                    train_pred
                ) * 100, 2
            ),

            "Test Acc": round(
                accuracy_score(
                    yc_test,
                    test_pred
                ) * 100, 2
            ),

            "Precision": round(
                precision_score(
                    yc_test,
                    test_pred
                ) * 100, 2
            ),

            "Recall": round(
                recall_score(
                    yc_test,
                    test_pred
                ) * 100, 2
            ),

            "F1": round(
                f1_score(
                    yc_test,
                    test_pred
                ) * 100, 2
            )
        }

    return (
        REG_MODELS,
        CLS_MODELS,
        reg_metrics,
        cls_metrics,
        Xr_train,
        Xr_test,
        yr_train,
        yr_test,
        Xc_train,
        Xc_test,
        yc_train,
        yc_test
    )

# =========================================================
# TRAIN
# =========================================================

with st.spinner("Training Models..."):

    (
        REG_MODELS,
        CLS_MODELS,
        REG_METRICS,
        CLS_METRICS,
        Xr_train,
        Xr_test,
        yr_train,
        yr_test,
        Xc_train,
        Xc_test,
        yc_train,
        yc_test
    ) = train_models()

st.success("Models Trained Successfully ✅")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "🎯 Score Predictor",
    "🏆 Win Predictor",
    "📊 Model Report",
    "🔍 Data Analysis",
    "📉 Visual Analytics"

])

# =========================================================
# TAB 1 — SCORE PREDICTOR
# =========================================================

with tab1:

    st.subheader("Predict Final Score")

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
            "Wickets",
            0,
            9,
            2
        )

        over_num = st.slider(
            "Overs",
            0,
            19,
            10
        )

        ball_num = st.slider(
            "Balls",
            0,
            5,
            0
        )

        overs = round(
            over_num + (ball_num / 6),
            2
        )

    if st.button("Predict Final Score"):

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
            IPL_VENUES
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

        over_num2 = st.slider(
            "Overs",
            0,
            19,
            10
        )

        ball_num2 = st.slider(
            "Balls",
            0,
            5,
            0
        )

        overs2 = round(
            over_num2 + (ball_num2 / 6),
            2
        )

    if st.button("Predict Winner"):

        crr = current_score / max(overs2, 0.1)

        rrr = (
            (target - current_score) * 6 /
            max((120 - overs2 * 6), 1)
        )

        pct_done = current_score / max(target, 1)

        pct_overs = overs2 / 20

        X2 = np.array([[
            TEAM_ENC[chasing_team],
            TEAM_ENC[defending_team],
            VENUE_ENC[venue2],
            target,
            wickets2,
            overs2,
            crr,
            rrr,
            pct_done,
            pct_overs
        ]])

        model = CLS_MODELS[model_name2]

        pred = model.predict(X2)[0]

        prob = model.predict_proba(X2)[0]

        winner = (
            chasing_team
            if pred == 1
            else defending_team
        )

        confidence = round(max(prob) * 100, 2)

        st.markdown(f"""
        <div class="result-box">
        <p>Predicted Winner</p>
        <p class="win-team">{winner}</p>
        <p>Confidence: {confidence}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))

# =========================================================
# TAB 3 — MODEL REPORT
# =========================================================

with tab3:

    st.subheader("Regression Model Report")

    reg_table = pd.DataFrame(REG_METRICS).T

    st.dataframe(
        reg_table,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Classification Model Report")

    cls_table = pd.DataFrame(CLS_METRICS).T

    st.dataframe(
        cls_table,
        use_container_width=True
    )

# =========================================================
# TAB 4 — DATA ANALYSIS
# =========================================================

with tab4:

    st.subheader("Dataset Analysis")

    st.metric(
        "Total Matches",
        df["match_id"].nunique()
    )

    st.metric(
        "Venues",
        len(IPL_VENUES)
    )

    st.metric(
        "Rows",
        len(df)
    )

    st.markdown("---")

    st.subheader("Top Teams by Runs")

    team_runs = df.groupby(
        "batting_team"
    )["team_runs"].sum().sort_values(
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        team_runs.index,
        team_runs.values
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

# =========================================================
# TAB 5 — VISUAL ANALYTICS
# =========================================================

with tab5:

    st.subheader("📉 Regression Model Comparison")

    reg_df = pd.DataFrame(
        REG_METRICS
    ).T.reset_index()

    reg_df.columns = [
        "Model",
        "Train R2",
        "Test R2",
        "RMSE",
        "MAE"
    ]

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(
        reg_df["Model"],
        reg_df["Test R2"]
    )

    ax1.set_title(
        "Regression Model Test R²"
    )

    st.pyplot(fig1)

    # =====================================================
    # CLASSIFICATION GRAPH
    # =====================================================

    st.markdown("---")

    st.subheader("🏆 Classification Accuracy")

    cls_df = pd.DataFrame(
        CLS_METRICS
    ).T.reset_index()

    cls_df.columns = [
        "Model",
        "Train Acc",
        "Test Acc",
        "Precision",
        "Recall",
        "F1"
    ]

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.bar(
        cls_df["Model"],
        cls_df["Test Acc"]
    )

    ax2.set_title(
        "Classification Accuracy"
    )

    st.pyplot(fig2)

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.markdown("---")

    st.subheader("Confusion Matrix")

    best_model = CLS_MODELS["Random Forest"]

    preds = best_model.predict(Xc_test)

    cm = confusion_matrix(
        yc_test,
        preds
    )

    fig3, ax3 = plt.subplots(figsize=(5, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        ax=ax3
    )

    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    st.pyplot(fig3)

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.markdown("---")

    st.subheader("Feature Importance")

    rf_model = REG_MODELS["Random Forest"]

    importance_df = pd.DataFrame({
        "Feature": Xr_train.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    fig4, ax4 = plt.subplots(figsize=(10, 5))

    ax4.bar(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xticks(rotation=45)

    st.pyplot(fig4)

    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    st.markdown("---")

    st.subheader("Actual vs Predicted")

    pred_scores = rf_model.predict(Xr_test)

    fig5, ax5 = plt.subplots(figsize=(7, 7))

    ax5.scatter(
        yr_test,
        pred_scores
    )

    ax5.set_xlabel("Actual")
    ax5.set_ylabel("Predicted")

    st.pyplot(fig5)

    # =====================================================
    # RESIDUAL PLOT
    # =====================================================

    st.markdown("---")

    st.subheader("Residual Plot")

    residuals = yr_test - pred_scores

    fig6, ax6 = plt.subplots(figsize=(10, 5))

    ax6.scatter(
        pred_scores,
        residuals
    )

    ax6.axhline(
        y=0,
        linestyle="--"
    )

    ax6.set_xlabel("Predicted")
    ax6.set_ylabel("Residuals")

    st.pyplot(fig6)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "<center>🏏 IPL Predictor Dashboard | Final Year Project ML System</center>",
    unsafe_allow_html=True
)
