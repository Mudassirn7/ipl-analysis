# =========================================================
# IPL PREDICTOR — FULL VERSION
# Models: RF, GB, LR, DT, AdaBoost, KNN, SVM, XGBoost
# Features: Comparison Charts + Business Insights Tab
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="IPL Predictor", page_icon="🏏", layout="wide")

st.markdown("""
<style>
html, body, .stApp { background-color: white; }
h1,h2,h3,h4,h5,h6,p,div,span,label { color: black !important; }
.ipl-title { text-align:center; font-size:60px; font-weight:bold; color:#ff6b00 !important; }
.result-box { padding:25px; border-radius:15px; background:#fff3e6;
              border:2px solid #ff6b00; text-align:center; margin-top:20px; }
.big-score  { font-size:70px; font-weight:bold; color:#ff6b00; }
.win-team   { font-size:45px; font-weight:bold; color:green; }
.stButton button { width:100%; background:#ff6b00; color:white; border:none;
                   border-radius:10px; padding:12px; font-size:20px; }
.analysis-card { background:#f8f9fa; border-left:4px solid #ff6b00;
                 padding:15px; margin:10px 0; border-radius:5px; }
.tuning-card   { background:#e8f4fd; border-left:4px solid #0066cc;
                 padding:15px; margin:10px 0; border-radius:5px; }
.biz-card      { background:#e8fdf0; border-left:4px solid #00aa44;
                 padding:15px; margin:10px 0; border-radius:5px; }
.rec-card      { background:#fff8e8; border-left:4px solid #ffaa00;
                 padding:15px; margin:10px 0; border-radius:5px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='ipl-title'>🏏 IPL Predictor</p>", unsafe_allow_html=True)
st.markdown("---")

# =========================================================
# LOAD DATASET
# =========================================================

CSV_FILE = "IPL.csv"

if not os.path.exists(CSV_FILE):
    try:
        os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
        os.environ["KAGGLE_KEY"]      = st.secrets["kaggle"]["key"]
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "chaitu20/ipl-dataset2008-2025", path=".", unzip=True
        )
    except Exception as e:
        st.error(f"Dataset download failed: {e}")
        st.stop()

if not os.path.exists(CSV_FILE):
    st.error("IPL.csv not found after download.")
    st.stop()

# =========================================================
# LOAD & CLEAN DATA
# =========================================================

raw_df = pd.read_csv(CSV_FILE)

TEAM_NAME_MAPPING = {
    "Delhi Daredevils":  "Delhi Capitals",
    "Kings XI Punjab":   "Punjab Kings",
    "Deccan Chargers":   "Sunrisers Hyderabad",
    "Gujarat Lions":     "Gujarat Titans"
}
VENUE_MAPPING = {
    "Arun Jaitley Stadium, Delhi":              "Arun Jaitley Stadium",
    "M Chinnaswamy Stadium":                    "M. Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium":                    "M. Chinnaswamy Stadium",
    "MA Chidambaram Stadium":                   "M. A. Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk":          "M. A. Chidambaram Stadium",
    "Punjab Cricket Association Stadium":       "PCA Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "PCA Stadium",
    "Rajiv Gandhi International Stadium":       "Rajiv Gandhi Intl. Cricket Stadium",
    "Rajiv Gandhi Intl. Cricket Stadium":       "Rajiv Gandhi Intl. Cricket Stadium"
}

raw_df["batting_team"] = raw_df["batting_team"].replace(TEAM_NAME_MAPPING)
raw_df["bowling_team"] = raw_df["bowling_team"].replace(TEAM_NAME_MAPPING)
raw_df["venue"]        = raw_df["venue"].replace(VENUE_MAPPING)

IPL_TEAMS = [
    "Chennai Super Kings", "Mumbai Indians",
    "Royal Challengers Bangalore", "Kolkata Knight Riders",
    "Delhi Capitals", "Sunrisers Hyderabad",
    "Rajasthan Royals", "Punjab Kings",
    "Lucknow Super Giants", "Gujarat Titans"
]

df = raw_df[
    raw_df["batting_team"].isin(IPL_TEAMS) &
    raw_df["bowling_team"].isin(IPL_TEAMS)
].copy()

IPL_VENUES = sorted(df["venue"].dropna().unique().tolist())

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["overs_completed"]  = df["over"] + (df["ball"] / 6)
df["current_run_rate"] = df["team_runs"] / df["overs_completed"].replace(0, 0.1)
df["final_score"]      = df.groupby(["match_id", "innings"])["team_runs"].transform("max")

# =========================================================
# ENCODING
# =========================================================

TEAM_ENC  = {team: idx for idx, team in enumerate(IPL_TEAMS)}
VENUE_ENC = {venue: idx for idx, venue in enumerate(IPL_VENUES)}

# =========================================================
# SCORE DATASET  (overs 6-16)
# =========================================================

score_df = df[
    df["overs_completed"].between(6, 16)
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
score_df["venue"]        = score_df["venue"].map(VENUE_ENC)
score_df = score_df.dropna()

# =========================================================
# WIN DATASET  (overs 6-18, no leakage)
# =========================================================

innings1        = df[df["innings"] == 1]
targets         = innings1.groupby("match_id")["team_runs"].max().reset_index()
targets.columns = ["match_id", "target"]

match_results = df[df["innings"] == 2].copy().merge(targets, on="match_id")
match_end     = (match_results
                 .sort_values(["match_id", "over", "ball"])
                 .groupby("match_id").last().reset_index())
match_end["won_chase"]  = (match_end["team_runs"] >= match_end["target"]).astype(int)
winner_lookup           = match_end[["match_id", "won_chase"]]

win_df = (df[(df["innings"] == 2) & df["overs_completed"].between(6, 18)]
          .copy()
          .merge(targets, on="match_id")
          .merge(winner_lookup, on="match_id"))

win_df["runs_needed"]       = win_df["target"] - win_df["team_runs"]
win_df["balls_left"]        = (120 - (win_df["over"] * 6 + win_df["ball"])).replace(0, 1)
win_df["required_run_rate"] = win_df["runs_needed"] * 6 / win_df["balls_left"]
win_df["pct_target_done"]   = win_df["team_runs"] / win_df["target"].replace(0, 1)
win_df["pct_overs_done"]    = win_df["overs_completed"] / 20

win_df = win_df[[
    "batting_team", "bowling_team", "venue",
    "target", "team_wicket", "overs_completed",
    "current_run_rate", "required_run_rate",
    "pct_target_done", "pct_overs_done", "won_chase"
]].dropna().copy()

win_df["batting_team"] = win_df["batting_team"].map(TEAM_ENC)
win_df["bowling_team"] = win_df["bowling_team"].map(TEAM_ENC)
win_df["venue"]        = win_df["venue"].map(VENUE_ENC)
win_df = win_df.dropna()

# =========================================================
# TRAIN ALL MODELS
# =========================================================

@st.cache_resource
def train_models():

    # ---------- REGRESSION ----------
    Xr = score_df[["batting_team","bowling_team","venue",
                   "current_runs","wickets","overs","crr"]]
    yr = score_df["final_score"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )

    REG_MODELS = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=6, min_samples_leaf=15, random_state=42
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=7, n_jobs=-1))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVR(kernel="rbf", C=10, epsilon=5))
        ]),
    }
    if XGBOOST_AVAILABLE:
        REG_MODELS["XGBoost"] = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=0, nthread=-1, tree_method="hist"
        )

    # SVM is slow — train on subset
    svm_idx = Xr_train.sample(n=min(5000, len(Xr_train)), random_state=42).index
    for name, model in REG_MODELS.items():
        if name == "SVM":
            model.fit(Xr_train.loc[svm_idx], yr_train.loc[svm_idx])
        else:
            model.fit(Xr_train, yr_train)

    # ---------- CLASSIFICATION ----------
    Xc = win_df[[
        "batting_team","bowling_team","venue","target","team_wicket",
        "overs_completed","current_run_rate","required_run_rate",
        "pct_target_done","pct_overs_done"
    ]]
    yc = win_df["won_chase"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=42
    )

    CLS_MODELS = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=15,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=500, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=7, n_jobs=-1))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1, probability=True))
        ]),
    }
    if XGBOOST_AVAILABLE:
        CLS_MODELS["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=0, nthread=-1,
            tree_method="hist", eval_metric="logloss"
        )

    svm_idx_c = Xc_train.sample(n=min(5000, len(Xc_train)), random_state=42).index
    for name, model in CLS_MODELS.items():
        if name == "SVM":
            model.fit(Xc_train.loc[svm_idx_c], yc_train.loc[svm_idx_c])
        else:
            model.fit(Xc_train, yc_train)

    # ---------- HYPERPARAMETER TUNING (RandomizedSearchCV) ----------
    param_dist = {
        "n_estimators":     [50, 100, 150],
        "max_depth":        [4, 6, 8],
        "min_samples_leaf": [5, 10, 20]
    }

    rf_reg_tuned = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_dist, n_iter=10, cv=3,
        scoring="r2", n_jobs=-1, random_state=42
    )
    rf_reg_tuned.fit(Xr_train, yr_train)
    best_reg_params = rf_reg_tuned.best_params_
    REG_MODELS["RF (Tuned)"] = rf_reg_tuned.best_estimator_

    rf_cls_tuned = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_dist, n_iter=10, cv=3,
        scoring="accuracy", n_jobs=-1, random_state=42
    )
    rf_cls_tuned.fit(Xc_train, yc_train)
    best_cls_params = rf_cls_tuned.best_params_
    CLS_MODELS["RF (Tuned)"] = rf_cls_tuned.best_estimator_

    # ---------- METRICS ----------
    reg_metrics = {}
    for name, model in REG_MODELS.items():
        tr_p  = model.predict(Xr_train)
        te_p  = model.predict(Xr_test)
        tr_r2 = r2_score(yr_train, tr_p)
        te_r2 = r2_score(yr_test,  te_p)
        reg_metrics[name] = {
            "Train R²": round(tr_r2, 4),
            "Test R²":  round(te_r2, 4),
            "RMSE":     round(np.sqrt(mean_squared_error(yr_test, te_p)), 2),
            "MAE":      round(mean_absolute_error(yr_test, te_p), 2),
            "Overfit":  "Yes" if abs(tr_r2 - te_r2) > 0.10 else "No"
        }

    cls_metrics = {}
    for name, model in CLS_MODELS.items():
        tr_p   = model.predict(Xc_train)
        te_p   = model.predict(Xc_test)
        tr_acc = accuracy_score(yc_train, tr_p) * 100
        te_acc = accuracy_score(yc_test,  te_p) * 100
        cls_metrics[name] = {
            "Train Acc %": round(tr_acc, 2),
            "Test Acc %":  round(te_acc, 2),
            "Precision %": round(precision_score(yc_test, te_p, zero_division=0) * 100, 2),
            "Recall %":    round(recall_score(yc_test, te_p, zero_division=0) * 100, 2),
            "F1 Score %":  round(f1_score(yc_test, te_p, zero_division=0) * 100, 2),
            "Overfit":     "Yes" if abs(tr_acc - te_acc) > 8 else "No"
        }

    return (
        REG_MODELS, CLS_MODELS,
        reg_metrics, cls_metrics,
        Xr_train, Xr_test, yr_train, yr_test,
        Xc_train, Xc_test, yc_train, yc_test,
        best_reg_params, best_cls_params
    )

# =========================================================
# TRAIN
# =========================================================

with st.spinner("⏳ Training all models — please wait (~20–30s)..."):
    (
        REG_MODELS, CLS_MODELS,
        REG_METRICS, CLS_METRICS,
        Xr_train, Xr_test, yr_train, yr_test,
        Xc_train, Xc_test, yc_train, yc_test,
        BEST_REG_PARAMS, BEST_CLS_PARAMS
    ) = train_models()

model_count = "9" if XGBOOST_AVAILABLE else "8"
st.success(f"✅ {model_count} Regression + {model_count} Classification Models Trained Successfully!")

# =========================================================
# HELPER — HORIZONTAL BAR CHART
# =========================================================

def plot_bar(metrics_dict, metric_key, title, higher_better=True):
    names  = list(metrics_dict.keys())
    values = [metrics_dict[n][metric_key] for n in names]
    best   = max(values) if higher_better else min(values)
    colors = ["#ff6b00" if v == best else "#ffc299" for v in values]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.55)
    ax.set_xlabel(metric_key, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=9
        )
    plt.tight_layout()
    return fig

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Score Predictor",
    "🏆 Win Predictor",
    "📊 Model Report",
    "📈 Model Comparison",
    "🔍 Data Analysis",
    "💼 Business Insights"
])

# =========================================================
# TAB 1 — SCORE PREDICTOR
# =========================================================

with tab1:
    st.subheader("🎯 Predict Final Score")
    st.caption("Enter mid-game state (overs 6–16) to predict the innings total.")

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox("Batting Team", IPL_TEAMS, key="bat1")
        bowling_team = st.selectbox(
            "Bowling Team",
            [x for x in IPL_TEAMS if x != batting_team],
            key="bowl1"
        )
        venue      = st.selectbox("Venue", IPL_VENUES, key="venue1")
        model_name = st.selectbox("Select ML Model", list(REG_MODELS.keys()), key="model1")

    with col2:
        current_runs = st.number_input("Current Runs", 0, 300, 80)
        wickets      = st.slider("Wickets Fallen", 0, 9, 2, key="wk1")
        over_num     = st.slider("Overs", 0, 19, 10, key="ov1")
        ball_num     = st.slider("Balls in Current Over", 0, 5, 0, key="ball1")
        overs        = round(over_num + (ball_num / 6), 2)
        st.caption(f"📍 Overs: {over_num}.{ball_num}")

    if st.button("🔮 Predict Final Score", key="btn1"):
        crr = current_runs / max(overs, 0.1)
        X   = np.array([[
            TEAM_ENC[batting_team], TEAM_ENC[bowling_team],
            VENUE_ENC[venue], current_runs, wickets, overs, crr
        ]])
        prediction = int(REG_MODELS[model_name].predict(X)[0])

        st.markdown(f"""
        <div class="result-box">
            <p style="font-size:20px;">Predicted Final Score for <b>{batting_team}</b></p>
            <p class="big-score">{prediction}</p>
            <p style="font-size:16px;">Model Used: <b>{model_name}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Mini model comparison on same inputs
        st.markdown("#### All Models — Prediction Comparison")
        preds = {
            name: int(model.predict(X)[0])
            for name, model in REG_MODELS.items()
        }
        pred_df = pd.DataFrame(
            list(preds.items()), columns=["Model", "Predicted Score"]
        ).sort_values("Predicted Score", ascending=False)
        st.dataframe(pred_df, use_container_width=True)

# =========================================================
# TAB 2 — WIN PREDICTOR
# =========================================================

with tab2:
    st.subheader("🏆 Predict Match Winner")
    st.caption("Enter 2nd innings mid-game state to predict win probability.")

    col1, col2 = st.columns(2)

    with col1:
        chasing_team   = st.selectbox("Chasing Team", IPL_TEAMS, key="ct")
        defending_team = st.selectbox(
            "Defending Team",
            [x for x in IPL_TEAMS if x != chasing_team],
            key="dt"
        )
        venue2      = st.selectbox("Venue", IPL_VENUES, key="venue2")
        model_name2 = st.selectbox("Select ML Model", list(CLS_MODELS.keys()), key="model2")

    with col2:
        target        = st.number_input("Target", 50, 300, 180)
        current_score = st.number_input("Current Score", 0, 300, 90)
        wickets2      = st.slider("Wickets Fallen", 0, 9, 3, key="wk2")
        over_num2     = st.slider("Overs", 0, 19, 10, key="ov2")
        ball_num2     = st.slider("Balls in Current Over", 0, 5, 0, key="ball2")
        overs2        = round(over_num2 + (ball_num2 / 6), 2)
        st.caption(f"📍 Overs: {over_num2}.{ball_num2}")

    if st.button("🔮 Predict Winner", key="btn2"):
        crr      = current_score / max(overs2, 0.1)
        rrr      = (target - current_score) * 6 / max((120 - overs2 * 6), 1)
        pct_done = current_score / max(target, 1)
        pct_ovrs = overs2 / 20

        X2 = np.array([[
            TEAM_ENC[chasing_team], TEAM_ENC[defending_team],
            VENUE_ENC[venue2], target, wickets2,
            overs2, crr, rrr, pct_done, pct_ovrs
        ]])

        model  = CLS_MODELS[model_name2]
        pred   = model.predict(X2)[0]
        prob   = model.predict_proba(X2)[0]
        winner = chasing_team if pred == 1 else defending_team
        conf   = round(max(prob) * 100, 2)

        st.markdown(f"""
        <div class="result-box">
            <p style="font-size:20px;">Predicted Winner</p>
            <p class="win-team">{winner}</p>
            <p style="font-size:18px;">Confidence: <b>{conf}%</b></p>
            <p style="font-size:15px;">Model Used: <b>{model_name2}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Win probability bar for all models
        st.markdown("#### All Models — Win Probability for Chasing Team")
        win_probs = {}
        for name, mdl in CLS_MODELS.items():
            try:
                p = mdl.predict_proba(X2)[0]
                win_probs[name] = round(p[1] * 100, 2)
            except Exception:
                win_probs[name] = round(mdl.predict(X2)[0] * 100, 2)

        wp_df = pd.DataFrame(
            list(win_probs.items()), columns=["Model", f"Win % ({chasing_team})"]
        ).sort_values(f"Win % ({chasing_team})", ascending=False)
        st.dataframe(wp_df, use_container_width=True)

# =========================================================
# TAB 3 — MODEL REPORT (tables)
# =========================================================

with tab3:
    st.subheader("📈 Regression Model Report")
    st.caption("Target: Final innings score | Features: 7 mid-game variables | Split: 80/20")
    st.dataframe(pd.DataFrame(REG_METRICS).T, use_container_width=True)

    st.markdown("---")

    st.subheader("🏆 Classification Model Report")
    st.caption("Target: won_chase (0/1) | Features: 10 mid-game variables | Split: 80/20")
    st.dataframe(pd.DataFrame(CLS_METRICS).T, use_container_width=True)

    st.markdown("---")
    st.subheader("⚙️ Hyperparameter Tuning — RandomizedSearchCV")
    st.markdown("""
    <div class="tuning-card">
    <b>Method:</b> RandomizedSearchCV — 10 random combinations × 3-fold CV<br>
    <b>Model Tuned:</b> Random Forest (Regression + Classification)<br>
    <b>Parameter Space:</b> n_estimators [50,100,150] | max_depth [4,6,8] | min_samples_leaf [5,10,20]<br>
    <b>Advantage:</b> Explores 10 out of 27 combinations — 3× faster than GridSearchCV
    with near-identical accuracy improvement.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Best Params — RF Tuned (Regression)**")
        st.json(BEST_REG_PARAMS)
    with c2:
        st.markdown("**Best Params — RF Tuned (Classification)**")
        st.json(BEST_CLS_PARAMS)

    if "RF (Tuned)" in REG_METRICS and "Random Forest" in REG_METRICS:
        base_r2   = REG_METRICS["Random Forest"]["Test R²"]
        tuned_r2  = REG_METRICS["RF (Tuned)"]["Test R²"]
        base_acc  = CLS_METRICS["Random Forest"]["Test Acc %"]
        tuned_acc = CLS_METRICS["RF (Tuned)"]["Test Acc %"]
        st.markdown(f"""
        <div class="tuning-card">
        <b>Tuning Impact:</b><br>
        &nbsp;&nbsp;• Regression R²: {base_r2} → {tuned_r2}
          {'✅ Improved' if tuned_r2 > base_r2 else '(already near-optimal)'}<br>
        &nbsp;&nbsp;• Classification Accuracy: {base_acc}% → {tuned_acc}%
          {'✅ Improved' if tuned_acc > base_acc else '(already near-optimal)'}
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 4 — MODEL COMPARISON (charts)
# =========================================================

with tab4:
    st.subheader("📈 Regression — Visual Model Comparison")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        fig = plot_bar(REG_METRICS, "Test R²",  "Test R²  (↑ higher = better)", True)
        st.pyplot(fig); plt.close()
    with rc2:
        fig = plot_bar(REG_METRICS, "RMSE",     "RMSE  (↓ lower = better)",     False)
        st.pyplot(fig); plt.close()
    with rc3:
        fig = plot_bar(REG_METRICS, "MAE",      "MAE  (↓ lower = better)",      False)
        st.pyplot(fig); plt.close()

    # Train vs Test R² grouped bar
    st.markdown("#### Train vs Test R² — Overfitting Check")
    names     = list(REG_METRICS.keys())
    train_r2s = [REG_METRICS[n]["Train R²"] for n in names]
    test_r2s  = [REG_METRICS[n]["Test R²"]  for n in names]
    x = np.arange(len(names)); w = 0.35
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.bar(x - w/2, train_r2s, w, label="Train R²", color="#ff6b00", alpha=0.85)
    ax2.bar(x + w/2, test_r2s,  w, label="Test R²",  color="#0066cc", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax2.set_ylabel("R²"); ax2.set_title("Train vs Test R² per Model", fontweight="bold")
    ax2.legend(); ax2.spines[["top","right"]].set_visible(False)
    fig2.patch.set_facecolor("white"); plt.tight_layout()
    st.pyplot(fig2); plt.close()

    st.markdown("---")
    st.subheader("🏆 Classification — Visual Model Comparison")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        fig = plot_bar(CLS_METRICS, "Test Acc %",  "Test Accuracy %  (↑ higher)", True)
        st.pyplot(fig); plt.close()
    with cc2:
        fig = plot_bar(CLS_METRICS, "F1 Score %",  "F1 Score %  (↑ higher)",      True)
        st.pyplot(fig); plt.close()
    with cc3:
        fig = plot_bar(CLS_METRICS, "Precision %", "Precision %  (↑ higher)",     True)
        st.pyplot(fig); plt.close()

    # Train vs Test Accuracy grouped bar
    st.markdown("#### Train vs Test Accuracy — Overfitting Check")
    names2     = list(CLS_METRICS.keys())
    train_accs = [CLS_METRICS[n]["Train Acc %"] for n in names2]
    test_accs  = [CLS_METRICS[n]["Test Acc %"]  for n in names2]
    x2 = np.arange(len(names2))
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.bar(x2 - w/2, train_accs, w, label="Train Acc %", color="#ff6b00", alpha=0.85)
    ax3.bar(x2 + w/2, test_accs,  w, label="Test Acc %",  color="#0066cc", alpha=0.85)
    ax3.set_xticks(x2); ax3.set_xticklabels(names2, rotation=20, ha="right", fontsize=10)
    ax3.set_ylabel("Accuracy %"); ax3.set_title("Train vs Test Accuracy per Model", fontweight="bold")
    ax3.legend(); ax3.spines[["top","right"]].set_visible(False)
    fig3.patch.set_facecolor("white"); plt.tight_layout()
    st.pyplot(fig3); plt.close()

    # Recall chart
    st.markdown("#### Recall % — Classification Models")
    fig4 = plot_bar(CLS_METRICS, "Recall %", "Recall %  (↑ higher = better)", True)
    st.pyplot(fig4); plt.close()

# =========================================================
# TAB 5 — DATA ANALYSIS
# =========================================================

with tab5:
    st.subheader("🔍 Data Analysis & ML Pipeline")

    st.markdown("### Step 1 — Data Loading")
    st.markdown(f"""
    <div class="analysis-card">
    <b>Source:</b> Kaggle — IPL Ball-by-Ball Dataset (2008–2025)<br>
    <b>Total Rows:</b> {len(raw_df):,} &nbsp;|&nbsp; <b>Columns:</b> {raw_df.shape[1]}<br>
    <b>Key Columns:</b> match_id, batting_team, bowling_team, venue,
    over, ball, team_runs, team_wicket, innings
    </div>""", unsafe_allow_html=True)

    st.markdown("### Step 2 — Data Cleaning")
    null_counts = raw_df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    st.markdown(f"""
    <div class="analysis-card">
    <b>Duplicate Rows:</b> {raw_df.duplicated().sum()}<br>
    <b>Columns with Nulls:</b> {len(null_cols)}<br>
    <b>Team Name Fixes:</b> {len(TEAM_NAME_MAPPING)} mappings
    (e.g., "Delhi Daredevils" → "Delhi Capitals")<br>
    <b>Venue Fixes:</b> {len(VENUE_MAPPING)} mappings<br>
    <b>Filter:</b> Only current 10 IPL teams retained
    </div>""", unsafe_allow_html=True)

    st.markdown("### Step 3 — Feature Engineering")
    st.markdown("""
    <div class="analysis-card">
    • <b>overs_completed</b> = over + ball/6 — decimal overs<br>
    • <b>CRR</b> = team_runs / overs_completed<br>
    • <b>final_score</b> = max team_runs per match-innings (regression target)<br>
    • <b>RRR</b> = runs_needed × 6 / balls_left (2nd innings only)<br>
    • <b>pct_target_done</b> = score / target (removes absolute scale bias)<br>
    • <b>pct_overs_done</b> = overs / 20
    </div>""", unsafe_allow_html=True)

    st.markdown("### Step 4 — Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches",        str(df["match_id"].nunique()))
    c2.metric("Regression Rows",      str(len(score_df)))
    c3.metric("Classification Rows",  str(len(win_df)))
    c4.metric("Unique Venues",        str(len(IPL_VENUES)))

    st.markdown("### Step 5 — Models Used")
    model_info = {
        "Model": [
            "Random Forest","Gradient Boosting","Linear / Logistic Regression",
            "Decision Tree","AdaBoost","KNN","SVM","XGBoost","RF (Tuned)"
        ],
        "Category": [
            "Bagging","Boosting","Linear","Single Tree",
            "Boosting","Instance-based","Kernel","XGB Boosting","Tuned Ensemble"
        ],
        "Used For": ["Both"] * 9
    }
    st.dataframe(pd.DataFrame(model_info), use_container_width=True)

    st.markdown("### Step 6 — Evaluation Metrics")
    st.markdown("""
    <div class="analysis-card">
    <b>Regression:</b> R² (Train & Test), RMSE, MAE, Overfit Check (|ΔR²| > 0.10)<br>
    <b>Classification:</b> Accuracy, Precision, Recall, F1 Score, Overfit Check (|ΔAcc| > 8%)
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Team Ball-by-Ball Row Count")
    team_counts = pd.concat([df["batting_team"], df["bowling_team"]]) \
                    .value_counts().reset_index()
    team_counts.columns = ["Team", "Ball-by-Ball Rows"]
    st.dataframe(team_counts, use_container_width=True)

# =========================================================
# TAB 6 — BUSINESS INSIGHTS
# =========================================================

with tab6:
    st.subheader("💼 Business-Oriented Recommendations")

    best_reg_model = max(REG_METRICS, key=lambda k: REG_METRICS[k]["Test R²"])
    best_cls_model = max(CLS_METRICS, key=lambda k: CLS_METRICS[k]["Test Acc %"])
    best_reg_r2    = REG_METRICS[best_reg_model]["Test R²"]
    best_reg_mae   = REG_METRICS[best_reg_model]["MAE"]
    best_cls_acc   = CLS_METRICS[best_cls_model]["Test Acc %"]
    best_cls_f1    = CLS_METRICS[best_cls_model]["F1 Score %"]

    # ---- Best models ----
    st.markdown("### 🥇 Best Performing Models")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"""
        <div class="biz-card">
        <b>✅ Best Score Prediction Model</b><br>
        <span style="font-size:22px;color:#00aa44;font-weight:bold;">{best_reg_model}</span><br>
        Test R² = {best_reg_r2} &nbsp;|&nbsp; MAE = {best_reg_mae} runs<br><br>
        <b>Business Use:</b> Real-time broadcast overlays showing projected innings total.
        </div>""", unsafe_allow_html=True)
    with b2:
        st.markdown(f"""
        <div class="biz-card">
        <b>✅ Best Win Prediction Model</b><br>
        <span style="font-size:22px;color:#00aa44;font-weight:bold;">{best_cls_model}</span><br>
        Test Accuracy = {best_cls_acc}% &nbsp;|&nbsp; F1 = {best_cls_f1}%<br><br>
        <b>Business Use:</b> Live win-probability meters for fantasy apps & sportsbooks.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- Use-case guide ----
    st.markdown("### 📌 Model Selection by Business Use Case")
    st.markdown("""
    <div class="rec-card">
    <b>🎙️ TV Broadcasters (Score Overlay):</b>
    Use Gradient Boosting / XGBoost — lowest MAE, best real-time accuracy.<br><br>
    <b>📱 Fantasy Apps (Dream11, MPL) — Win Probability:</b>
    Use RF (Tuned) — highest accuracy, well-calibrated probability scores.<br><br>
    <b>📊 Team Management & Analysts:</b>
    Ensemble of top-3 models — reduces variance for strategic decisions.<br><br>
    <b>💰 Sports Betting / Sportsbooks:</b>
    Logistic Regression or SVM — interpretable, calibrated probabilities.<br><br>
    <b>⚡ Real-Time APIs (low latency):</b>
    Decision Tree or Linear Regression — sub-millisecond inference, no scaling needed.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- Overfitting analysis ----
    st.markdown("### ⚠️ Overfitting Analysis")
    overfit_reg = {k for k, v in REG_METRICS.items() if v["Overfit"] == "Yes"}
    overfit_cls = {k for k, v in CLS_METRICS.items() if v["Overfit"] == "Yes"}

    if overfit_reg or overfit_cls:
        st.markdown(f"""
        <div class="rec-card">
        <b>Overfit — Regression:</b>  {', '.join(overfit_reg) if overfit_reg else 'None'}<br>
        <b>Overfit — Classification:</b> {', '.join(overfit_cls) if overfit_cls else 'None'}<br><br>
        <b>Recommended Fixes:</b><br>
        &nbsp;&nbsp;• Increase min_samples_leaf to reduce tree complexity<br>
        &nbsp;&nbsp;• Add L2 regularisation (for linear models)<br>
        &nbsp;&nbsp;• Collect more training data (more IPL seasons)<br>
        &nbsp;&nbsp;• Use cross-validation instead of single train/test split
        </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ No significant overfitting detected across all models.")

    st.markdown("---")

    # ---- Key takeaways ----
    st.markdown("### 🏁 Key Business Takeaways")
    st.markdown("""
    <div class="biz-card">
    <b>1. Ensemble models dominate:</b> RF, GBM, XGBoost consistently outperform
    linear and single-tree models — use ensembles for production.<br><br>
    <b>2. RandomizedSearchCV is cost-efficient:</b> Near-identical tuning gains
    at 3× lower compute vs GridSearchCV — ideal for cloud deployments.<br><br>
    <b>3. KNN & SVM scale poorly:</b> Competitive accuracy but slow at large N —
    best for offline analytics, not live APIs.<br><br>
    <b>4. AdaBoost is a solid middle ground:</b> Good accuracy, fast inference,
    minimal memory — suitable for resource-constrained mobile apps.<br><br>
    <b>5. Win probability > score for fan engagement:</b> Users respond more to
    live win% meters (e.g., "CSK 73% likely to win") than projected scores.<br><br>
    <b>6. Retrain every IPL season:</b> Team composition, player form, and venue
    conditions evolve — stale models lose predictive accuracy over time.
    </div>""", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    f"<center>🏏 IPL Predictor &nbsp;|&nbsp; Ball-by-Ball Dataset 2008–2025 &nbsp;|&nbsp; "
    f"{model_count} Models per Task &nbsp;|&nbsp; RandomizedSearchCV Tuning &nbsp;|&nbsp; "
    f"FAST-NUCES CF Campus</center>",
    unsafe_allow_html=True
)
