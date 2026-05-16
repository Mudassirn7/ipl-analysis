# =========================================================
# IPL SCORE & WIN PREDICTOR 
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
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IPL Score & Win Predictor",
    page_icon="🏏",
    layout="wide"
)

st.markdown("""
<style>

/* ── backgrounds ── */
html, body, .stApp, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stVerticalBlock"] {
    background-color: #0d1117 !important;
}

/* ── all base text bright white ── */
html, body, .stApp, .stMarkdown, .stText,
p, span, div, li, td, th, label,
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"] {
    color: #e6edf3 !important;
}

/* ── headings ── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown h4 {
    color: #f0f0f0 !important;
}

/* ── input labels ── */
.stSelectbox label,
.stNumberInput label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #b0bac4 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* ── select box ── */
.stSelectbox > div > div,
.stSelectbox > div > div > div {
    background-color: #161b22 !important;
    color: #f0f0f0 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── select box dropdown options ── */
[data-baseweb="popover"] {
    background-color: #1c2128 !important;
    border: 1px solid #30363d !important;
}

[role="listbox"] {
    background-color: #1c2128 !important;
}

[role="option"] {
    background-color: #1c2128 !important;
    color: #e6edf3 !important;
    padding: 10px 12px !important;
    border-bottom: 1px solid #21262d !important;
}

[role="option"]:hover {
    background-color: #21262d !important;
    color: #f0f0f0 !important;
}

[role="option"][aria-selected="true"] {
    background-color: #ff6b00 !important;
    color: #ffffff !important;
}

/* ── number input ── */
.stNumberInput > div > div > input {
    background-color: #161b22 !important;
    color: #f0f0f0 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    text-align: center !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

.stNumberInput button {
    background-color: #21262d !important;
    color: #ff6b00 !important;
    border: none !important;
}

/* ── predict button ── */
.stButton > button {
    width: 100% !important;
    background: #ff6b00 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

.stButton > button:hover { background: #e05a00 !important; }
.stButton > button p     { color: #ffffff !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #161b22 !important;
    border-radius: 10px !important;
    padding: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #8b949e !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
}

.stTabs [data-baseweb="tab"] p {
    color: #8b949e !important;
}

.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] p {
    background-color: #ff6b00 !important;
    color: #ffffff !important;
}

/* ── dataframe ── */
.stDataFrame, .stDataFrame *,
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] *,
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] span,
[data-testid="stDataFrame"] div,
[data-testid="stDataFrame"] p,
.glideDataEditor *,
.dvn-scroller * {
    background-color: #161b22 !important;
    color: #f0f0f0 !important;
}

/* ── metrics ── */
[data-testid="metric-container"] {
    background-color: #161b22 !important;
    border: 0.5px solid #30363d !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #b0bac4 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricValue"] div {
    color: #ff6b00 !important;
}

/* ── alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
}

.stSuccess, [data-testid="stAlert"][kind="success"] {
    background-color: #0d2318 !important;
    border: 0.5px solid #238636 !important;
    color: #3fb950 !important;
}

.stInfo, [data-testid="stAlert"][kind="info"] {
    background-color: #0d1f33 !important;
    border: 0.5px solid #1f6feb !important;
    color: #79c0ff !important;
}

.stSuccess p, .stInfo p,
[data-testid="stAlert"] p { color: inherit !important; }

/* ── caption / small text ── */
.stCaption,
[data-testid="stCaptionContainer"] p {
    color: #8b949e !important;
    font-size: 12px !important;
}

hr { border-color: #21262d !important; }

/* ── progress bar ── */
.stProgress > div > div { background-color: #ff6b00 !important; }

/* ── custom HTML classes ── */
.app-header {
    background: #161b22;
    border: 0.5px solid #30363d;
    border-radius: 12px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
}

.app-header h1 {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #f0f0f0 !important;
    margin: 0 !important;
}

.app-header p {
    font-size: 12px !important;
    color: #8b949e !important;
    margin: 0 !important;
}

.vs-divider {
    text-align: center;
    font-size: 22px;
    font-weight: 800;
    color: #ff6b00 !important;
    padding-top: 22px;
}

.result-box {
    background: #0d1117;
    border: 1.5px solid #ff6b00;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-top: 16px;
}

.result-box .rlabel {
    font-size: 11px !important;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}

.result-box .rscore {
    font-size: 56px !important;
    font-weight: 800 !important;
    color: #ff6b00 !important;
    line-height: 1.1;
}

.result-box .rwinner {
    font-size: 30px !important;
    font-weight: 800 !important;
    color: #3fb950 !important;
    line-height: 1.2;
}

.result-box .rconf {
    font-size: 13px !important;
    color: #8b949e !important;
    margin-top: 6px;
}

.section-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
    margin-top: 14px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div style="width:40px;height:40px;background:#ff6b00;border-radius:9px;
                display:flex;align-items:center;justify-content:center;
                font-size:22px;flex-shrink:0;">🏏</div>
    <div>
        <h1>IPL Score &amp; Win Predictor</h1>
        <p>ML-powered match analysis · 2008–2025 data</p>
    </div>
</div>
""", unsafe_allow_html=True)

CSV_FILE = "IPL.csv"

if not os.path.exists(CSV_FILE):
    try:
        os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
        os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "chaitu20/ipl-dataset2008-2025", path=".", unzip=True
        )
    except Exception as e:
        st.error(f"Dataset download failed: {e}")
        st.stop()

if not os.path.exists(CSV_FILE):
    st.error("IPL.csv not found.")
    st.stop()

raw_df = pd.read_csv(CSV_FILE)

TEAM_NAME_MAPPING = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Gujarat Lions": "Gujarat Titans"
}

VENUE_MAPPING = {
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    "M Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "MA Chidambaram Stadium": "M. A. Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk": "M. A. Chidambaram Stadium",
    "Punjab Cricket Association Stadium": "PCA Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "PCA Stadium"
}

raw_df["batting_team"] = raw_df["batting_team"].replace(TEAM_NAME_MAPPING)
raw_df["bowling_team"] = raw_df["bowling_team"].replace(TEAM_NAME_MAPPING)
raw_df["venue"] = raw_df["venue"].replace(VENUE_MAPPING)

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

IPL_VENUES = sorted(df["venue"].dropna().unique().tolist())

df["overs_completed"] = df["over"] + (df["ball"] / 6)
df["current_run_rate"] = df["team_runs"] / df["overs_completed"].replace(0, 0.1)
df["final_score"] = df.groupby(["match_id", "innings"])["team_runs"].transform("max")

TEAM_ENC  = {team: idx for idx, team in enumerate(IPL_TEAMS)}
VENUE_ENC = {venue: idx for idx, venue in enumerate(IPL_VENUES)}

score_df = df[
    (df["overs_completed"] >= 6) & (df["overs_completed"] <= 16)
][["batting_team","bowling_team","venue","team_runs","team_wicket",
   "overs_completed","current_run_rate","final_score"]].dropna().copy()

score_df.columns = ["batting_team","bowling_team","venue",
                    "current_runs","wickets","overs","crr","final_score"]

score_df["batting_team"] = score_df["batting_team"].map(TEAM_ENC)
score_df["bowling_team"] = score_df["bowling_team"].map(TEAM_ENC)
score_df["venue"]        = score_df["venue"].map(VENUE_ENC)

innings1 = df[df["innings"] == 1]
targets  = innings1.groupby("match_id")["team_runs"].max().reset_index()
targets.columns = ["match_id", "target"]

match_results = df[df["innings"] == 2].copy().merge(targets, on="match_id")
match_end = match_results.sort_values(["match_id","over","ball"]).groupby("match_id").last().reset_index()
match_end["won_chase"] = (match_end["team_runs"] >= match_end["target"]).astype(int)
winner_lookup = match_end[["match_id","won_chase"]]

win_df = df[
    (df["innings"] == 2) & (df["overs_completed"] >= 6) & (df["overs_completed"] <= 18)
].copy().merge(targets, on="match_id").merge(winner_lookup, on="match_id")

win_df["runs_needed"]       = win_df["target"] - win_df["team_runs"]
win_df["balls_left"]        = (120 - (win_df["over"]*6 + win_df["ball"])).replace(0, 1)
win_df["required_run_rate"] = win_df["runs_needed"] * 6 / win_df["balls_left"]
win_df["pct_target_done"]   = win_df["team_runs"] / win_df["target"].replace(0, 1)
win_df["pct_overs_done"]    = win_df["overs_completed"] / 20

win_df = win_df[[
    "batting_team","bowling_team","venue","target","team_wicket",
    "overs_completed","current_run_rate","required_run_rate",
    "pct_target_done","pct_overs_done","won_chase"
]].dropna()

win_df["batting_team"] = win_df["batting_team"].map(TEAM_ENC)
win_df["bowling_team"] = win_df["bowling_team"].map(TEAM_ENC)
win_df["venue"]        = win_df["venue"].map(VENUE_ENC)

@st.cache_resource
def train_models():
    Xr = score_df[["batting_team","bowling_team","venue",
                   "current_runs","wickets","overs","crr"]]
    yr = score_df["final_score"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    REG_MODELS = {
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=6, min_samples_leaf=15, random_state=42),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=42),
        "XGBoost":           XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    for m in REG_MODELS.values(): m.fit(Xr_train, yr_train)

    Xc = win_df[["batting_team","bowling_team","venue","target","team_wicket",
                 "overs_completed","current_run_rate","required_run_rate",
                 "pct_target_done","pct_overs_done"]]
    yc = win_df["won_chase"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

    CLS_MODELS = {
        "Random Forest":     RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=15, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":     DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
        "Extra Trees":       ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42),
        "XGBoost":           XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=42)
    }
    for m in CLS_MODELS.values(): m.fit(Xc_train, yc_train)

    reg_metrics = {}
    for name, m in REG_MODELS.items():
        tp = m.predict(Xr_train); te = m.predict(Xr_test)
        reg_metrics[name] = {
            "Train R²": round(r2_score(yr_train, tp), 4),
            "Test R²":  round(r2_score(yr_test, te), 4),
            "RMSE":     round(np.sqrt(mean_squared_error(yr_test, te)), 2),
            "MAE":      round(mean_absolute_error(yr_test, te), 2)
        }

    cls_metrics = {}
    for name, m in CLS_MODELS.items():
        tp = m.predict(Xc_train); te = m.predict(Xc_test)
        cls_metrics[name] = {
            "Train Accuracy": round(accuracy_score(yc_train, tp)*100, 2),
            "Test Accuracy":  round(accuracy_score(yc_test, te)*100, 2),
            "Precision":      round(precision_score(yc_test, te)*100, 2),
            "Recall":         round(recall_score(yc_test, te)*100, 2),
            "F1 Score":       round(f1_score(yc_test, te)*100, 2)
        }

    return (REG_MODELS, CLS_MODELS, reg_metrics, cls_metrics,
            Xr_train, Xr_test, yr_train, yr_test,
            Xc_train, Xc_test, yc_train, yc_test)


with st.spinner("Training models..."):
    (REG_MODELS, CLS_MODELS, REG_METRICS, CLS_METRICS,
     Xr_train, Xr_test, yr_train, yr_test,
     Xc_train, Xc_test, yc_train, yc_test) = train_models()

st.success("Models trained successfully ✅")

if "score_result" not in st.session_state:
    st.session_state.score_result = None
if "win_result" not in st.session_state:
    st.session_state.win_result = None

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Score Predictor",
    "🏆 Win Predictor",
    "📊 Model Report",
    "🔍 Data Analysis",
    "📉 Visual Analytics"
])

def dark_fig(w=6, h=3):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax

# ── TAB 1 ──
with tab1:
    st.markdown('<div class="section-label">Match Setup</div>', unsafe_allow_html=True)
    bat_col, vs_col, bowl_col = st.columns([5, 1, 5])
    with bat_col:
        batting_team = st.selectbox("🏏 Batting Team", IPL_TEAMS, key="bat1")
    with vs_col:
        st.markdown('<div class="vs-divider">VS</div>', unsafe_allow_html=True)
    with bowl_col:
        bowl_options = [x for x in IPL_TEAMS if x != batting_team]
        bowling_team = st.selectbox("🎯 Bowling Team", bowl_options, key="bowl1")

    st.markdown('<div class="section-label">Live Match State</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: current_runs = st.number_input("Current Runs", min_value=0, max_value=300, value=80, step=1, key="runs1")
    with c2: wickets = st.number_input("Wickets", min_value=0, max_value=9, value=2, step=1, key="wk1")
    with c3: over_num = st.number_input("Overs", min_value=0, max_value=19, value=10, step=1, key="ov1")
    with c4: ball_num = st.number_input("Balls", min_value=0, max_value=5, value=0, step=1, key="bl1")

    overs = round(over_num + ball_num / 6, 2)
    st.caption(f"Overs completed: **{overs}**")

    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1: venue = st.selectbox("Venue", IPL_VENUES, key="venue1")
    with s2: model_name = st.selectbox("ML Model", list(REG_MODELS.keys()), key="model1")

    if st.button("Predict Final Score 🏏", key="btn1"):
        crr = current_runs / max(overs, 0.1)
        X = np.array([[TEAM_ENC[batting_team], TEAM_ENC[bowling_team],
                       VENUE_ENC[venue], current_runs, wickets, overs, crr]])
        st.session_state.score_result = {
            "prediction": int(REG_MODELS[model_name].predict(X)[0]),
            "batting_team": batting_team,
            "model_name": model_name
        }

    if st.session_state.score_result:
        r = st.session_state.score_result
        st.markdown(f"""
        <div class="result-box">
            <div class="rlabel">Predicted Final Score</div>
            <div class="rscore">{r['prediction']}</div>
            <div class="rconf">{r['batting_team']} batting · {r['model_name']}</div>
        </div>""", unsafe_allow_html=True)

# ── TAB 2 ──
with tab2:
    st.markdown('<div class="section-label">Match Setup</div>', unsafe_allow_html=True)
    c1, vs_c, c2 = st.columns([5, 1, 5])
    with c1: chasing_team = st.selectbox("🏃 Chasing Team", IPL_TEAMS, key="ct")
    with vs_c: st.markdown('<div class="vs-divider">VS</div>', unsafe_allow_html=True)
    with c2: defending_team = st.selectbox("🛡️ Defending Team", [x for x in IPL_TEAMS if x != chasing_team], key="dt")

    st.markdown('<div class="section-label">Chase Situation</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1: target = st.number_input("Target", min_value=50, max_value=300, value=180, step=1, key="tgt")
    with r2: current_score = st.number_input("Current Score", min_value=0, max_value=300, value=90, step=1, key="cs")

    c1, c2, c3 = st.columns(3)
    with c1: wickets2 = st.number_input("Wickets Fallen", min_value=0, max_value=9, value=3, step=1, key="wk2")
    with c2: over2 = st.number_input("Overs", min_value=0, max_value=19, value=10, step=1, key="ov2")
    with c3: ball2 = st.number_input("Balls", min_value=0, max_value=5, value=0, step=1, key="bl2")

    overs2 = round(over2 + ball2 / 6, 2)
    runs_needed = target - current_score
    balls_left  = max(120 - (over2*6 + ball2), 1)
    rrr = round(runs_needed * 6 / balls_left, 2)
    st.caption(f"Overs completed: **{overs2}** · Runs needed: **{runs_needed}** · Required RR: **{rrr}**")

    s1, s2 = st.columns(2)
    with s1: venue2 = st.selectbox("Venue", IPL_VENUES, key="venue2")
    with s2: model_name2 = st.selectbox("ML Model", list(CLS_MODELS.keys()), key="model2")

    if st.button("Predict Winner 🏆", key="btn2"):
        crr2  = current_score / max(overs2, 0.1)
        pct_d = current_score / max(target, 1)
        pct_o = overs2 / 20
        X2 = np.array([[TEAM_ENC[chasing_team], TEAM_ENC[defending_team],
                        VENUE_ENC[venue2], target, wickets2, overs2,
                        crr2, rrr, pct_d, pct_o]])
        model = CLS_MODELS[model_name2]
        pred  = model.predict(X2)[0]
        prob  = model.predict_proba(X2)[0]
        st.session_state.win_result = {
            "winner":     chasing_team if pred == 1 else defending_team,
            "conf":       round(max(prob) * 100, 2),
            "model_name": model_name2
        }

    if st.session_state.win_result:
        r = st.session_state.win_result
        st.markdown(f"""
        <div class="result-box">
            <div class="rlabel">Predicted Winner</div>
            <div class="rwinner">{r['winner']}</div>
            <div class="rconf">Confidence: {r['conf']}% · {r['model_name']}</div>
        </div>""", unsafe_allow_html=True)
        st.progress(int(r['conf']))

# ── TAB 3 ──
with tab3:
    def render_report(title, metrics_dict, highlight_key, better="max"):
        st.markdown(f'''<div style="font-size:11px;font-weight:700;color:#8b949e;text-transform:uppercase;
            letter-spacing:0.8px;margin:10px 0 8px">{title}</div>''', unsafe_allow_html=True)
        models = list(metrics_dict.keys())
        metric_keys = list(list(metrics_dict.values())[0].keys())
        scores = {m: metrics_dict[m][highlight_key] for m in models}
        best = max(scores, key=scores.get) if better == "max" else min(scores, key=scores.get)

        header_cols = st.columns([2.5] + [1]*len(metric_keys))
        header_cols[0].markdown('<div style="background:#0d1117;color:#ff6b00;font-size:11px;font-weight:700;padding:8px 10px;border-radius:6px;text-transform:uppercase">Model</div>', unsafe_allow_html=True)
        for i, mk in enumerate(metric_keys):
            header_cols[i+1].markdown(f'<div style="background:#0d1117;color:#ff6b00;font-size:11px;font-weight:700;padding:8px 4px;border-radius:6px;text-align:center;text-transform:uppercase">{mk}</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:2px;background:#ff6b00;border-radius:2px;margin-bottom:4px"></div>', unsafe_allow_html=True)

        for model in models:
            is_best = model == best
            bg   = "#0f2d0f" if is_best else "#161b22"
            bdr  = "1px solid #238636" if is_best else "1px solid #21262d"
            badge = '&nbsp;<span style="background:#ff6b00;color:#fff;font-size:9px;padding:2px 5px;border-radius:3px;font-weight:700">BEST</span>' if is_best else ""
            tc   = "#3fb950" if is_best else "#e6edf3"
            fw   = "700" if is_best else "400"
            row  = st.columns([2.5] + [1]*len(metric_keys))
            row[0].markdown(f'<div style="background:{bg};border:{bdr};border-radius:8px;padding:10px 12px;font-weight:700;color:#f0f0f0;font-size:13px">{model}{badge}</div>', unsafe_allow_html=True)
            for i, mk in enumerate(metric_keys):
                val = metrics_dict[model][mk]
                row[i+1].markdown(f'<div style="background:{bg};border:{bdr};border-radius:8px;padding:10px 4px;text-align:center;color:{tc};font-size:13px;font-weight:{fw}">{val}</div>', unsafe_allow_html=True)

    render_report("Regression Model Report", REG_METRICS, "Test R²")
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    render_report("Classification Model Report", CLS_METRICS, "Test Accuracy")

# =========================================================
# TAB 4 — DATA ANALYSIS (EXPANDED)
# =========================================================

with tab4:

    # ── Overview Metrics ──
    st.markdown('<div class="section-label">Dataset Overview</div>', unsafe_allow_html=True)
    total_matches  = df["match_id"].nunique()
    total_seasons  = df["season"].nunique() if "season" in df.columns else "2008-2025"
    total_teams    = len(IPL_TEAMS)
    total_venues   = len(IPL_VENUES)
    total_rows     = len(df)
    total_runs     = int(df["team_runs"].sum()) if "team_runs" in df.columns else "N/A"

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Total Matches",  total_matches)
    m2.metric("Seasons",        total_seasons)
    m3.metric("Teams",          total_teams)
    m4.metric("Venues",         total_venues)
    m5.metric("Total Rows",     f"{total_rows:,}")
    m6.metric("Total Runs",     f"{total_runs:,}" if isinstance(total_runs, int) else total_runs)

    st.markdown("---")

    # ── 1. Season-wise Matches ──
    st.markdown('<div class="section-label">1. Season-wise Matches Played</div>', unsafe_allow_html=True)
    if "season" in df.columns:
        season_matches = df.groupby("season")["match_id"].nunique().reset_index()
        season_matches.columns = ["Season","Matches"]
        fig, ax = dark_fig(7, 3)
        ax.bar(season_matches["Season"].astype(str), season_matches["Matches"],
               color="#ff6b00", width=0.6, edgecolor="#0d1117")
        ax.set_ylabel("Matches", color="#8b949e", fontsize=9)
        ax.set_title("Matches Per Season", color="#f0f0f0", fontsize=11, fontweight="bold")
        plt.xticks(rotation=45, fontsize=7, color="#8b949e", ha="right")
        plt.yticks(fontsize=7, color="#8b949e")
        plt.tight_layout()
        col_l,col_m,col_r = st.columns([1,4,1])
        with col_m: st.pyplot(fig)
        plt.close()
        peak = season_matches.loc[season_matches["Matches"].idxmax()]
        st.info(f"✅ **{peak['Season']}** had the most matches ({int(peak['Matches'])}) — likely due to playoff expansion or double-header scheduling.")

    st.markdown("---")

    # ── 2. Team-wise Total Runs ──
    st.markdown('<div class="section-label">2. Total Runs Scored by Each Team</div>', unsafe_allow_html=True)
    team_runs = df.groupby("batting_team")["team_runs"].sum().sort_values(ascending=False)
    COLORS_10 = ["#ff6b00","#e05a00","#3fb950","#58a6ff","#f0f0f0",
                 "#d29922","#bc8cff","#ff7b72","#39d353","#79c0ff"]
    fig, ax = dark_fig(7, 3.5)
    ax.bar(team_runs.index, team_runs.values,
           color=COLORS_10[:len(team_runs)], width=0.6, edgecolor="#0d1117")
    ax.set_ylabel("Total Runs", color="#8b949e", fontsize=9)
    ax.set_title("Total Runs by Batting Team (All Seasons)", color="#f0f0f0", fontsize=11, fontweight="bold")
    plt.xticks(rotation=40, fontsize=7, color="#8b949e", ha="right")
    plt.yticks(fontsize=7, color="#8b949e")
    plt.tight_layout()
    col_l,col_m,col_r = st.columns([1,4,1])
    with col_m: st.pyplot(fig)
    plt.close()
    top_team = team_runs.idxmax()
    st.info(f"✅ **{top_team}** has scored the most runs across all IPL seasons — consistent batting lineup and more seasons played.")

    st.markdown("---")

    # ── 3. Batting First vs Chasing Win % ──
    st.markdown('<div class="section-label">3. Batting First vs Chasing — Win Rate</div>', unsafe_allow_html=True)
    if "toss_decision" in df.columns and "toss_winner" in df.columns and "match_won_by" in df.columns:
        match_level = df.drop_duplicates("match_id")[["match_id","toss_winner","toss_decision","match_won_by"]].dropna()
        match_level["toss_winner_won"] = match_level["toss_winner"] == match_level["match_won_by"]
        bat_first_wins = match_level[match_level["toss_decision"]=="bat"]["toss_winner_won"].mean()*100
        chase_wins     = match_level[match_level["toss_decision"]=="field"]["toss_winner_won"].mean()*100
        bat_first_wins = round(bat_first_wins, 1)
        chase_wins     = round(chase_wins, 1)
        fig, ax = dark_fig(4, 3)
        bars = ax.bar(["Bat First","Chase/Field"], [bat_first_wins, chase_wins],
                      color=["#ff6b00","#3fb950"], width=0.4, edgecolor="#0d1117")
        ax.set_ylabel("Win % (Toss Winner)", color="#8b949e", fontsize=9)
        ax.set_title("Win Rate After Toss Decision", color="#f0f0f0", fontsize=11, fontweight="bold")
        ax.set_ylim(0,100)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{bar.get_height():.1f}%", ha="center", color="#f0f0f0", fontsize=9, fontweight="bold")
        plt.yticks(fontsize=7, color="#8b949e")
        plt.tight_layout()
        col_l,col_m,col_r = st.columns([2,3,2])
        with col_m: st.pyplot(fig)
        plt.close()
        leader = "fielding/chasing" if chase_wins > bat_first_wins else "batting first"
        st.info(f"✅ Toss winners who chose to **{leader}** won more matches — captains prefer a visible target in T20.")
    else:
        st.caption(f"Columns available: {[c for c in df.columns if 'toss' in c.lower() or 'won' in c.lower() or 'winner' in c.lower()]}")

    st.markdown("---")

    # ── 4. Top 10 Venues by Matches ──
    st.markdown('<div class="section-label">4. Top 10 Venues by Matches Hosted</div>', unsafe_allow_html=True)
    venue_matches = df.drop_duplicates("match_id").groupby("venue")["match_id"].count().sort_values(ascending=False).head(10)
    fig, ax = dark_fig(7, 3.5)
    ax.barh(venue_matches.index[::-1], venue_matches.values[::-1],
            color="#58a6ff", edgecolor="#0d1117", height=0.6)
    ax.set_xlabel("Matches Hosted", color="#8b949e", fontsize=9)
    ax.set_title("Top 10 Venues", color="#f0f0f0", fontsize=11, fontweight="bold")
    plt.xticks(fontsize=7, color="#8b949e")
    plt.yticks(fontsize=7, color="#8b949e")
    plt.tight_layout()
    col_l,col_m,col_r = st.columns([1,4,1])
    with col_m: st.pyplot(fig)
    plt.close()
    top_venue = venue_matches.idxmax()
    st.info(f"✅ **{top_venue}** has hosted the most IPL matches — a home-ground advantage hub for its team.")

    st.markdown("---")

    # ── 5. Average Score per Over (Run Rate Curve) ──
    st.markdown('<div class="section-label">5. Average Runs Scored Per Over</div>', unsafe_allow_html=True)
    if "over" in df.columns and "runs_off_bat" in df.columns:
        # runs scored in that specific over only (not cumulative)
        over_avg = df.groupby("over")["runs_batter"].sum() / df["match_id"].nunique()
        over_avg = over_avg[over_avg.index <= 19].reset_index()
        over_avg.columns = ["over","avg_runs"]
        over_avg["over_label"] = (over_avg["over"]+1).astype(int).astype(str)
        # Powerplay / Middle / Death zones
        colors = []
        for o in over_avg["over"]:
            if o < 6:   colors.append("#3fb950")   # powerplay green
            elif o < 15: colors.append("#ff6b00")  # middle orange
            else:        colors.append("#f85149")  # death red
        fig, ax = dark_fig(7, 3.2)
        bars = ax.bar(over_avg["over_label"], over_avg["avg_runs"],
                      color=colors, width=0.7, edgecolor="#0d1117")
        ax.set_xlabel("Over Number", color="#8b949e", fontsize=9)
        ax.set_ylabel("Avg Runs (per match)", color="#8b949e", fontsize=9)
        ax.set_title("Avg Runs Scored Per Over — All IPL Matches", color="#f0f0f0", fontsize=11, fontweight="bold")
        # Legend patches
        from matplotlib.patches import Patch
        legend = [Patch(color="#3fb950",label="Powerplay (1-6)"),
                  Patch(color="#ff6b00",label="Middle (7-15)"),
                  Patch(color="#f85149",label="Death (16-20)")]
        ax.legend(handles=legend, fontsize=7, facecolor="#161b22",
                  edgecolor="#30363d", labelcolor="#e6edf3")
        plt.xticks(fontsize=7, color="#8b949e")
        plt.yticks(fontsize=7, color="#8b949e")
        plt.tight_layout()
        col_l,col_m,col_r = st.columns([1,4,1])
        with col_m: st.pyplot(fig)
        plt.close()
        peak_over = int(over_avg.loc[over_avg["avg_runs"].idxmax(), "over"]) + 1
        st.info(f"✅ Over **{peak_over}** produces the most runs on average — death over batsmen target boundaries to maximise the final total.")
    elif "over" in df.columns and "runs_total" in df.columns:
        # fallback: use ball-level runs difference per over
        df_sorted = df.sort_values(["match_id","innings","over","ball"])
        ball_runs = df_sorted.groupby(["match_id","innings","over"])["team_runs"].last() -                     df_sorted.groupby(["match_id","innings","over"])["team_runs"].first()
        over_avg2 = ball_runs.groupby(level="over").mean()
        over_avg2 = over_avg2[over_avg2.index <= 19].reset_index()
        over_avg2.columns = ["over","avg_runs"]
        colors2 = ["#3fb950" if o<6 else "#ff6b00" if o<15 else "#f85149" for o in over_avg2["over"]]
        fig, ax = dark_fig(7, 3.2)
        ax.bar((over_avg2["over"]+1).astype(int).astype(str), over_avg2["avg_runs"],
               color=colors2, width=0.7, edgecolor="#0d1117")
        ax.set_xlabel("Over Number", color="#8b949e", fontsize=9)
        ax.set_ylabel("Avg Runs Per Over", color="#8b949e", fontsize=9)
        ax.set_title("Avg Runs Per Over — All IPL Matches", color="#f0f0f0", fontsize=11, fontweight="bold")
        from matplotlib.patches import Patch
        legend2 = [Patch(color="#3fb950",label="Powerplay (1-6)"),
                   Patch(color="#ff6b00",label="Middle (7-15)"),
                   Patch(color="#f85149",label="Death (16-20)")]
        ax.legend(handles=legend2, fontsize=7, facecolor="#161b22",
                  edgecolor="#30363d", labelcolor="#e6edf3")
        plt.xticks(fontsize=7, color="#8b949e")
        plt.yticks(fontsize=7, color="#8b949e")
        plt.tight_layout()
        col_l,col_m,col_r = st.columns([1,4,1])
        with col_m: st.pyplot(fig)
        plt.close()
        peak2 = int(over_avg2.loc[over_avg2["avg_runs"].idxmax(), "over"]) + 1
        st.info(f"✅ Over **{peak2}** produces the most runs on average — death over batsmen target boundaries to maximise the final total.")

    st.markdown("---")

    # ── 6. Toss Decision Distribution ──
    st.markdown('<div class="section-label">6. Toss Decision — Bat vs Field</div>', unsafe_allow_html=True)
    if "toss_decision" in df.columns:
        toss_counts = df.drop_duplicates("match_id")["toss_decision"].value_counts()
        toss_counts.index = [str(i).capitalize() for i in toss_counts.index]
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        wedge_colors = ["#ff6b00","#3fb950"]
        wedges, texts, autotexts = ax.pie(
            toss_counts.values,
            labels=toss_counts.index,
            autopct="%1.1f%%",
            colors=wedge_colors,
            startangle=90,
            wedgeprops={"edgecolor":"#0d1117","linewidth":2}
        )
        for t in texts: t.set_color("#8b949e"); t.set_fontsize(9)
        for at in autotexts: at.set_color("#f0f0f0"); at.set_fontsize(9); at.set_fontweight("bold")
        ax.set_title("Toss Decision Split", color="#f0f0f0", fontsize=11, fontweight="bold")
        plt.tight_layout()
        col_l,col_m,col_r = st.columns([2,3,2])
        with col_m: st.pyplot(fig)
        plt.close()
        dominant = toss_counts.idxmax()
        st.info(f"✅ Most captains choose to **{dominant}** after winning the toss — chasing a visible target is the preferred strategy in modern T20.")

    st.markdown("---")

    # ── 7. Wickets Distribution ──
    st.markdown('<div class="section-label">7. Wickets Distribution Per Over</div>', unsafe_allow_html=True)
    wicket_col = None
    for c in ["wicket_kind","is_wicket","wicket_type","player_dismissed","player_out","wickets"]:
        if c in df.columns:
            wicket_col = c
            break
    if wicket_col:
        if wicket_col == "is_wicket":
            wkt_df = df[df[wicket_col]==1]
        else:
            wkt_df = df[df[wicket_col].notna() & (df[wicket_col].astype(str).str.strip() != "") & (df[wicket_col].astype(str) != "nan") & (df[wicket_col] != 0)]
        wicket_by_over = wkt_df.groupby("over")[wicket_col].count().reset_index()
        wicket_by_over = wicket_by_over[wicket_by_over["over"] <= 19]
        wicket_by_over["over_label"] = (wicket_by_over["over"]+1).astype(int).astype(str)
        wkt_colors = ["#3fb950" if o<6 else "#ff6b00" if o<15 else "#f85149" for o in wicket_by_over["over"]]
        fig, ax = dark_fig(7, 3.2)
        ax.bar(wicket_by_over["over_label"], wicket_by_over[wicket_col],
               color=wkt_colors, width=0.7, edgecolor="#0d1117")
        ax.set_xlabel("Over Number", color="#8b949e", fontsize=9)
        ax.set_ylabel("Total Wickets Fallen", color="#8b949e", fontsize=9)
        ax.set_title("Wickets Fallen Per Over — All IPL Matches", color="#f0f0f0", fontsize=11, fontweight="bold")
        from matplotlib.patches import Patch
        legend_w = [Patch(color="#3fb950",label="Powerplay (1-6)"),
                    Patch(color="#ff6b00",label="Middle (7-15)"),
                    Patch(color="#f85149",label="Death (16-20)")]
        ax.legend(handles=legend_w, fontsize=7, facecolor="#161b22",
                  edgecolor="#30363d", labelcolor="#e6edf3")
        plt.xticks(fontsize=7, color="#8b949e")
        plt.yticks(fontsize=7, color="#8b949e")
        plt.tight_layout()
        col_l,col_m,col_r = st.columns([1,4,1])
        with col_m: st.pyplot(fig)
        plt.close()
        peak_w_over = int(wicket_by_over.loc[wicket_by_over[wicket_col].idxmax(), "over"]) + 1
        st.info(f"✅ Over **{peak_w_over}** sees the most wickets — powerplay swing, spin in middle overs, and death over gambles all contribute to dismissals.")
    else:
        st.caption(f"Wicket column not found. Available: {list(df.columns)}")

    st.markdown("---")

    # ── 8. Season-wise Avg Score ──
    st.markdown('<div class="section-label">8. Season-wise Average Innings Score</div>', unsafe_allow_html=True)
    if "season" in df.columns:
        # Get final score per innings per match (max of cumulative team_runs)
        inn_scores = df.groupby(["season","match_id","innings"])["team_runs"].max().reset_index()
        inn_scores.columns = ["Season","match_id","innings","Final Score"]

        season_stats = inn_scores.groupby("Season")["Final Score"].agg(
            Avg="mean", High="max", Low="min"
        ).reset_index()
        season_stats["Avg"] = season_stats["Avg"].round(1)

        x = range(len(season_stats))
        fig, ax = dark_fig(8, 3.5)

        # Bar for avg score
        bars = ax.bar(x, season_stats["Avg"],
                      color="#bc8cff", width=0.6, edgecolor="#0d1117", label="Avg Score", zorder=2)

        # Error bars showing min-max range
        yerr_low  = season_stats["Avg"] - season_stats["Low"]
        yerr_high = season_stats["High"] - season_stats["Avg"]
        ax.errorbar(x, season_stats["Avg"],
                    yerr=[yerr_low, yerr_high],
                    fmt="none", color="#f0f0f0", capsize=4, linewidth=1.2,
                    label="Min-Max Range", zorder=3)

        # Value labels on bars
        for i, (bar, avg) in enumerate(zip(bars, season_stats["Avg"])):
            ax.text(bar.get_x()+bar.get_width()/2, avg+1,
                    f"{avg:.0f}", ha="center", va="bottom",
                    color="#e6edf3", fontsize=6.5, fontweight="bold")

        ax.set_xticks(list(x))
        ax.set_xticklabels(season_stats["Season"].astype(str),
                           rotation=45, fontsize=7, color="#8b949e", ha="right")
        ax.set_ylabel("Innings Score (Runs)", color="#8b949e", fontsize=9)
        ax.set_title("Average Innings Score Per Season  |  Error bars = Min & Max", color="#f0f0f0", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        plt.yticks(fontsize=7, color="#8b949e")
        plt.tight_layout()
        col_l,col_m,col_r = st.columns([1,4,1])
        with col_m: st.pyplot(fig)
        plt.close()

        peak_s  = season_stats.loc[season_stats["Avg"].idxmax(), "Season"]
        peak_hi = season_stats.loc[season_stats["High"].idxmax(), "High"]
        st.info(f"✅ **{peak_s}** had the highest average innings score. Highest ever innings: **{int(peak_hi)}** runs — T20 batting has grown more aggressive with better pitches and shorter boundaries.")

# ── TAB 5 ──
with tab5:
    st.subheader("Regression Model Comparison")
    reg_df = pd.DataFrame(REG_METRICS).T.reset_index()
    reg_df.columns = ["Model","Train R2","Test R2","RMSE","MAE"]

    fig1, ax1 = dark_fig(6, 3)
    ax1.bar(reg_df["Model"], reg_df["Test R2"], color="#ff6b00", width=0.5)
    ax1.set_ylabel("R² Score", color="#8b949e")
    ax1.set_title("Test R² by Model", color="#f0f0f0")
    plt.xticks(rotation=15, fontsize=8, color="#8b949e")
    plt.yticks(color="#8b949e")
    plt.tight_layout()
    col_a, col_b, col_c = st.columns([1,2,1])
    with col_b: st.pyplot(fig1)
    plt.close()
    best_reg = reg_df.loc[reg_df["Test R2"].idxmax(), "Model"]
    st.info(f"✅ **{best_reg}** has the highest Test R² — most accurate score predictor.")

    st.markdown("---")
    st.subheader("Classification Model Accuracy")
    cls_df = pd.DataFrame(CLS_METRICS).T.reset_index()
    cls_df.columns = ["Model","Train Accuracy","Test Accuracy","Precision","Recall","F1 Score"]

    fig2, ax2 = dark_fig(6, 3)
    ax2.bar(cls_df["Model"], cls_df["Test Accuracy"], color="#ff6b00", width=0.5)
    ax2.set_ylabel("Accuracy %", color="#8b949e")
    ax2.set_title("Test Accuracy by Model", color="#f0f0f0")
    plt.xticks(rotation=15, fontsize=8, color="#8b949e")
    plt.yticks(color="#8b949e")
    plt.tight_layout()
    col_a, col_b, col_c = st.columns([1,2,1])
    with col_b: st.pyplot(fig2)
    plt.close()
    best_cls = cls_df.loc[cls_df["Test Accuracy"].idxmax(), "Model"]
    st.info(f"✅ **{best_cls}** achieved the highest test accuracy for win prediction.")

    st.markdown("---")
    st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
    cm_model_name = st.selectbox("Select Model", list(CLS_MODELS.keys()), key="cm_model")
    preds = CLS_MODELS[cm_model_name].predict(Xc_test)
    cm    = confusion_matrix(yc_test, preds)
    tn, fp, fn, tp_v = cm.ravel()
    total = tn + fp + fn + tp_v
    acc   = round((tn + tp_v) / total * 100, 1)

    col_matrix, col_interp = st.columns([1, 1])

    with col_matrix:
        fig3, ax3 = plt.subplots(figsize=(2.8, 2.2))
        fig3.patch.set_facecolor("#0d1117")
        ax3.set_facecolor("#161b22")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax3,
                    linewidths=1, linecolor="#0d1117",
                    annot_kws={"color":"#111111", "fontsize":11, "fontweight":"bold"},
                    cbar=False)
        ax3.set_xlabel("Predicted", color="#8b949e", fontsize=8)
        ax3.set_ylabel("Actual",    color="#8b949e", fontsize=8)
        ax3.set_xticklabels(["Loss","Win"], color="#8b949e", fontsize=7)
        ax3.set_yticklabels(["Loss","Win"], color="#8b949e", fontsize=7, rotation=0)
        ax3.tick_params(colors="#8b949e")
        ax3.set_title(cm_model_name, color="#f0f0f0", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_interp:
        correct = tp_v + tn
        incorrect = fp + fn
        correct_pct = round((correct / total * 100), 1)
        
        card_html = (
            '<div style="background:linear-gradient(135deg, #161b22 0%, #1a202a 100%);border:1.5px solid #30363d;border-radius:12px;padding:20px;margin-top:8px;box-shadow:0 4px 12px rgba(0,0,0,0.3);">'
            '<div style="font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;font-weight:600;">📊 Prediction Breakdown</div>'
            
            # Correct Predictions Row
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">'
            '<div style="background:#0f2d0f;border:1.5px solid #238636;border-radius:10px;padding:14px;text-align:center;box-shadow:0 2px 6px rgba(56, 185, 80, 0.1);">'
            f'<div style="color:#3fb950;font-size:28px;font-weight:800;line-height:1;">{tp_v}</div>'
            '<div style="color:#3fb950;font-size:9px;text-transform:uppercase;letter-spacing:0.5px;margin-top:6px;font-weight:600;">✓ True Wins</div>'
            '</div>'
            '<div style="background:#0f2d0f;border:1.5px solid #238636;border-radius:10px;padding:14px;text-align:center;box-shadow:0 2px 6px rgba(56, 185, 80, 0.1);">'
            f'<div style="color:#3fb950;font-size:28px;font-weight:800;line-height:1;">{tn}</div>'
            '<div style="color:#3fb950;font-size:9px;text-transform:uppercase;letter-spacing:0.5px;margin-top:6px;font-weight:600;">✓ True Losses</div>'
            '</div>'
            '</div>'
            
            # Incorrect Predictions Row
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;">'
            '<div style="background:#2d0f0f;border:1.5px solid #f85149;border-radius:10px;padding:14px;text-align:center;box-shadow:0 2px 6px rgba(248, 81, 73, 0.1);">'
            f'<div style="color:#f85149;font-size:28px;font-weight:800;line-height:1;">{fp}</div>'
            '<div style="color:#f85149;font-size:9px;text-transform:uppercase;letter-spacing:0.5px;margin-top:6px;font-weight:600;">✗ False Wins</div>'
            '</div>'
            '<div style="background:#2d0f0f;border:1.5px solid #f85149;border-radius:10px;padding:14px;text-align:center;box-shadow:0 2px 6px rgba(248, 81, 73, 0.1);">'
            f'<div style="color:#f85149;font-size:28px;font-weight:800;line-height:1;">{fn}</div>'
            '<div style="color:#f85149;font-size:9px;text-transform:uppercase;letter-spacing:0.5px;margin-top:6px;font-weight:600;">✗ Missed Wins</div>'
            '</div>'
            '</div>'
            
            # Accuracy Bar
            '<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:12px;margin-bottom:12px;">'
            '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            '<span style="color:#8b949e;font-size:10px;font-weight:600;text-transform:uppercase;">Accuracy</span>'
            f'<span style="color:#ff6b00;font-size:11px;font-weight:700;">{acc}%</span>'
            '</div>'
            '<div style="background:#21262d;border-radius:6px;height:8px;overflow:hidden;border:1px solid #30363d;">'
            f'<div style="background:linear-gradient(90deg, #ff6b00 0%, #e05a00 100%);height:100%;width:{acc}%;border-radius:6px;"></div>'
            '</div>'
            '</div>'
            
            # Summary Stats
            '<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:12px;">'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;text-align:center;font-size:10px;">'
            f'<div><span style="color:#8b949e;">Total Predictions</span><div style="color:#f0f0f0;font-size:14px;font-weight:700;margin-top:4px;">{total}</div></div>'
            f'<div><span style="color:#8b949e;">Correct</span><div style="color:#3fb950;font-size:14px;font-weight:700;margin-top:4px;">{correct_pct}%</div></div>'
            '</div>'
            '</div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Feature Importance")
    feat_model = st.selectbox("Select Model", ["Random Forest","Extra Trees","XGBoost"], key="fm")
    imp_df = pd.DataFrame({
        "Feature":    Xr_train.columns,
        "Importance": REG_MODELS[feat_model].feature_importances_
    }).sort_values("Importance", ascending=False)

    fig4, ax4 = dark_fig(6, 3)
    ax4.bar(imp_df["Feature"], imp_df["Importance"], color="#ff6b00", width=0.5)
    ax4.set_ylabel("Importance", color="#8b949e")
    ax4.set_title("Feature Importance", color="#f0f0f0")
    plt.xticks(rotation=40, fontsize=8, color="#8b949e", ha="right")
    plt.yticks(color="#8b949e")
    plt.tight_layout()
    col_a, col_b, col_c = st.columns([1,2,1])
    with col_b: st.pyplot(fig4)
    plt.close()
    st.info(f"✅ **{imp_df.iloc[0]['Feature']}** is the most important feature.")

    st.markdown("---")
    st.subheader("Actual vs Predicted (Random Forest)")
    pred_scores = REG_MODELS["Random Forest"].predict(Xr_test)

    fig5, ax5 = dark_fig(4, 3.5)
    ax5.scatter(yr_test, pred_scores, alpha=0.4, s=10, color="#ff6b00")
    mx = max(yr_test.max(), pred_scores.max())
    ax5.plot([0, mx], [0, mx], "--", color="#3fb950", linewidth=1)
    ax5.set_xlabel("Actual Score",    color="#8b949e")
    ax5.set_ylabel("Predicted Score", color="#8b949e")
    ax5.set_title("Actual vs Predicted", color="#f0f0f0")
    plt.tight_layout()
    col_a, col_b, col_c = st.columns([1,2,1])
    with col_b: st.pyplot(fig5)
    plt.close()
    r2 = round(r2_score(yr_test, pred_scores), 4)
    st.info(f"✅ Random Forest R² = **{r2}**")

    st.markdown("---")
    st.subheader("Residual Plot")
    residuals = yr_test - pred_scores

    fig6, ax6 = dark_fig(6, 3)
    ax6.scatter(pred_scores, residuals, alpha=0.4, s=10, color="#ff6b00")
    ax6.axhline(y=0, linestyle="--", color="#f85149", linewidth=1)
    ax6.set_xlabel("Predicted",  color="#8b949e")
    ax6.set_ylabel("Residuals",  color="#8b949e")
    ax6.set_title("Residual Plot", color="#f0f0f0")
    plt.tight_layout()
    col_a, col_b, col_c = st.columns([1,2,1])
    with col_b: st.pyplot(fig6)
    plt.close()
    mean_res = round(float(np.mean(residuals)), 2)
    st.info(f"✅ Mean residual = **{mean_res}** — no major prediction bias.")

st.markdown("---")
st.markdown(
    "<center style='color:#3d4451;font-size:12px;'>🏏 IPL Score & Win Predictor · Final Year Project · ML System</center>",
    unsafe_allow_html=True
)
