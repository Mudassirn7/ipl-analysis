import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
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
.best-badge { background:#ff6b00; color:white !important; padding:3px 10px; border-radius:20px; font-size:13px; font-weight:bold; }
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

TEAM_ENC  = {team: idx for idx, team in enumerate(IPL_TEAMS)}
VENUE_ENC = {venue: idx for idx, venue in enumerate(IPL_VENUES)}

# =========================================================
# SCORE DATASET
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
score_df["venue"]        = score_df["venue"].map(VENUE_ENC)
score_df = score_df.dropna()

# =========================================================
# WIN DATASET
# =========================================================

innings1 = df[df["innings"] == 1]
targets  = innings1.groupby("match_id")["team_runs"].max().reset_index()
targets.columns = ["match_id", "target"]

match_results = df[df["innings"] == 2].copy()
match_results = match_results.merge(targets, on="match_id")

match_end = match_results.sort_values(
    ["match_id", "over", "ball"]
).groupby("match_id").last().reset_index()

match_end["won_chase"] = (match_end["team_runs"] >= match_end["target"]).astype(int)
winner_lookup = match_end[["match_id", "won_chase"]]

win_df = df[
    (df["innings"] == 2) &
    (df["overs_completed"] >= 6) &
    (df["overs_completed"] <= 18)
].copy()

win_df = win_df.merge(targets, on="match_id")
win_df = win_df.merge(winner_lookup, on="match_id")

win_df["runs_needed"]       = win_df["target"] - win_df["team_runs"]
win_df["balls_left"]        = 120 - ((win_df["over"] * 6) + win_df["ball"])
win_df["balls_left"]        = win_df["balls_left"].replace(0, 1)
win_df["required_run_rate"] = win_df["runs_needed"] * 6 / win_df["balls_left"]
win_df["pct_target_done"]   = win_df["team_runs"] / win_df["target"].replace(0, 1)
win_df["pct_overs_done"]    = win_df["overs_completed"] / 20

win_df = win_df[[
    "batting_team", "bowling_team", "venue",
    "target", "team_wicket", "overs_completed",
    "current_run_rate", "required_run_rate",
    "pct_target_done", "pct_overs_done",
    "won_chase"
]].dropna().copy()

win_df["batting_team"] = win_df["batting_team"].map(TEAM_ENC)
win_df["bowling_team"] = win_df["bowling_team"].map(TEAM_ENC)
win_df["venue"]        = win_df["venue"].map(VENUE_ENC)
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
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=6, min_samples_leaf=15, random_state=42)
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
        "Random Forest":      RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=15, random_state=42),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
        "Logistic Regression":LogisticRegression(max_iter=1000),
        "Decision Tree":      DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
    }
    for model in CLS_MODELS.values():
        model.fit(Xc_train, yc_train)

    # --- METRICS ---
    reg_metrics = {}
    for name, model in REG_MODELS.items():
        train_pred = model.predict(Xr_train)
        test_pred  = model.predict(Xr_test)
        train_r2   = r2_score(yr_train, train_pred)
        test_r2    = r2_score(yr_test,  test_pred)
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
# TABS — added new "📈 Model Comparison" tab
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Score Predictor",
    "🏆 Win Predictor",
    "📊 Model Report",
    "📈 Model Comparison",
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
        bowling_team = st.selectbox("Bowling Team", [x for x in IPL_TEAMS if x != batting_team], key="bowl1")
        venue        = st.selectbox("Venue", IPL_VENUES, key="venue1")
        model_name   = st.selectbox("ML Model", list(REG_MODELS.keys()), key="model1")

    with col2:
        current_runs = st.number_input("Current Runs", 0, 300, 80)
        wickets      = st.slider("Wickets", 0, 9, 2, key="wk1")
        over_num     = st.slider("Overs", 0, 19, 10, key="ov1")
        ball_num     = st.slider("Balls", 0, 5, 0, key="ball1")
        overs        = round(over_num + (ball_num / 6), 2)
        st.caption(f"{over_num}.{ball_num} overs")

    if st.button("Predict Final Score", key="btn1"):
        crr = current_runs / max(overs, 0.1)
        X   = np.array([[TEAM_ENC[batting_team], TEAM_ENC[bowling_team], VENUE_ENC[venue], current_runs, wickets, overs, crr]])
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
        defending_team = st.selectbox("Defending Team", [x for x in IPL_TEAMS if x != chasing_team], key="dt")
        venue2         = st.selectbox("Venue", IPL_VENUES, key="venue2")
        model_name2    = st.selectbox("ML Model", list(CLS_MODELS.keys()), key="model2")

    with col2:
        target        = st.number_input("Target", 50, 300, 180)
        current_score = st.number_input("Current Score", 0, 300, 90)
        wickets2      = st.slider("Wickets Fallen", 0, 9, 3, key="wk2")
        over_num2     = st.slider("Overs", 0, 19, 10, key="ov2")
        ball_num2     = st.slider("Balls", 0, 5, 0, key="ball2")
        overs2        = round(over_num2 + (ball_num2 / 6), 2)
        st.caption(f"{over_num2}.{ball_num2} overs")

    if st.button("Predict Winner", key="btn2"):
        crr       = current_score / max(overs2, 0.1)
        rrr       = (target - current_score) * 6 / max((120 - overs2 * 6), 1)
        pct_done  = current_score / max(target, 1)
        pct_overs = overs2 / 20
        X2 = np.array([[TEAM_ENC[chasing_team], TEAM_ENC[defending_team], VENUE_ENC[venue2],
                         target, wickets2, overs2, crr, rrr, pct_done, pct_overs]])
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
# TAB 3 — MODEL REPORT (tables)
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
# TAB 4 — MODEL COMPARISON (graphs)  ← NEW TAB
# =========================================================

with tab4:
    st.subheader("📈 Model Comparison — Visual Analysis")
    st.markdown("Graphical comparison of all ML models across every key metric for both tasks.")

    ORANGE = "#ff6b00"
    BLUE   = "#1f77b4"
    GREEN  = "#2ca02c"
    RED    = "#d62728"
    PURPLE = "#9467bd"
    PALETTE = [ORANGE, BLUE, GREEN, RED]

    reg_df = pd.DataFrame(REG_METRICS).T.reset_index().rename(columns={"index": "Model"})
    cls_df = pd.DataFrame(CLS_METRICS).T.reset_index().rename(columns={"index": "Model"})

    # convert to numeric
    for col in ["Train R²", "Test R²", "RMSE", "MAE"]:
        reg_df[col] = pd.to_numeric(reg_df[col])
    for col in ["Train Acc %", "Test Acc %", "Precision %", "Recall %", "F1 Score %"]:
        cls_df[col] = pd.to_numeric(cls_df[col])

    models_reg = reg_df["Model"].tolist()
    models_cls = cls_df["Model"].tolist()

    # -------------------------------------------------------
    # SECTION A: REGRESSION GRAPHS
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎯 Regression Model Comparison")

    # --- Graph 1: Grouped bar — Train R² vs Test R² ---
    st.markdown("#### Graph 1: Train R² vs Test R² (Overfitting Check)")
    st.caption("Smaller gap between bars = better generalization. Large gap = model is overfitting.")

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    x = np.arange(len(models_reg))
    w = 0.35
    bars1 = ax1.bar(x - w/2, reg_df["Train R²"], w, label="Train R²", color=ORANGE, edgecolor="white", linewidth=0.7)
    bars2 = ax1.bar(x + w/2, reg_df["Test R²"],  w, label="Test R²",  color=BLUE,   edgecolor="white", linewidth=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_reg, fontsize=10)
    ax1.set_ylabel("R² Score", fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Train R² vs Test R² — Regression Models", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=ORANGE, fontweight="bold")
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=BLUE, fontweight="bold")
    ax1.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    best_reg = reg_df.loc[reg_df["Test R²"].idxmax(), "Model"]
    st.success(f"✅ **Best Regression Model (Test R²):** {best_reg} — Highest generalization on unseen data.")

    # --- Graph 2: RMSE & MAE side by side ---
    st.markdown("#### Graph 2: RMSE & MAE Comparison")
    st.caption("Lower RMSE and MAE = more accurate score predictions. RMSE penalizes large errors more.")

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4))

    # RMSE
    colors_rmse = [ORANGE if m == best_reg else "#cccccc" for m in models_reg]
    bars_rmse = ax2a.bar(models_reg, reg_df["RMSE"], color=colors_rmse, edgecolor="white")
    ax2a.set_title("RMSE (lower = better)", fontsize=12, fontweight="bold")
    ax2a.set_ylabel("RMSE (runs)")
    ax2a.set_ylim(0, reg_df["RMSE"].max() * 1.3)
    for bar in bars_rmse:
        ax2a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                  f"{bar.get_height():.1f}", ha="center", fontsize=9, fontweight="bold")
    ax2a.spines[["top","right"]].set_visible(False)
    ax2a.tick_params(axis="x", labelrotation=15)

    # MAE
    colors_mae = [ORANGE if m == best_reg else "#cccccc" for m in models_reg]
    bars_mae = ax2b.bar(models_reg, reg_df["MAE"], color=colors_mae, edgecolor="white")
    ax2b.set_title("MAE (lower = better)", fontsize=12, fontweight="bold")
    ax2b.set_ylabel("MAE (runs)")
    ax2b.set_ylim(0, reg_df["MAE"].max() * 1.3)
    for bar in bars_mae:
        ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                  f"{bar.get_height():.1f}", ha="center", fontsize=9, fontweight="bold")
    ax2b.spines[["top","right"]].set_visible(False)
    ax2b.tick_params(axis="x", labelrotation=15)

    best_patch = mpatches.Patch(color=ORANGE, label=f"Best model ({best_reg})")
    other_patch = mpatches.Patch(color="#cccccc", label="Other models")
    fig2.legend(handles=[best_patch, other_patch], loc="upper right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # --- Graph 3: Radar chart — Regression (normalized) ---
    st.markdown("#### Graph 3: Radar Chart — Regression Models (Normalized Metrics)")
    st.caption("Each axis is normalized 0–1. Larger area = better overall performance.")

    reg_radar = reg_df.copy()
    reg_radar["R² Score"]  = reg_radar["Test R²"]
    reg_radar["1-RMSE_n"]  = 1 - (reg_radar["RMSE"] - reg_radar["RMSE"].min()) / (reg_radar["RMSE"].max() - reg_radar["RMSE"].min() + 1e-9)
    reg_radar["1-MAE_n"]   = 1 - (reg_radar["MAE"]  - reg_radar["MAE"].min())  / (reg_radar["MAE"].max()  - reg_radar["MAE"].min()  + 1e-9)
    reg_radar["Train-Test Gap (inv)"] = 1 - abs(reg_radar["Train R²"] - reg_radar["Test R²"]) / 0.3

    radar_cols = ["R² Score", "1-RMSE_n", "1-MAE_n", "Train-Test Gap (inv)"]
    radar_labels = ["Test R²", "Low RMSE", "Low MAE", "No Overfit"]
    N = len(radar_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    radar_colors = [ORANGE, BLUE, GREEN, RED]

    for i, row in reg_radar.iterrows():
        values = row[radar_cols].tolist()
        values += values[:1]
        ax3.plot(angles, values, linewidth=2, linestyle="solid", label=row["Model"], color=radar_colors[i % 4])
        ax3.fill(angles, values, alpha=0.10, color=radar_colors[i % 4])

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(radar_labels, fontsize=10, fontweight="bold")
    ax3.set_ylim(0, 1)
    ax3.set_title("Regression — Multi-Metric Radar", fontsize=12, fontweight="bold", pad=15)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax3.spines["polar"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # --- Graph 4: Actual vs Predicted scatter ---
    st.markdown("#### Graph 4: Actual vs Predicted Score — All Models")
    st.caption("Points close to the diagonal line = accurate predictions. Spread = error.")

    yr_test_vals = yr_test.values
    fig4, axes4 = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for i, (name, model) in enumerate(REG_MODELS.items()):
        preds = model.predict(Xr_test)
        r2    = r2_score(yr_test_vals, preds)
        axes4[i].scatter(yr_test_vals, preds, alpha=0.25, s=12, color=radar_colors[i], edgecolors="none")
        mn, mx = yr_test_vals.min(), yr_test_vals.max()
        axes4[i].plot([mn, mx], [mn, mx], "k--", linewidth=1.2)
        axes4[i].set_title(f"{name}\nR²={r2:.3f}", fontsize=10, fontweight="bold")
        axes4[i].set_xlabel("Actual Score", fontsize=9)
        if i == 0:
            axes4[i].set_ylabel("Predicted Score", fontsize=9)
        axes4[i].spines[["top","right"]].set_visible(False)
    plt.suptitle("Actual vs Predicted — Regression Models", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # -------------------------------------------------------
    # SECTION B: CLASSIFICATION GRAPHS
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🏆 Classification Model Comparison")

    best_cls = cls_df.loc[cls_df["Test Acc %"].idxmax(), "Model"]

    # --- Graph 5: Grouped bar — all classification metrics ---
    st.markdown("#### Graph 5: All Classification Metrics Side-by-Side")
    st.caption("Higher bar = better. F1 Score balances precision and recall — most reliable single metric.")

    cls_metrics_cols = ["Test Acc %", "Precision %", "Recall %", "F1 Score %"]
    x5  = np.arange(len(models_cls))
    w5  = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    metric_colors = [ORANGE, BLUE, GREEN, PURPLE]

    fig5, ax5 = plt.subplots(figsize=(12, 5))
    for j, (metric, color) in enumerate(zip(cls_metrics_cols, metric_colors)):
        bars = ax5.bar(x5 + offsets[j] * w5, cls_df[metric], w5,
                       label=metric, color=color, edgecolor="white", linewidth=0.5)
        for bar in bars:
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{bar.get_height():.1f}", ha="center", fontsize=7, fontweight="bold")
    ax5.set_xticks(x5)
    ax5.set_xticklabels(models_cls, fontsize=11)
    ax5.set_ylabel("Score (%)", fontsize=11)
    ax5.set_ylim(0, 115)
    ax5.set_title("Classification Metrics Comparison — All Models", fontsize=13, fontweight="bold")
    ax5.legend(fontsize=10)
    ax5.axhline(100, color="grey", linestyle="--", linewidth=0.6, alpha=0.4)
    ax5.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    st.success(f"✅ **Best Classification Model (Test Accuracy):** {best_cls} — Highest accuracy on unseen data.")

    # --- Graph 6: Train Acc vs Test Acc (overfit) ---
    st.markdown("#### Graph 6: Train Accuracy vs Test Accuracy (Overfitting Check)")
    st.caption("Models where Train Acc >> Test Acc are overfitting and may not generalize well.")

    fig6, ax6 = plt.subplots(figsize=(9, 4))
    x6 = np.arange(len(models_cls))
    b1 = ax6.bar(x6 - 0.2, cls_df["Train Acc %"], 0.38, label="Train Acc %", color=ORANGE, edgecolor="white")
    b2 = ax6.bar(x6 + 0.2, cls_df["Test Acc %"],  0.38, label="Test Acc %",  color=BLUE,   edgecolor="white")
    ax6.set_xticks(x6)
    ax6.set_xticklabels(models_cls, fontsize=10)
    ax6.set_ylabel("Accuracy (%)", fontsize=11)
    ax6.set_ylim(0, 115)
    ax6.set_title("Train vs Test Accuracy — Classification Models", fontsize=13, fontweight="bold")
    ax6.legend(fontsize=10)
    for bar in [*b1, *b2]:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax6.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    # --- Graph 7: Radar chart — Classification ---
    st.markdown("#### Graph 7: Radar Chart — Classification Models (Multi-Metric)")
    st.caption("Larger shaded area = stronger overall performance across all metrics.")

    radar_cls_cols  = ["Test Acc %", "Precision %", "Recall %", "F1 Score %"]
    radar_cls_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    cls_radar = cls_df[["Model"] + radar_cls_cols].copy()
    for col in radar_cls_cols:
        cls_radar[col] = cls_radar[col] / 100.0

    N2 = len(radar_cls_cols)
    angles2 = np.linspace(0, 2 * np.pi, N2, endpoint=False).tolist()
    angles2 += angles2[:1]

    fig7, ax7 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, row in cls_radar.iterrows():
        values = row[radar_cls_cols].tolist()
        values += values[:1]
        ax7.plot(angles2, values, linewidth=2, linestyle="solid", label=row["Model"], color=radar_colors[i % 4])
        ax7.fill(angles2, values, alpha=0.10, color=radar_colors[i % 4])

    ax7.set_xticks(angles2[:-1])
    ax7.set_xticklabels(radar_cls_labels, fontsize=10, fontweight="bold")
    ax7.set_ylim(0, 1)
    ax7.set_title("Classification — Multi-Metric Radar", fontsize=12, fontweight="bold", pad=15)
    ax7.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax7.spines["polar"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig7)
    plt.close()

    # --- Graph 8: Confusion Matrices — all models side by side ---
    st.markdown("#### Graph 8: Confusion Matrices — All Classification Models")
    st.caption("True Positives (top-left) and True Negatives (bottom-right) should be highest. Off-diagonal = errors.")

    Xc_feat = win_df[[
        "batting_team","bowling_team","venue","target","team_wicket",
        "overs_completed","current_run_rate","required_run_rate",
        "pct_target_done","pct_overs_done"
    ]]
    yc_full = win_df["won_chase"]
    _, Xc_t, _, yc_t = train_test_split(Xc_feat, yc_full, test_size=0.2, random_state=42)

    fig8, axes8 = plt.subplots(1, 4, figsize=(18, 4))
    class_labels = ["Def.\nWon", "Chase\nWon"]
    for i, (name, model) in enumerate(CLS_MODELS.items()):
        preds_cm = model.predict(Xc_t)
        cm = confusion_matrix(yc_t, preds_cm)
        im = axes8[i].imshow(cm, cmap="Oranges", aspect="auto")
        axes8[i].set_xticks([0, 1])
        axes8[i].set_yticks([0, 1])
        axes8[i].set_xticklabels(class_labels, fontsize=9)
        axes8[i].set_yticklabels(class_labels, fontsize=9)
        axes8[i].set_xlabel("Predicted", fontsize=9)
        if i == 0:
            axes8[i].set_ylabel("Actual", fontsize=9)
        for row in range(2):
            for col in range(2):
                axes8[i].text(col, row, str(cm[row, col]),
                              ha="center", va="center",
                              fontsize=14, fontweight="bold",
                              color="white" if cm[row, col] > cm.max() * 0.5 else "black")
        acc = accuracy_score(yc_t, preds_cm) * 100
        axes8[i].set_title(f"{name}\nAcc: {acc:.1f}%", fontsize=10, fontweight="bold")
    plt.suptitle("Confusion Matrices — All Classification Models", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()

    # --- Graph 9: Feature Importance — both tasks ---
    st.markdown("#### Graph 9: Feature Importance — Random Forest (Both Tasks)")
    st.caption("Higher importance = that feature contributes more to the model's decisions.")

    reg_feats  = ["batting_team","bowling_team","venue","current_runs","wickets","overs","crr"]
    cls_feats  = ["batting_team","bowling_team","venue","target","team_wicket",
                  "overs_completed","crr","rrr","pct_target","pct_overs"]

    rf_reg_imp = REG_MODELS["Random Forest"].feature_importances_
    rf_cls_imp = CLS_MODELS["Random Forest"].feature_importances_

    fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(14, 4))

    # Regression importance
    sort_idx_r = np.argsort(rf_reg_imp)
    ax9a.barh([reg_feats[i] for i in sort_idx_r], rf_reg_imp[sort_idx_r], color=ORANGE)
    ax9a.set_title("Feature Importance\nRF Regressor (Score Prediction)", fontsize=11, fontweight="bold")
    ax9a.set_xlabel("Importance")
    ax9a.spines[["top","right"]].set_visible(False)

    # Classification importance
    sort_idx_c = np.argsort(rf_cls_imp)
    ax9b.barh([cls_feats[i] for i in sort_idx_c], rf_cls_imp[sort_idx_c], color=BLUE)
    ax9b.set_title("Feature Importance\nRF Classifier (Win Prediction)", fontsize=11, fontweight="bold")
    ax9b.set_xlabel("Importance")
    ax9b.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig9)
    plt.close()

    # -------------------------------------------------------
    # FINAL SUMMARY TABLE
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🏅 Best Model Summary")

    best_reg_row = reg_df.loc[reg_df["Test R²"].idxmax()]
    best_cls_row = cls_df.loc[cls_df["Test Acc %"].idxmax()]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="result-box">
        <p style="font-size:16px; font-weight:bold;">🎯 Best Regression Model</p>
        <p class="win-team" style="font-size:28px;">{best_reg_row['Model']}</p>
        <p>Test R²: <b>{best_reg_row['Test R²']}</b> &nbsp;|&nbsp;
           RMSE: <b>{best_reg_row['RMSE']}</b> &nbsp;|&nbsp;
           MAE: <b>{best_reg_row['MAE']}</b></p>
        <p style="color:#555; font-size:13px;">
        Ensemble boosting captures non-linear run-progression patterns
        better than linear or single-tree models.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="result-box">
        <p style="font-size:16px; font-weight:bold;">🏆 Best Classification Model</p>
        <p class="win-team" style="font-size:28px;">{best_cls_row['Model']}</p>
        <p>Test Acc: <b>{best_cls_row['Test Acc %']}%</b> &nbsp;|&nbsp;
           F1: <b>{best_cls_row['F1 Score %']}%</b> &nbsp;|&nbsp;
           Precision: <b>{best_cls_row['Precision %']}%</b></p>
        <p style="color:#555; font-size:13px;">
        Strong win/loss signal captured through RRR, CRR, and % target done features.
        Minimal overfitting with regularized tree depth.
        </p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 5 — DATA ANALYSIS
# =========================================================

with tab5:
    st.subheader("🔍 Data Analysis & Pipeline")

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

    st.markdown("### Step 2: Data Cleaning")
    null_counts = raw_df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    st.markdown(f"""
    <div class="analysis-card">
    <b>Duplicate Rows:</b> {raw_df.duplicated().sum()}<br>
    <b>Columns with Null Values:</b> {len(null_cols)}<br>
    <b>Team Name Fixes Applied:</b> {len(TEAM_NAME_MAPPING)} mappings<br>
    <b>Venue Name Fixes Applied:</b> {len(VENUE_MAPPING)} mappings<br>
    <b>Teams Filtered:</b> Only current 10 IPL teams kept
    </div>
    """, unsafe_allow_html=True)

    if len(null_cols) > 0:
        st.caption("Null value counts per column:")
        st.dataframe(null_cols.reset_index().rename(columns={"index": "Column", 0: "Null Count"}), use_container_width=True)
    else:
        st.caption("✅ No null values found in key columns.")

    st.markdown("### Step 3: Feature Engineering")
    st.markdown("""
    <div class="analysis-card">
    <b>New Features Created:</b><br>
    • <b>overs_completed</b> = over + (ball / 6)<br>
    • <b>current_run_rate (CRR)</b> = team_runs / overs_completed<br>
    • <b>final_score</b> = max team_runs per match-innings<br>
    • <b>required_run_rate (RRR)</b> = runs_needed × 6 / balls_left<br>
    • <b>pct_target_done</b> = current_score / target<br>
    • <b>pct_overs_done</b> = overs_completed / 20
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Step 4: Label / Target Creation")
    st.markdown("""
    <div class="analysis-card">
    <b>Regression Target:</b> final_score<br>
    <b>Classification Target:</b> won_chase (0 or 1) — determined from last ball, no leakage
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Step 5: Dataset Summary After Processing")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches",       str(df["match_id"].nunique()))
    col2.metric("Regression Rows",     str(len(score_df)))
    col3.metric("Classification Rows", str(len(win_df)))
    col4.metric("Unique Venues",       str(len(IPL_VENUES)))

    st.markdown("### Step 6: Label Encoding")
    st.markdown("""
    <div class="analysis-card">
    <b>Method:</b> Integer Label Encoding (manual dictionary)<br>
    <b>batting_team / bowling_team:</b> Each team mapped to 0–9<br>
    <b>venue:</b> Each venue mapped to integer index
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Step 7: Train-Test Split")
    st.markdown("""
    <div class="analysis-card">
    <b>Split Ratio:</b> 80% Train / 20% Test | <b>random_state:</b> 42
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Step 8: Models Used")
    model_info = {
        "Model": ["Random Forest","Gradient Boosting","Linear Regression / Logistic Regression","Decision Tree"],
        "Type":  ["Ensemble (Bagging)","Ensemble (Boosting)","Linear","Single Tree"],
        "Used For": ["Both","Both","Regression + Classification","Both"]
    }
    st.dataframe(pd.DataFrame(model_info), use_container_width=True)

    st.markdown("### Step 9: Evaluation Metrics")
    st.markdown("""
    <div class="analysis-card">
    <b>Regression:</b> R² (Train & Test), RMSE, MAE, Overfit Check<br>
    <b>Classification:</b> Accuracy (Train & Test), Precision, Recall, F1 Score, Overfit Check
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Team Match Count (Filtered Dataset)")
    team_counts = pd.concat([df["batting_team"], df["bowling_team"]]).value_counts().reset_index()
    team_counts.columns = ["Team", "Ball-by-Ball Rows"]
    st.dataframe(team_counts, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    "<center>IPL Predictor — Ball-by-Ball Dataset | ML Pipeline with 4 Models | FAST NUCES CF Campus 2026</center>",
    unsafe_allow_html=True
)
