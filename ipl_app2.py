import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Predictor",
    page_icon="🏏",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8e8;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1b2a 50%, #0a0a0f 100%);
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
}

.title-box {
    background: linear-gradient(90deg, #ff6b00, #ff9d00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    text-align: center;
    letter-spacing: 4px;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #7a8a9a;
    font-size: 1.1rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,107,0,0.2);
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
}

.result-box {
    background: linear-gradient(135deg, rgba(255,107,0,0.15), rgba(255,157,0,0.05));
    border: 2px solid #ff6b00;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
}

.big-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    color: #ff6b00;
    line-height: 1;
}

.win-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: #00e676;
    letter-spacing: 3px;
}

.lose-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: #ff4444;
    letter-spacing: 3px;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #ff9d00 !important;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 1px;
}

.stButton > button {
    background: linear-gradient(90deg, #ff6b00, #ff9d00) !important;
    color: #000 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.4rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 40px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255,107,0,0.4) !important;
}

.divider {
    border: none;
    border-top: 1px solid rgba(255,107,0,0.2);
    margin: 24px 0;
}

[data-testid="stMetricValue"] {
    color: #ff9d00 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
}

.tab-info {
    background: rgba(255,107,0,0.08);
    border-left: 3px solid #ff6b00;
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 20px;
    font-size: 0.95rem;
    color: #aab;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #ff6b00, #ff9d00) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# IPL DATA — Teams, Venues
# ─────────────────────────────────────────────

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

IPL_VENUES = [
    "Wankhede Stadium, Mumbai",
    "M. A. Chidambaram Stadium, Chennai",
    "Eden Gardens, Kolkata",
    "Narendra Modi Stadium, Ahmedabad",
    "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi Intl. Cricket Stadium, Hyderabad",
    "M. Chinnaswamy Stadium, Bangalore",
    "Sawai Mansingh Stadium, Jaipur",
    "PCA Stadium, Mohali",
    "BRSABVE Cricket Stadium, Lucknow"
]

# Team encoding map
TEAM_ENC = {t: i for i, t in enumerate(IPL_TEAMS)}
VENUE_ENC = {v: i for i, v in enumerate(IPL_VENUES)}

# ─────────────────────────────────────────────
# SYNTHETIC TRAINING DATA
# IPL-realistic distributions for model training
# ─────────────────────────────────────────────

@st.cache_resource
def train_models():
    np.random.seed(42)
    n = 5000

    # ---- REGRESSION TRAINING DATA (Score Prediction) ----
    batting_team = np.random.randint(0, 10, n)
    bowling_team = np.random.randint(0, 10, n)
    venue        = np.random.randint(0, 10, n)
    current_runs = np.random.randint(0, 200, n)
    wickets      = np.random.randint(0, 10, n)
    overs        = np.random.uniform(1, 20, n)
    last5_runs   = np.random.randint(0, 80, n)
    last5_wkts   = np.random.randint(0, 5, n)

    # Realistic final score formula
    crr = current_runs / np.maximum(overs, 0.1)
    remaining_overs = 20 - overs
    projected = current_runs + (crr * remaining_overs)
    wicket_penalty = wickets * 4
    momentum_bonus = last5_runs * 0.3
    noise = np.random.normal(0, 12, n)

    final_score = projected - wicket_penalty + momentum_bonus + noise
    final_score = np.clip(final_score, 60, 260).astype(int)

    X_reg = np.column_stack([batting_team, bowling_team, venue,
                              current_runs, wickets, overs,
                              last5_runs, last5_wkts])

    # Train 3 regression models
    rf_reg  = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_reg  = GradientBoostingRegressor(n_estimators=100, random_state=42)
    lr_reg  = LinearRegression()
    dt_reg  = DecisionTreeRegressor(max_depth=8, random_state=42)

    rf_reg.fit(X_reg, final_score)
    gb_reg.fit(X_reg, final_score)
    lr_reg.fit(X_reg, final_score)
    dt_reg.fit(X_reg, final_score)

    # ---- CLASSIFICATION TRAINING DATA (Win Prediction) ----
    team1       = np.random.randint(0, 10, n)
    team2       = np.random.randint(0, 10, n)
    venue2      = np.random.randint(0, 10, n)
    target      = np.random.randint(120, 230, n)
    curr_score  = np.random.randint(0, 200, n)
    overs2      = np.random.uniform(1, 20, n)
    wkts2       = np.random.randint(0, 10, n)
    rrr         = (target - curr_score) / np.maximum(20 - overs2, 0.1)
    crr2        = curr_score / np.maximum(overs2, 0.1)

    # Win probability logic
    win_prob = 0.5 + 0.25 * (crr2 - rrr) / 5 - 0.05 * wkts2
    win_prob = np.clip(win_prob, 0.05, 0.95)
    win_label = (np.random.rand(n) < win_prob).astype(int)

    X_cls = np.column_stack([team1, team2, venue2, target,
                              curr_score, overs2, wkts2, rrr, crr2])

    rf_cls  = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_cls  = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr_cls  = LogisticRegression(max_iter=500)
    dt_cls  = DecisionTreeClassifier(max_depth=8, random_state=42)

    rf_cls.fit(X_cls, win_label)
    gb_cls.fit(X_cls, win_label)
    lr_cls.fit(X_cls, win_label)
    dt_cls.fit(X_cls, win_label)

    return {
        'rf_reg': rf_reg, 'gb_reg': gb_reg,
        'lr_reg': lr_reg, 'dt_reg': dt_reg,
        'rf_cls': rf_cls, 'gb_cls': gb_cls,
        'lr_cls': lr_cls, 'dt_cls': dt_cls
    }

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="title-box">🏏 IPL PREDICTOR</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning for Business Analytics — Spring 2026</p>', unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Load models (cached)
with st.spinner("Loading ML models..."):
    models = train_models()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Score Predictor", "🏆  Win Predictor", "📊  Model Info"])

# ══════════════════════════════════════════════
# TAB 1 — SCORE PREDICTOR
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="tab-info">Predict the <b>final innings score</b> based on current match conditions.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏟️ Match Setup")
        batting_team  = st.selectbox("Batting Team", IPL_TEAMS, key="sc_bat")
        bowling_team_list = [t for t in IPL_TEAMS if t != batting_team]
        bowling_team  = st.selectbox("Bowling Team", bowling_team_list, key="sc_bowl")
        venue         = st.selectbox("Venue", IPL_VENUES, key="sc_venue")

    with col2:
        st.markdown("### 📈 Current Innings Status")
        current_runs  = st.number_input("Current Runs", min_value=0, max_value=250, value=85, step=1)
        wickets       = st.slider("Wickets Fallen", 0, 9, 2)
        overs         = st.slider("Overs Completed", 1.0, 19.5, 10.0, step=0.1)
        last5_runs    = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=100, value=45, step=1)
        last5_wkts    = st.slider("Wickets in Last 5 Overs", 0, 5, 1)

    st.markdown("")
    model_choice_reg = st.selectbox(
        "Select ML Model",
        ["Random Forest", "Gradient Boosting", "Linear Regression", "Decision Tree"],
        key="reg_model"
    )

    if st.button("🎯 PREDICT FINAL SCORE", key="btn_score"):
        bt = TEAM_ENC[batting_team]
        bw = TEAM_ENC[bowling_team] if bowling_team in TEAM_ENC else 0
        v  = VENUE_ENC[venue]
        X  = np.array([[bt, bw, v, current_runs, wickets, overs, last5_runs, last5_wkts]])

        model_map = {
            "Random Forest":      models['rf_reg'],
            "Gradient Boosting":  models['gb_reg'],
            "Linear Regression":  models['lr_reg'],
            "Decision Tree":      models['dt_reg']
        }
        pred = int(model_map[model_choice_reg].predict(X)[0])
        pred = max(60, min(260, pred))

        crr = round(current_runs / max(overs, 0.1), 2)
        remaining = round(20 - overs, 1)

        st.markdown(f"""
        <div class="result-box">
            <p style="color:#aab; font-size:1rem; letter-spacing:2px; margin-bottom:4px;">PREDICTED FINAL SCORE</p>
            <p class="big-number">{pred}</p>
            <p style="color:#ff9d00; font-size:1.2rem;">{batting_team}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Run Rate", f"{crr}")
        m2.metric("Overs Remaining", f"{remaining}")
        m3.metric("Runs Required to Beat Pred.", f"{pred - current_runs}")

# ══════════════════════════════════════════════
# TAB 2 — WIN PREDICTOR
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="tab-info">Predict the <b>match winner</b> based on live chase conditions.</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🏟️ Match Setup")
        team1 = st.selectbox("Team 1 (Batting in Chase)", IPL_TEAMS, key="win_t1")
        team2_list = [t for t in IPL_TEAMS if t != team1]
        team2 = st.selectbox("Team 2 (Defending)", team2_list, key="win_t2")
        venue2 = st.selectbox("Venue", IPL_VENUES, key="win_venue")

    with col4:
        st.markdown("### 📊 Chase Conditions")
        target      = st.number_input("Target Score", min_value=50, max_value=280, value=175, step=1)
        curr_score  = st.number_input("Current Score (Chasing Team)", min_value=0, max_value=280, value=90, step=1)
        overs2      = st.slider("Overs Completed", 1.0, 19.5, 10.0, step=0.1, key="win_ov")
        wkts2       = st.slider("Wickets Fallen", 0, 9, 3, key="win_wk")

    st.markdown("")
    model_choice_cls = st.selectbox(
        "Select ML Model",
        ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree"],
        key="cls_model"
    )

    if st.button("🏆 PREDICT WINNER", key="btn_win"):
        t1  = TEAM_ENC[team1]
        t2  = TEAM_ENC[team2]
        v2  = VENUE_ENC[venue2]
        rrr = (target - curr_score) / max(20 - overs2, 0.1)
        crr2 = curr_score / max(overs2, 0.1)
        X2  = np.array([[t1, t2, v2, target, curr_score, overs2, wkts2, rrr, crr2]])

        model_map2 = {
            "Random Forest":      models['rf_cls'],
            "Gradient Boosting":  models['gb_cls'],
            "Logistic Regression": models['lr_cls'],
            "Decision Tree":      models['dt_cls']
        }
        pred_cls = model_map2[model_choice_cls].predict(X2)[0]
        proba    = model_map2[model_choice_cls].predict_proba(X2)[0]

        winner     = team1 if pred_cls == 1 else team2
        win_conf   = round(max(proba) * 100, 1)
        label_cls  = "win-label" if pred_cls == 1 else "lose-label"

        st.markdown(f"""
        <div class="result-box">
            <p style="color:#aab; font-size:1rem; letter-spacing:2px; margin-bottom:4px;">PREDICTED WINNER</p>
            <p class="{label_cls}">🏆 {winner}</p>
            <p style="color:#aab; margin-top:8px;">Confidence: <b style="color:#ff9d00;">{win_conf}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m4, m5, m6 = st.columns(3)
        m4.metric("Current Run Rate", f"{round(crr2, 2)}")
        m5.metric("Required Run Rate", f"{round(rrr, 2)}")
        m6.metric("Runs Needed", f"{target - curr_score}")

        st.markdown("**Win Probability**")
        st.progress(int(proba[1] * 100))
        st.caption(f"{team1}: {round(proba[1]*100,1)}%  |  {team2}: {round(proba[0]*100,1)}%")

# ══════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Models Used in This App")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#### 🎯 Regression Models (Score Prediction)")
        reg_df = pd.DataFrame({
            "Model": ["Random Forest", "Gradient Boosting", "Linear Regression", "Decision Tree"],
            "Type": ["Ensemble", "Ensemble", "Linear", "Tree"],
            "Metric": ["RMSE / R²", "RMSE / R²", "RMSE / R²", "RMSE / R²"]
        })
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

    with col6:
        st.markdown("#### 🏆 Classification Models (Win Prediction)")
        cls_df = pd.DataFrame({
            "Model": ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree"],
            "Type": ["Ensemble", "Ensemble", "Linear", "Tree"],
            "Metric": ["Accuracy / F1", "Accuracy / F1", "Accuracy / F1", "Accuracy / F1"]
        })
        st.dataframe(cls_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📋 Features Used")

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("""
        **Score Predictor Features:**
        - Batting Team
        - Bowling Team
        - Venue
        - Current Runs
        - Wickets Fallen
        - Overs Completed
        - Runs in Last 5 Overs
        - Wickets in Last 5 Overs
        """)
    with fc2:
        st.markdown("""
        **Win Predictor Features:**
        - Team 1 (Chasing)
        - Team 2 (Defending)
        - Venue
        - Target Score
        - Current Score
        - Overs Completed
        - Wickets Fallen
        - Current Run Rate (calculated)
        - Required Run Rate (calculated)
        """)

    st.markdown("---")
    st.info("📌 Models are trained on IPL-realistic synthetic data distributions. For production use, connect to the Kaggle IPL ball-by-ball dataset.")

# Footer
st.markdown("<br><hr class='divider'>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#444; font-size:0.85rem;">ML for Business Analytics — Spring 2026 | FAST NUCES CF Campus</p>', unsafe_allow_html=True)
