import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier,
                               AdaBoostRegressor, AdaBoostClassifier,
                               ExtraTreesRegressor, ExtraTreesClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, f1_score, precision_score, recall_score)

# ─────────────────────────────────────────────
st.set_page_config(page_title="IPL Predictor", page_icon="🏏", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@400;600;700&display=swap');

html, body { background-color: #ffffff !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .block-container {
    background-color: #ffffff !important;
}
p, span, div, li, td, th, .stMarkdown p, [data-testid="stText"] {
    color: #111111 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
h1, h2, h3, h4 { color: #111111 !important; font-family: 'Rajdhani', sans-serif !important; }
label, .stSelectbox label, .stSlider label, .stNumberInput label,
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
    color: #222222 !important; font-weight: 700 !important;
    font-size: 0.95rem !important; font-family: 'Rajdhani', sans-serif !important;
}
.stSelectbox > div > div, input[type="number"] {
    background: #f9f9f9 !important; color: #111111 !important; border: 1px solid #cccccc !important;
}
button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
    color: #333333 !important; font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] p { color: #d44000 !important; }
[data-testid="stMetricValue"] {
    color: #d44000 !important; font-family: 'Bebas Neue', sans-serif !important;
}
[data-testid="stMetricLabel"] p { color: #444444 !important; }
.stButton > button {
    background: linear-gradient(90deg, #d44000, #ff8c00) !important;
    color: #ffffff !important; font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.3rem !important; letter-spacing: 2px !important;
    border: none !important; border-radius: 8px !important;
    padding: 12px 32px !important; width: 100% !important;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #d44000, #ff8c00) !important;
}
[data-testid="stDataFrame"] * { color: #111111 !important; }
.stCaption, small { color: #555555 !important; }

.ipl-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 3.5rem; text-align: center;
    background: linear-gradient(90deg, #d44000, #ff8c00);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: 4px; margin-bottom: 0;
}
.ipl-subtitle { text-align: center; color: #555555 !important; font-size: 0.95rem; letter-spacing: 2px; text-transform: uppercase; }
.tab-info { background: #fff3e0; border-left: 4px solid #ff6d00; padding: 10px 16px; border-radius: 0 8px 8px 0; margin-bottom: 20px; color: #333333 !important; font-size: 1rem; }
.section-title { font-family: 'Bebas Neue', sans-serif; font-size: 1.3rem; color: #d44000 !important; letter-spacing: 2px; border-bottom: 2px solid #ffcc99; padding-bottom: 4px; margin-bottom: 14px; }
.result-box { background: linear-gradient(135deg, #fff3e0, #ffe0b2); border: 2px solid #ff6d00; border-radius: 14px; padding: 28px; text-align: center; margin-top: 16px; }
.big-score { font-family: 'Bebas Neue', sans-serif; font-size: 5rem; color: #d44000 !important; -webkit-text-fill-color: #d44000 !important; line-height: 1; }
.winner-name { font-family: 'Bebas Neue', sans-serif; font-size: 2.4rem; color: #1a7a1a !important; -webkit-text-fill-color: #1a7a1a !important; letter-spacing: 2px; }
.res-label { color: #666666 !important; font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase; }
.team-sub { color: #d44000 !important; font-size: 1.1rem; font-weight: 700; }
.data-badge { background: #e8f5e9; border: 1px solid #4caf50; color: #1b5e20 !important; border-radius: 6px; padding: 6px 14px; font-size: 0.88rem; display: inline-block; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1mr2IIjhMOtRp0ZDlVLw_IFxmAY_ExGUL"

@st.cache_data(show_spinner=False)
def load_data():
    """Load real IPL ball-by-ball CSV from Google Drive."""
    df = pd.read_csv(GDRIVE_URL, low_memory=False)
    return df

@st.cache_data(show_spinner=False)
def build_features(df):
    """
    From ball-by-ball data, build two ML-ready datasets:

    REGRESSION  → predict final innings score
      Features per innings (up to over N):
        batting_team_enc, bowling_team_enc, venue_enc,
        runs_so_far, wickets_so_far, overs_so_far,
        last5_runs, last5_wickets, run_rate
      Target: final_score

    CLASSIFICATION → predict match winner (1st innings team wins or not)
      Features per match:
        batting_team_enc, bowling_team_enc, venue_enc,
        target, chasing_runs_at_10, chasing_wickets_at_10,
        rrr_at_10, crr_at_10, powerplay_runs
      Target: 1 = team batting 1st wins, 0 = team batting 2nd wins
    """
    # ── Label encode teams & venues ──
    all_teams  = pd.concat([df['batting_team'], df['bowling_team']]).dropna().unique()
    all_venues = df['venue'].dropna().unique()

    team_enc  = {t: i for i, t in enumerate(sorted(all_teams))}
    venue_enc = {v: i for i, v in enumerate(sorted(all_venues))}

    df = df.copy()
    df['batting_team_enc'] = df['batting_team'].map(team_enc).fillna(0).astype(int)
    df['bowling_team_enc'] = df['bowling_team'].map(team_enc).fillna(0).astype(int)
    df['venue_enc']        = df['venue'].map(venue_enc).fillna(0).astype(int)

    # ── REGRESSION DATASET ──
    reg_rows = []
    inn1 = df[df['innings'] == 1].copy()
    grouped = inn1.groupby(['match_id'])

    for match_id, mdf in grouped:
        mdf = mdf.sort_values(['over', 'ball'])
        final_score = mdf['runs_total'].sum()
        if final_score < 50:
            continue

        bat_enc  = mdf['batting_team_enc'].iloc[0]
        bowl_enc = mdf['bowling_team_enc'].iloc[0]
        ven_enc  = mdf['venue_enc'].iloc[0]

        # Snapshot at over 10 (ball 60)
        snap = mdf[mdf['ball_no'] <= 60]
        if len(snap) < 10:
            continue

        runs_so_far   = snap['runs_total'].sum()
        wkts_so_far   = snap['striker_out'].sum() if 'striker_out' in snap else 0
        overs_so_far  = min(len(snap) / 6, 20)
        last5_balls   = mdf[(mdf['ball_no'] > 30) & (mdf['ball_no'] <= 60)]
        last5_runs    = last5_balls['runs_total'].sum()
        last5_wkts    = last5_balls['striker_out'].sum() if 'striker_out' in last5_balls else 0
        run_rate      = runs_so_far / max(overs_so_far, 0.1)

        reg_rows.append([bat_enc, bowl_enc, ven_enc,
                         runs_so_far, int(wkts_so_far), overs_so_far,
                         last5_runs, int(last5_wkts), round(run_rate, 2),
                         final_score])

    reg_df = pd.DataFrame(reg_rows, columns=[
        'batting_team', 'bowling_team', 'venue',
        'runs_so_far', 'wickets_so_far', 'overs_so_far',
        'last5_runs', 'last5_wickets', 'run_rate', 'final_score'])

    # ── CLASSIFICATION DATASET ──
    cls_rows = []
    for match_id, mdf in df.groupby('match_id'):
        inn1m = mdf[mdf['innings'] == 1]
        inn2m = mdf[mdf['innings'] == 2]
        if inn1m.empty or inn2m.empty:
            continue

        target     = inn1m['runs_total'].sum() + 1
        bat_enc    = inn1m['batting_team_enc'].iloc[0]
        bowl_enc   = inn1m['bowling_team_enc'].iloc[0]
        ven_enc    = inn1m['venue_enc'].iloc[0]

        # powerplay (overs 0-5)
        pp = inn1m[inn1m['over'] < 6]
        powerplay_runs = pp['runs_total'].sum()

        # chase snapshot at over 10
        inn2m = inn2m.sort_values(['over', 'ball'])
        snap2 = inn2m[inn2m['ball_no'] <= 60]
        if len(snap2) < 6:
            continue

        cs_at10  = snap2['runs_total'].sum()
        wk_at10  = snap2['striker_out'].sum() if 'striker_out' in snap2 else 0
        ov_at10  = min(len(snap2) / 6, 20)
        rrr_at10 = (target - cs_at10) / max(20 - ov_at10, 0.1)
        crr_at10 = cs_at10 / max(ov_at10, 0.1)

        # winner: 1 = team batting 1st wins
        if 'match_won_by' not in mdf.columns:
            continue
        won_by = mdf['match_won_by'].dropna()
        if won_by.empty:
            continue
        won_by_str = str(won_by.iloc[0])
        inn1_team  = inn1m['batting_team'].iloc[0]
        label = 1 if inn1_team in won_by_str else 0

        cls_rows.append([bat_enc, bowl_enc, ven_enc,
                         target, cs_at10, ov_at10, int(wk_at10),
                         round(rrr_at10, 2), round(crr_at10, 2),
                         powerplay_runs, label])

    cls_df = pd.DataFrame(cls_rows, columns=[
        'batting_team', 'bowling_team', 'venue',
        'target', 'cs_at10', 'ov_at10', 'wk_at10',
        'rrr_at10', 'crr_at10', 'powerplay_runs', 'winner'])

    return reg_df, cls_df, team_enc, venue_enc, all_teams, all_venues

@st.cache_resource(show_spinner=False)
def train_all_models(_reg_df, _cls_df):
    """Train 8 regression + 8 classification models on real IPL data."""
    # ── REGRESSION ──
    Xr = _reg_df[['batting_team','bowling_team','venue','runs_so_far',
                   'wickets_so_far','overs_so_far','last5_runs','last5_wickets','run_rate']].values
    yr = _reg_df['final_score'].values
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    reg_models = {
        'Random Forest':       RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42),
        'Gradient Boosting':   GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
        'Extra Trees':         ExtraTreesRegressor(n_estimators=200, max_depth=12, random_state=42),
        'AdaBoost':            AdaBoostRegressor(n_estimators=100, random_state=42),
        'Decision Tree':       DecisionTreeRegressor(max_depth=10, random_state=42),
        'Linear Regression':   LinearRegression(),
        'Ridge Regression':    Ridge(alpha=1.0),
        'Lasso Regression':    Lasso(alpha=0.5),
    }
    reg_trained = {}
    reg_metrics = {}
    for name, mdl in reg_models.items():
        mdl.fit(Xr_tr, yr_tr)
        reg_trained[name] = mdl
        tr_p = mdl.predict(Xr_tr)
        te_p = mdl.predict(Xr_te)
        reg_metrics[name] = {
            'train_r2':   round(r2_score(yr_tr, tr_p), 4),
            'test_r2':    round(r2_score(yr_te, te_p), 4),
            'train_rmse': round(mean_squared_error(yr_tr, tr_p)**0.5, 2),
            'test_rmse':  round(mean_squared_error(yr_te, te_p)**0.5, 2),
            'test_mae':   round(mean_absolute_error(yr_te, te_p), 2),
        }

    # ── CLASSIFICATION ──
    Xc = _cls_df[['batting_team','bowling_team','venue','target',
                   'cs_at10','ov_at10','wk_at10','rrr_at10','crr_at10','powerplay_runs']].values
    yc = _cls_df['winner'].values
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42)

    cls_models = {
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
        'Extra Trees':         ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=42),
        'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'KNN':                 KNeighborsClassifier(n_neighbors=7),
        'SVM':                 SVC(probability=True, kernel='rbf', C=1.0),
    }
    cls_trained = {}
    cls_metrics = {}
    for name, mdl in cls_models.items():
        mdl.fit(Xc_tr, yc_tr)
        cls_trained[name] = mdl
        tr_p = mdl.predict(Xc_tr)
        te_p = mdl.predict(Xc_te)
        cls_metrics[name] = {
            'train_acc': round(accuracy_score(yc_tr, tr_p)*100, 2),
            'test_acc':  round(accuracy_score(yc_te, te_p)*100, 2),
            'f1':        round(f1_score(yc_te, te_p, zero_division=0)*100, 2),
            'precision': round(precision_score(yc_te, te_p, zero_division=0)*100, 2),
            'recall':    round(recall_score(yc_te, te_p, zero_division=0)*100, 2),
        }

    return reg_trained, reg_metrics, cls_trained, cls_metrics, Xr_te, yr_te, Xc_te, yc_te

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="ipl-title">🏏 IPL PREDICTOR</p>', unsafe_allow_html=True)
st.markdown('<p class="ipl-subtitle">Machine Learning for Business Analytics — Spring 2026 | FAST NUCES CF</p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────
with st.spinner("📥 Loading real IPL dataset from Google Drive..."):
    try:
        raw_df = load_data()
        data_ok = True
    except Exception as e:
        st.error(f"❌ Could not load data: {e}")
        data_ok = False

if data_ok:
    with st.spinner("⚙️ Building features from ball-by-ball data..."):
        reg_df, cls_df, team_enc, venue_enc, all_teams, all_venues = build_features(raw_df)

    st.markdown(f'<span class="data-badge">✅ Real IPL Data Loaded — {len(raw_df):,} deliveries | {reg_df.shape[0]} regression samples | {cls_df.shape[0]} classification samples</span>', unsafe_allow_html=True)

    with st.spinner("🤖 Training 16 ML models on real IPL data..."):
        reg_trained, reg_metrics, cls_trained, cls_metrics, Xr_te, yr_te, Xc_te, yc_te = train_all_models(reg_df, cls_df)

    st.success("✅ All 16 models trained on real IPL data!")

    IPL_TEAMS  = sorted(all_teams.tolist())
    IPL_VENUES = sorted(all_venues.tolist())

    # ─────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯  Score Predictor",
        "🏆  Win Predictor",
        "📊  Model Info",
        "📈  Model Report",
        "🗃️  Data Explorer"
    ])

    # ══ TAB 1 — Score Predictor ══
    with tab1:
        st.markdown('<div class="tab-info">Enter current match conditions to <b>predict the final innings score.</b> Uses real IPL data from 2008–2024.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-title">MATCH SETUP</p>', unsafe_allow_html=True)
            bat_t  = st.selectbox("Batting Team", IPL_TEAMS, key="s_bat")
            bowl_t = st.selectbox("Bowling Team", [t for t in IPL_TEAMS if t != bat_t], key="s_bowl")
            ven_s  = st.selectbox("Venue", IPL_VENUES, key="s_ven")
            mdl_r  = st.selectbox("ML Model", list(reg_trained.keys()), key="mdl_r")
        with c2:
            st.markdown('<p class="section-title">INNINGS STATUS</p>', unsafe_allow_html=True)
            curr_r   = st.number_input("Current Runs",            min_value=0, max_value=250, value=85)
            wkt_s    = st.slider("Wickets Fallen",                0, 9, 2)
            ov_num_s = st.slider("Overs Completed (0-19)",        0, 19, 10, key="sc_ov")
            bl_num_s = st.slider("Balls in Current Over (0-5)",   0, 5, 0, key="sc_bl")
            ovs_s    = round(ov_num_s + bl_num_s / 6, 4)
            st.caption(f"📌 {ov_num_s}.{bl_num_s} overs = {ovs_s:.2f}")
            l5r_s   = st.number_input("Runs in Last 5 Overs",    min_value=0, max_value=120, value=45)
            l5w_s   = st.slider("Wickets in Last 5 Overs",       0, 9, 1)

        if st.button("🎯 PREDICT FINAL SCORE", key="b_score"):
            rr = curr_r / max(ovs_s, 0.1)
            X = np.array([[
                team_enc.get(bat_t, 0),
                team_enc.get(bowl_t, 0),
                venue_enc.get(ven_s, 0),
                curr_r, wkt_s, ovs_s,
                l5r_s, l5w_s, round(rr, 2)
            ]])
            pred = int(np.clip(reg_trained[mdl_r].predict(X)[0], 60, 280))
            crr_ = round(rr, 2)
            rem_ = round(20 - ovs_s, 1)

            st.markdown(f"""
            <div class="result-box">
                <p class="res-label">Predicted Final Score</p>
                <p class="big-score">{pred}</p>
                <p class="team-sub">{bat_t}</p>
                <p style="color:#666;font-size:0.85rem;">Model: {mdl_r}</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            a, b, c = st.columns(3)
            a.metric("Current Run Rate", crr_)
            b.metric("Overs Remaining", rem_)
            c.metric("Runs to Predicted Target", pred - curr_r)

    # ══ TAB 2 — Win Predictor ══
    with tab2:
        st.markdown('<div class="tab-info">Enter live chase conditions to <b>predict the match winner.</b></div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<p class="section-title">MATCH SETUP</p>', unsafe_allow_html=True)
            tm1   = st.selectbox("Team 1 — Chasing",   IPL_TEAMS, key="w_t1")
            tm2   = st.selectbox("Team 2 — Defending", [t for t in IPL_TEAMS if t != tm1], key="w_t2")
            ven_w = st.selectbox("Venue", IPL_VENUES, key="w_ven")
            mdl_c = st.selectbox("ML Model", list(cls_trained.keys()), key="mdl_c")
        with c4:
            st.markdown('<p class="section-title">CHASE CONDITIONS</p>', unsafe_allow_html=True)
            tgt_w    = st.number_input("Target Score",                  min_value=50, max_value=300, value=175)
            cs_w     = st.number_input("Current Score (Chasing Team)", min_value=0,  max_value=300, value=90)
            ov_num_w = st.slider("Overs Completed (0-19)", 0, 19, 10, key="w_ov")
            bl_num_w = st.slider("Balls in Current Over (0-5)", 0, 5, 0, key="w_bl")
            ov_w     = round(ov_num_w + bl_num_w / 6, 4)
            st.caption(f"📌 {ov_num_w}.{bl_num_w} overs = {ov_w:.2f}")
            wk_w    = st.slider("Wickets Fallen", 0, 9, 3, key="w_wk")
            pp_runs = st.number_input("Powerplay Runs (Overs 1–6, 1st innings)", min_value=0, max_value=120, value=48)

        if st.button("🏆 PREDICT WINNER", key="b_win"):
            rrr_  = (tgt_w - cs_w) / max(20 - ov_w, 0.1)
            crr2_ = cs_w / max(ov_w, 0.1)
            X2 = np.array([[
                team_enc.get(tm2, 0),   # batting 1st = defending
                team_enc.get(tm1, 0),   # bowling 1st = chasing
                venue_enc.get(ven_w, 0),
                tgt_w, cs_w, ov_w, wk_w,
                round(rrr_, 2), round(crr2_, 2), pp_runs
            ]])
            pred_c = cls_trained[mdl_c].predict(X2)[0]
            prob   = cls_trained[mdl_c].predict_proba(X2)[0]
            # pred_c=1 means batting-1st team (tm2 here) wins; 0 means chaser wins
            winner = tm2 if pred_c == 1 else tm1
            conf   = round(max(prob)*100, 1)

            st.markdown(f"""
            <div class="result-box">
                <p class="res-label">Predicted Winner</p>
                <p class="winner-name">🏆 {winner}</p>
                <p style="color:#555;font-size:1rem;">Confidence: <b style="color:#d44000;">{conf}%</b></p>
                <p style="color:#666;font-size:0.85rem;">Model: {mdl_c}</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            d, e, f_ = st.columns(3)
            d.metric("Current Run Rate",  round(crr2_, 2))
            e.metric("Required Run Rate", round(rrr_, 2))
            f_.metric("Runs Needed",      tgt_w - cs_w)
            st.markdown(f"**Win Probability — {tm1} (Chasing)**")
            st.progress(int(prob[0]*100))
            st.caption(f"🟠 {tm1} (Chase): {round(prob[0]*100,1)}%   |   ⚫ {tm2} (Defend): {round(prob[1]*100,1)}%")

    # ══ TAB 3 — Model Info ══
    with tab3:
        st.markdown('<p class="section-title">REGRESSION MODELS — Score Prediction (8 Models)</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Model":    list(reg_trained.keys()),
            "Type":     ["Ensemble","Ensemble","Ensemble","Ensemble","Tree","Linear","Linear","Linear"],
            "Features Used": ["9 features"] * 8,
            "Metrics":  ["RMSE, MAE, R²"] * 8,
        }), use_container_width=True, hide_index=True)

        st.markdown("<br>")
        st.markdown('<p class="section-title">CLASSIFICATION MODELS — Win Prediction (8 Models)</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Model":    list(cls_trained.keys()),
            "Type":     ["Ensemble","Ensemble","Ensemble","Ensemble","Tree","Linear","Instance","Kernel"],
            "Features Used": ["10 features"] * 8,
            "Metrics":  ["Accuracy, F1-Score, Precision, Recall"] * 8,
        }), use_container_width=True, hide_index=True)

        st.markdown("<br>")
        g, h = st.columns(2)
        with g:
            st.markdown('<p class="section-title">REGRESSION FEATURES (9)</p>', unsafe_allow_html=True)
            for f in ["Batting Team (encoded)", "Bowling Team (encoded)", "Venue (encoded)",
                      "Runs So Far", "Wickets So Far", "Overs Completed",
                      "Runs in Last 5 Overs", "Wickets in Last 5 Overs", "Current Run Rate (derived)"]:
                st.markdown(f"✅ {f}")
        with h:
            st.markdown('<p class="section-title">CLASSIFICATION FEATURES (10)</p>', unsafe_allow_html=True)
            for f in ["Batting Team 1st (encoded)", "Bowling Team 1st (encoded)", "Venue (encoded)",
                      "Target Score", "Chase Score at Over 10", "Overs at Snapshot",
                      "Wickets at Over 10", "Required Run Rate (derived)", "Current Run Rate (derived)",
                      "Powerplay Runs (1st innings)"]:
                st.markdown(f"✅ {f}")

        st.info(f"📌 Trained on REAL IPL ball-by-ball data — {len(raw_df):,} deliveries from 2008–2024.")

    # ══ TAB 4 — Model Report ══
    with tab4:
        st.markdown('<div class="tab-info">Complete performance report — Training vs Testing for all 16 models on real IPL data.</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-title">REGRESSION — Score Prediction (8 Models)</p>', unsafe_allow_html=True)
        st.markdown(f"**80% Train / 20% Test | {reg_df.shape[0]} real IPL innings snapshots**")
        reg_rows = []
        for name, m in reg_metrics.items():
            reg_rows.append({
                'Model':      name,
                'Train R²':   m['train_r2'],
                'Test R²':    m['test_r2'],
                'Train RMSE': m['train_rmse'],
                'Test RMSE':  m['test_rmse'],
                'Test MAE':   m['test_mae'],
                'Overfit?':   '⚠️ Yes' if m['train_r2'] - m['test_r2'] > 0.1 else '✅ No'
            })
        reg_table = pd.DataFrame(reg_rows)
        best_reg  = reg_table.loc[reg_table['Test R²'].idxmax(), 'Model']
        st.dataframe(reg_table, use_container_width=True, hide_index=True)
        st.success(f"🏆 Best Regression Model: **{best_reg}** (highest Test R²)")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Metric Explanation:**")
            st.markdown("- **R² Score:** 1.0 = perfect, 0 = useless model")
            st.markdown("- **RMSE:** Avg error in runs — lower is better")
            st.markdown("- **MAE:** Mean absolute error — lower is better")
        with r2:
            st.markdown("**Overfitting Check:**")
            st.markdown("- Train R² >> Test R² → overfitting")
            st.markdown("- Ensemble models (RF, GB) usually generalize best")

        st.markdown("---")

        st.markdown('<p class="section-title">CLASSIFICATION — Win Prediction (8 Models)</p>', unsafe_allow_html=True)
        st.markdown(f"**80% Train / 20% Test | {cls_df.shape[0]} real IPL match snapshots**")
        cls_rows = []
        for name, m in cls_metrics.items():
            cls_rows.append({
                'Model':      name,
                'Train Acc':  str(m['train_acc'])+'%',
                'Test Acc':   str(m['test_acc'])+'%',
                'F1-Score':   str(m['f1'])+'%',
                'Precision':  str(m['precision'])+'%',
                'Recall':     str(m['recall'])+'%',
                'Overfit?':   '⚠️ Yes' if m['train_acc'] - m['test_acc'] > 5 else '✅ No'
            })
        cls_table = pd.DataFrame(cls_rows)
        best_acc  = pd.Series({r['Model']: float(r['Test Acc'].replace('%','')) for r in cls_rows})
        best_cls  = best_acc.idxmax()
        st.dataframe(cls_table, use_container_width=True, hide_index=True)
        st.success(f"🏆 Best Classification Model: **{best_cls}** (highest Test Accuracy)")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Metric Explanation:**")
            st.markdown("- **Test Accuracy:** Real performance on unseen matches")
            st.markdown("- **F1-Score:** Balance of Precision & Recall")
        with c2:
            st.markdown("**More Metrics:**")
            st.markdown("- **Precision:** Correct wins predicted / total wins predicted")
            st.markdown("- **Recall:** Correct wins predicted / actual wins")

        st.markdown("---")
        st.markdown('<p class="section-title">BUSINESS RECOMMENDATIONS</p>', unsafe_allow_html=True)
        st.markdown(f"""
**Score Prediction ({best_reg})** can be used by:
- 📺 **Broadcasters** — Real-time projected score overlays
- 💸 **Fantasy Platforms** — Recommend team selections mid-match
- 🎰 **Betting Platforms** — Dynamic odds recalculation

**Win Prediction ({best_cls})** can be used by:
- 📊 **Team Management** — Identify momentum shifts
- 📱 **Fan Engagement Apps** — Live win probability meters
- 🏟️ **Venue Operators** — Crowd flow planning (late-game surges)
        """)

    # ══ TAB 5 — Data Explorer ══
    with tab5:
        st.markdown('<p class="section-title">RAW DATA SAMPLE</p>', unsafe_allow_html=True)
        st.markdown(f"**Dataset:** {len(raw_df):,} deliveries × {raw_df.shape[1]} columns from IPL 2008–2024")
        st.dataframe(raw_df.head(50), use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-title">REGRESSION DATASET (engineered)</p>', unsafe_allow_html=True)
        st.dataframe(reg_df.head(30), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<p class="section-title">CLASSIFICATION DATASET (engineered)</p>', unsafe_allow_html=True)
        st.dataframe(cls_df.head(30), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<p class="section-title">DATASET STATS</p>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Deliveries", f"{len(raw_df):,}")
        s2.metric("Unique Teams",     len(all_teams))
        s3.metric("Unique Venues",    len(all_venues))
        s4.metric("Regression Samples", reg_df.shape[0])

        st.markdown(f"**Win Rate (batting 1st):** {cls_df['winner'].mean()*100:.1f}%")
        st.markdown(f"**Avg Final Score (1st innings):** {reg_df['final_score'].mean():.1f} runs")
        st.markdown(f"**Avg Runs at Over 10:** {reg_df['runs_so_far'].mean():.1f}")

else:
    st.error("Could not load the IPL dataset. Please check the Google Drive link or your internet connection.")

st.markdown("---")
st.markdown('<p style="text-align:center;color:#888;font-size:0.85rem;">ML for Business Analytics — Spring 2026 | FAST NUCES Chiniot-Faisalabad Campus | Real IPL Data 2008–2024</p>', unsafe_allow_html=True)
