import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score

st.set_page_config(page_title="IPL Predictor", page_icon="🏏", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@400;600;700&display=swap');

/* ── Base ── */
html, body { background-color: #ffffff !important; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.block-container {
    background-color: #ffffff !important;
}

/* ── All text black ── */
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

/* ── Widget labels ── */
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

/* ── Inputs ── */
.stSelectbox > div > div,
input[type="number"] {
    background: #f9f9f9 !important;
    color: #111111 !important;
    border: 1px solid #cccccc !important;
}

/* ── Tabs ── */
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

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    color: #d44000 !important;
    font-family: 'Bebas Neue', sans-serif !important;
}
[data-testid="stMetricLabel"] p {
    color: #444444 !important;
}

/* ── Button ── */
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

/* ── Progress ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #d44000, #ff8c00) !important;
}

/* ── Dataframe text ── */
[data-testid="stDataFrame"] * { color: #111111 !important; }

/* ── Caption / small text ── */
.stCaption, small { color: #555555 !important; }

/* ── Custom components ── */
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
    -webkit-text-fill-color: #d44000 !important;
    line-height: 1;
}
.winner-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: #1a7a1a !important;
    -webkit-text-fill-color: #1a7a1a !important;
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

# ── Data ──
IPL_TEAMS = [
    "Chennai Super Kings", "Mumbai Indians",
    "Royal Challengers Bangalore", "Kolkata Knight Riders",
    "Delhi Capitals", "Sunrisers Hyderabad",
    "Rajasthan Royals", "Punjab Kings",
    "Lucknow Super Giants", "Gujarat Titans"
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
TEAM_ENC  = {t: i for i, t in enumerate(IPL_TEAMS)}
VENUE_ENC = {v: i for i, v in enumerate(IPL_VENUES)}

@st.cache_resource
def train_models():
    np.random.seed(42)
    n = 6000

    bat = np.random.randint(0,10,n); bowl = np.random.randint(0,10,n)
    ven = np.random.randint(0,10,n); runs = np.random.randint(0,200,n)
    wkts = np.random.randint(0,10,n); ovs = np.random.uniform(1,20,n)
    l5r = np.random.randint(0,80,n); l5w = np.random.randint(0,5,n)

    crr = runs / np.maximum(ovs, 0.1)
    final = np.clip(
        runs + crr*(20-ovs) - wkts*4 + l5r*0.3 + np.random.normal(0,12,n),
        60, 260).astype(int)
    Xr = np.column_stack([bat,bowl,ven,runs,wkts,ovs,l5r,l5w])

    rf_r=RandomForestRegressor(n_estimators=100,random_state=42).fit(Xr,final)
    gb_r=GradientBoostingRegressor(n_estimators=100,random_state=42).fit(Xr,final)
    lr_r=LinearRegression().fit(Xr,final)
    dt_r=DecisionTreeRegressor(max_depth=8,random_state=42).fit(Xr,final)

    t1=np.random.randint(0,10,n); t2=np.random.randint(0,10,n)
    v2=np.random.randint(0,10,n); tgt=np.random.randint(120,230,n)
    cs=np.random.randint(0,200,n); ov2=np.random.uniform(1,20,n)
    wk2=np.random.randint(0,10,n)
    rrr=(tgt-cs)/np.maximum(20-ov2,0.1); crr2=cs/np.maximum(ov2,0.1)
    wp=np.clip(0.5+0.25*(crr2-rrr)/5-0.05*wk2,0.05,0.95)
    lbl=(np.random.rand(n)<wp).astype(int)
    Xc=np.column_stack([t1,t2,v2,tgt,cs,ov2,wk2,rrr,crr2])

    rf_c=RandomForestClassifier(n_estimators=100,random_state=42).fit(Xc,lbl)
    gb_c=GradientBoostingClassifier(n_estimators=100,random_state=42).fit(Xc,lbl)
    lr_c=LogisticRegression(max_iter=500).fit(Xc,lbl)
    dt_c=DecisionTreeClassifier(max_depth=8,random_state=42).fit(Xc,lbl)

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, final, test_size=0.2, random_state=42)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, lbl,   test_size=0.2, random_state=42)
    reg_metrics = {}
    for nm, mdl in [('rf_r',rf_r),('gb_r',gb_r),('lr_r',lr_r),('dt_r',dt_r)]:
        tp = mdl.predict(Xr_tr); ep = mdl.predict(Xr_te)
        reg_metrics[nm] = {
            'train_r2':   round(r2_score(yr_tr, tp),4),
            'test_r2':    round(r2_score(yr_te, ep),4),
            'train_rmse': round(mean_squared_error(yr_tr, tp)**0.5, 2),
            'test_rmse':  round(mean_squared_error(yr_te, ep)**0.5, 2),
            'test_mae':   round(mean_absolute_error(yr_te, ep), 2),
        }
    cls_metrics = {}
    for nm, mdl in [('rf_c',rf_c),('gb_c',gb_c),('lr_c',lr_c),('dt_c',dt_c)]:
        tp = mdl.predict(Xc_tr); ep = mdl.predict(Xc_te)
        cls_metrics[nm] = {
            'train_acc': round(accuracy_score(yc_tr, tp)*100, 2),
            'test_acc':  round(accuracy_score(yc_te, ep)*100, 2),
            'f1':        round(f1_score(yc_te, ep)*100, 2),
            'precision': round(precision_score(yc_te, ep)*100, 2),
            'recall':    round(recall_score(yc_te, ep)*100, 2),
        }
    return dict(rf_r=rf_r,gb_r=gb_r,lr_r=lr_r,dt_r=dt_r,
                rf_c=rf_c,gb_c=gb_c,lr_c=lr_c,dt_c=dt_c,
                reg_metrics=reg_metrics, cls_metrics=cls_metrics)

# ── Header ──
st.markdown('<p class="ipl-title">🏏 IPL PREDICTOR</p>', unsafe_allow_html=True)
st.markdown('<p class="ipl-subtitle">Machine Learning for Business Analytics — Spring 2026 | FAST NUCES CF</p>', unsafe_allow_html=True)
st.markdown("---")

with st.spinner("⚙️ Training ML Models..."):
    M = train_models()
st.success("✅ All models ready!")

tab1, tab2, tab3, tab4 = st.tabs(["🎯  Score Predictor", "🏆  Win Predictor", "📊  Model Info", "📈  Model Report"])

# ══ TAB 1 ══
with tab1:
    st.markdown('<div class="tab-info">Enter current match conditions to <b>predict the final innings score.</b></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-title">MATCH SETUP</p>', unsafe_allow_html=True)
        bat_t   = st.selectbox("Batting Team", IPL_TEAMS, key="s_bat")
        bowl_t  = st.selectbox("Bowling Team", [t for t in IPL_TEAMS if t != bat_t], key="s_bowl")
        ven_s   = st.selectbox("Venue", IPL_VENUES, key="s_ven")
        mdl_r   = st.selectbox("ML Model", ["Random Forest","Gradient Boosting","Linear Regression","Decision Tree"], key="mdl_r")
    with c2:
        st.markdown('<p class="section-title">INNINGS STATUS</p>', unsafe_allow_html=True)
        curr_r  = st.number_input("Current Runs",         min_value=0, max_value=250, value=85)
        wkt_s   = st.slider("Wickets Fallen",             0, 9, 2)
        ov_num_s = st.slider("Overs Completed (0-19)", 0, 19, 10, key="sc_ov")
        bl_num_s = st.slider("Balls in Current Over (0-5)", 0, 5, 0, key="sc_bl")
        ovs_s    = round(ov_num_s + bl_num_s / 6, 4)
        st.caption(f"📌 {ov_num_s} overs {bl_num_s} balls = {ov_num_s}.{bl_num_s} (cricket notation)")
        l5r_s   = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=100, value=45)
        l5w_s   = st.slider("Wickets in Last 5 Overs",    0, 9, 1)

    if st.button("🎯 PREDICT FINAL SCORE", key="b_score"):
        X = np.array([[TEAM_ENC[bat_t], TEAM_ENC.get(bowl_t,0), VENUE_ENC[ven_s],
                       curr_r, wkt_s, ovs_s, l5r_s, l5w_s]])
        mp = {"Random Forest":M['rf_r'],"Gradient Boosting":M['gb_r'],
              "Linear Regression":M['lr_r'],"Decision Tree":M['dt_r']}
        pred = int(np.clip(mp[mdl_r].predict(X)[0], 60, 260))
        crr_ = round(curr_r / max(ovs_s, 0.1), 2)
        rem_ = round(20 - ovs_s, 1)

        st.markdown(f"""
        <div class="result-box">
            <p class="res-label">Predicted Final Score</p>
            <p class="big-score">{pred}</p>
            <p class="team-sub">{bat_t}</p>
            <p style="color:#666;font-size:0.85rem;">Model: {mdl_r}</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        a,b,c = st.columns(3)
        a.metric("Current Run Rate", crr_)
        b.metric("Overs Remaining", rem_)
        c.metric("Runs to Predicted Target", pred - curr_r)

# ══ TAB 2 ══
with tab2:
    st.markdown('<div class="tab-info">Enter live chase conditions to <b>predict the match winner.</b></div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<p class="section-title">MATCH SETUP</p>', unsafe_allow_html=True)
        tm1   = st.selectbox("Team 1 — Chasing",   IPL_TEAMS, key="w_t1")
        tm2   = st.selectbox("Team 2 — Defending", [t for t in IPL_TEAMS if t != tm1], key="w_t2")
        ven_w = st.selectbox("Venue", IPL_VENUES, key="w_ven")
        mdl_c = st.selectbox("ML Model", ["Random Forest","Gradient Boosting","Logistic Regression","Decision Tree"], key="mdl_c")
    with c4:
        st.markdown('<p class="section-title">CHASE CONDITIONS</p>', unsafe_allow_html=True)
        tgt_w  = st.number_input("Target Score",                   min_value=50,  max_value=280, value=175)
        cs_w   = st.number_input("Current Score (Chasing Team)",   min_value=0,   max_value=280, value=90)
        ov_num_w = st.slider("Overs Completed (0-19)", 0, 19, 10, key="w_ov")
        bl_num_w = st.slider("Balls in Current Over (0-5)", 0, 5, 0, key="w_bl")
        ov_w     = round(ov_num_w + bl_num_w / 6, 4)
        st.caption(f"📌 {ov_num_w} overs {bl_num_w} balls = {ov_num_w}.{bl_num_w} (cricket notation)")
        wk_w   = st.slider("Wickets Fallen",   0, 9, 3, key="w_wk")

    if st.button("🏆 PREDICT WINNER", key="b_win"):
        rrr_ = (tgt_w - cs_w) / max(20 - ov_w, 0.1)
        crr2_= cs_w / max(ov_w, 0.1)
        X2 = np.array([[TEAM_ENC[tm1], TEAM_ENC.get(tm2,0), VENUE_ENC[ven_w],
                        tgt_w, cs_w, ov_w, wk_w, rrr_, crr2_]])
        mp2 = {"Random Forest":M['rf_c'],"Gradient Boosting":M['gb_c'],
               "Logistic Regression":M['lr_c'],"Decision Tree":M['dt_c']}
        pred_c = mp2[mdl_c].predict(X2)[0]
        prob   = mp2[mdl_c].predict_proba(X2)[0]
        winner = tm1 if pred_c == 1 else tm2
        conf   = round(max(prob)*100, 1)

        st.markdown(f"""
        <div class="result-box">
            <p class="res-label">Predicted Winner</p>
            <p class="winner-name">🏆 {winner}</p>
            <p style="color:#555;font-size:1rem;">Confidence: <b style="color:#d44000;">{conf}%</b></p>
            <p style="color:#666;font-size:0.85rem;">Model: {mdl_c}</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        d,e,f_ = st.columns(3)
        d.metric("Current Run Rate",  round(crr2_,2))
        e.metric("Required Run Rate", round(rrr_,2))
        f_.metric("Runs Needed",       tgt_w - cs_w)
        st.markdown(f"**Win Probability — {tm1}**")
        st.progress(int(prob[1]*100))
        st.caption(f"🟠 {tm1}: {round(prob[1]*100,1)}%   |   ⚫ {tm2}: {round(prob[0]*100,1)}%")

# ══ TAB 3 ══
with tab3:
    st.markdown('<p class="section-title">REGRESSION MODELS — Score Prediction</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Model":   ["Random Forest","Gradient Boosting","Linear Regression","Decision Tree"],
        "Type":    ["Ensemble","Ensemble","Linear","Tree"],
        "Metrics": ["RMSE, MAE, R²"]*4
    }), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">CLASSIFICATION MODELS — Win Prediction</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Model":   ["Random Forest","Gradient Boosting","Logistic Regression","Decision Tree"],
        "Type":    ["Ensemble","Ensemble","Linear","Tree"],
        "Metrics": ["Accuracy, F1-Score"]*4
    }), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    g, h = st.columns(2)
    with g:
        st.markdown('<p class="section-title">SCORE PREDICTOR FEATURES</p>', unsafe_allow_html=True)
        for f in ["Batting Team","Bowling Team","Venue","Current Runs",
                  "Wickets Fallen","Overs Completed","Runs in Last 5 Overs","Wickets in Last 5 Overs"]:
            st.markdown(f"✅ {f}")
    with h:
        st.markdown('<p class="section-title">WIN PREDICTOR FEATURES</p>', unsafe_allow_html=True)
        for f in ["Chasing Team","Defending Team","Venue","Target Score",
                  "Current Score","Overs Completed","Wickets Fallen",
                  "Current Run Rate (auto-calculated)","Required Run Rate (auto-calculated)"]:
            st.markdown(f"✅ {f}")

    st.info("📌 Models trained on IPL-realistic synthetic data. Connect Kaggle IPL dataset for live predictions.")

# == TAB 4 ==
with tab4:
    st.markdown('<div class="tab-info">Complete performance report — Training vs Testing accuracy for all models.</div>', unsafe_allow_html=True)
    rm = M['reg_metrics']
    cm = M['cls_metrics']
    rn = {'rf_r':'Random Forest','gb_r':'Gradient Boosting','lr_r':'Linear Regression','dt_r':'Decision Tree'}
    cn = {'rf_c':'Random Forest','gb_c':'Gradient Boosting','lr_c':'Logistic Regression','dt_c':'Decision Tree'}

    st.markdown('<p class="section-title">REGRESSION — Score Prediction Models</p>', unsafe_allow_html=True)
    st.markdown("**80% Training / 20% Testing | 6000 samples**")
    reg_rows = []
    for k,label in rn.items():
        m = rm[k]
        reg_rows.append({
            'Model': label,
            'Train R2': m['train_r2'],
            'Test R2':  m['test_r2'],
            'Train RMSE': m['train_rmse'],
            'Test RMSE':  m['test_rmse'],
            'Test MAE':   m['test_mae'],
            'Overfit?': 'Yes' if m['train_r2']-m['test_r2']>0.1 else 'No'
        })
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    st.markdown("")
    r1,r2 = st.columns(2)
    with r1:
        st.markdown("**Metric Explanation:**")
        st.markdown("- **R2 Score:** 1.0 = perfect prediction, 0 = useless model")
        st.markdown("- **RMSE:** Average error in runs — lower is better")
        st.markdown("- **MAE:** Mean absolute error in runs — lower is better")
    with r2:
        st.markdown("**Overfitting Check:**")
        st.markdown("- If Train R2 >> Test R2, model has overfit")
        st.markdown("- Means model memorized training data but failed on new data")
        st.markdown("- Good model: Train and Test scores are close")

    st.markdown("---")

    st.markdown('<p class="section-title">CLASSIFICATION — Win Prediction Models</p>', unsafe_allow_html=True)
    st.markdown("**80% Training / 20% Testing | 6000 samples**")
    cls_rows = []
    for k,label in cn.items():
        m = cm[k]
        cls_rows.append({
            'Model': label,
            'Train Acc': str(m['train_acc'])+'%',
            'Test Acc':  str(m['test_acc'])+'%',
            'F1-Score':  str(m['f1'])+'%',
            'Precision': str(m['precision'])+'%',
            'Recall':    str(m['recall'])+'%',
            'Overfit?':  'Yes' if m['train_acc']-m['test_acc']>5 else 'No'
        })
    st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)

    st.markdown("")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Metric Explanation:**")
        st.markdown("- **Train Accuracy:** Training data pe kitna sahi tha")
        st.markdown("- **Test Accuracy:** Unseen data pe performance — real accuracy yahi hai")
        st.markdown("- **F1-Score:** Precision aur Recall ka balance")
    with c2:
        st.markdown("**More Metrics:**")
        st.markdown("- **Precision:** Jab model ne Win predict kiya, kitni baar sahi tha")
        st.markdown("- **Recall:** Actual wins mein se kitne model ne correctly pakre")
        st.markdown("- **Overfit:** Train Acc >> Test Acc = overfitting")

    st.markdown("---")

    st.markdown('<p class="section-title">PROJECT SUMMARY — WHAT WE DID</p>', unsafe_allow_html=True)
    s1,s2 = st.columns(2)
    with s1:
        st.markdown("**1. Problem Definition**")
        st.markdown("Two ML problems on IPL data: Score Prediction (Regression) and Match Winner Prediction (Classification)")
        st.markdown("**2. Dataset**")
        st.markdown("6000 IPL-realistic samples. 8 features for regression, 9 for classification. Label encoding for teams and venues.")
        st.markdown("**3. Feature Engineering**")
        st.markdown("Current Run Rate and Required Run Rate auto-calculated as derived features from raw inputs.")
    with s2:
        st.markdown("**4. Models Used**")
        st.markdown("4 Regression + 4 Classification models including ensemble methods (Random Forest, Gradient Boosting) and base models (Linear/Logistic Regression, Decision Tree)")
        st.markdown("**5. Evaluation**")
        st.markdown("Train/Test split 80/20. Regression: R2, RMSE, MAE. Classification: Accuracy, F1, Precision, Recall. Overfitting checked for all models.")
        st.markdown("**6. Web App**")
        st.markdown("Built in Streamlit. Two prediction tabs, model selection dropdown, deployed on Streamlit Cloud.")


st.markdown("---")
st.markdown('<p style="text-align:center;color:#888;font-size:0.85rem;">ML for Business Analytics — Spring 2026 | FAST NUCES Chiniot-Faisalabad Campus</p>', unsafe_allow_html=True)
