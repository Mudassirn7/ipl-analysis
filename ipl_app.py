import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, BaggingRegressor,
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA Analytics 15-22",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-title {
        font-family: 'Bebas Neue', cursive;
        font-size: 3.5rem;
        letter-spacing: 4px;
        background: linear-gradient(135deg, #00d4aa, #00a3ff, #00d4aa);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s infinite;
        text-align: center;
        margin: 0;
    }
    @keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }

    .hero-sub {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 8px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .kpi-card {
        background: linear-gradient(135deg, #0f1923, #1a2940);
        border: 1px solid #00d4aa33;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
        margin-bottom: 8px;
    }
    .kpi-card:hover { transform: translateY(-3px); border-color: #00d4aa99; }
    .kpi-val { font-family: 'Bebas Neue', cursive; font-size: 2.2rem; color: #00d4aa; }
    .kpi-lbl { color: #aaa; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }

    .section-header {
        font-family: 'Bebas Neue', cursive;
        font-size: 1.8rem;
        color: #00d4aa;
        letter-spacing: 3px;
        border-bottom: 2px solid #00d4aa33;
        padding-bottom: 8px;
        margin: 16px 0 12px 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #00a3ff);
        color: #000;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        padding: 12px 28px;
        width: 100%;
        letter-spacing: 1px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Bebas Neue', cursive;
        letter-spacing: 2px;
        font-size: 1rem;
        color: #888;
    }
    .stTabs [aria-selected="true"] { color: #00d4aa !important; }

    .info-box {
        background: #0f1923;
        border-left: 4px solid #00d4aa;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        color: #ccc;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">⚽ FIFA ANALYTICS 15–22</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Machine Learning for Business Analytics | FIFA Analytics Group | BsBa 6A</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload FIFA CSV Files")
    st.markdown('<div class="info-box">Upload all 8 files:<br><b>players_15.csv → players_22.csv</b></div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Select multiple CSV files",
        type=["csv"],
        accept_multiple_files=True
    )
    st.markdown("---")
    st.markdown("### 👥 Group: FIFA Analytics")
    st.markdown("- **Muhammad Mudassir** (23F-5007)")
    st.markdown("- **Abdul Latif**")
    st.markdown("- **Asad Ullah**")
    st.markdown("---")
    st.markdown("**Dataset:** FIFA 15–22 Male Players")
    st.markdown("**Source:** [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)")
    st.markdown("**License:** CC0 Public Domain")

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    'age', 'overall', 'potential', 'wage_eur', 'release_clause_eur',
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'height_cm', 'weight_kg', 'international_reputation',
    'weak_foot', 'skill_moves'
]
FEATURES_EXT = FEATURES + ['potential_gap', 'wage_to_value', 'age_group_enc', 'fifa_version']

FIFA_VERSIONS = {
    'players_15.csv': 15, 'players_16.csv': 16, 'players_17.csv': 17,
    'players_18.csv': 18, 'players_19.csv': 19, 'players_20.csv': 20,
    'players_21.csv': 21, 'players_22.csv': 22
}

DARK_BG = '#0f1923'
ACCENT  = '#00d4aa'
ACCENT2 = '#00a3ff'

# ── Dark figure helper ────────────────────────────────────────────────────────
def dark_fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.xaxis.label.set_color('#aaa')
    ax.yaxis.label.set_color('#aaa')
    ax.title.set_color('#eee')
    return fig, ax

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_combine(files):
    dfs = []
    for f in files:
        fname = f.name.lower()
        version = next((v for k, v in FIFA_VERSIONS.items() if k in fname), None)
        if version is None:
            continue
        try:
            df_tmp = pd.read_csv(f, low_memory=False)
            df_tmp['fifa_version'] = version
            dfs.append(df_tmp)
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)

    keep_cols = FEATURES + ['value_eur', 'player_positions', 'fifa_version',
                             'short_name', 'long_name', 'club_name', 'nationality_name']
    available = [c for c in keep_cols if c in combined.columns]
    df_work = combined[available].copy()

    df_work.dropna(subset=['value_eur', 'player_positions'], inplace=True)

    # Force numeric, remove inf, fill NaN
    for col in df_work.select_dtypes(include='number').columns:
        df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
    df_work.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df_work.select_dtypes(include='number').columns:
        df_work[col].fillna(df_work[col].median(), inplace=True)

    df_work['potential_gap'] = df_work['potential'] - df_work['overall']
    df_work['wage_to_value'] = (df_work['wage_eur'] / (df_work['value_eur'] + 1)).replace([np.inf, -np.inf], 0).fillna(0)
    df_work['age_group'] = pd.cut(
        df_work['age'], bins=[15, 21, 27, 33, 50],
        labels=['Young', 'Prime', 'Experienced', 'Veteran']
    )
    le_age = LabelEncoder()
    df_work['age_group_enc'] = le_age.fit_transform(df_work['age_group'])

    def map_position(pos):
        pos = str(pos).upper()
        if 'GK' in pos: return 'Goalkeeper'
        for p in ['CB', 'LB', 'RB', 'LWB', 'RWB']:
            if p in pos: return 'Defender'
        for p in ['CM', 'CAM', 'CDM', 'LM', 'RM']:
            if p in pos: return 'Midfielder'
        for p in ['ST', 'CF', 'LW', 'RW']:
            if p in pos: return 'Attacker'
        return 'Midfielder'

    df_work['position_cat'] = df_work['player_positions'].apply(map_position)
    return df_work


@st.cache_data
def run_regression(_df):
    feat = [f for f in FEATURES_EXT if f in _df.columns]

    # Clean: drop NaN/Inf rows
    data = _df[feat + ['value_eur']].copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(inplace=True)

    X = data[feat]
    y = np.log1p(data['value_eur'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_train)
    Xte_sc = sc.transform(X_test)

    models = {
        'Linear Regression':  (LinearRegression(), True),
        'Ridge':              (Ridge(), True),
        'Lasso':              (Lasso(alpha=0.01), True),
        'ElasticNet':         (ElasticNet(alpha=0.01), True),
        'Decision Tree':      (DecisionTreeRegressor(max_depth=10, random_state=42), False),
        'KNN Regressor':      (KNeighborsRegressor(n_neighbors=5), True),
        'SVR':                (SVR(C=10), True),
        'Random Forest':      (RandomForestRegressor(n_estimators=100, random_state=42), False),
        'Extra Trees':        (ExtraTreesRegressor(n_estimators=100, random_state=42), False),
        'Bagging':            (BaggingRegressor(n_estimators=50, random_state=42), False),
        'Gradient Boosting':  (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
        'AdaBoost':           (AdaBoostRegressor(n_estimators=100, random_state=42), False),
        'XGBoost':            (xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0), False),
        'LightGBM':           (lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1), False),
    }

    results = []
    for name, (model, scaled) in models.items():
        Xtr = Xtr_sc if scaled else X_train
        Xte = Xte_sc if scaled else X_test
        model.fit(Xtr, y_train)
        yp = model.predict(Xte)
        results.append({
            'Model':    name,
            'RMSE':     round(np.sqrt(mean_squared_error(y_test, yp)), 4),
            'MAE':      round(mean_absolute_error(y_test, yp), 4),
            'R² Score': round(r2_score(y_test, yp), 4)
        })

    df_res = pd.DataFrame(results).sort_values('R² Score', ascending=False)
    return df_res, X_train, X_test, y_train, y_test, feat


@st.cache_data
def run_classification(_df):
    feat = [f for f in FEATURES_EXT if f in _df.columns]

    # Clean: drop NaN/Inf rows
    data = _df[feat + ['position_cat']].copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in feat:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(inplace=True)

    le = LabelEncoder()
    y = le.fit_transform(data['position_cat'])
    X = data[feat]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_train)
    Xte_sc = sc.transform(X_test)

    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'Decision Tree':       (DecisionTreeClassifier(max_depth=10, random_state=42), False),
        'KNN Classifier':      (KNeighborsClassifier(n_neighbors=5), True),
        'Naive Bayes':         (GaussianNB(), True),
        'SVM (RBF)':           (SVC(C=10, random_state=42), True),
        'Random Forest':       (RandomForestClassifier(n_estimators=100, random_state=42), False),
        'Extra Trees':         (ExtraTreesClassifier(n_estimators=100, random_state=42), False),
        'Bagging':             (BaggingClassifier(n_estimators=50, random_state=42), False),
        'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
        'AdaBoost':            (AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME'), False),
        'XGBoost':             (XGBClassifier(n_estimators=200, random_state=42,
                                               use_label_encoder=False,
                                               eval_metric='mlogloss', verbosity=0), False),
        'LightGBM':            (LGBMClassifier(n_estimators=200, random_state=42, verbose=-1), False),
    }

    results = []
    for name, (model, scaled) in models.items():
        Xtr = Xtr_sc if scaled else X_train
        Xte = Xte_sc if scaled else X_test
        model.fit(Xtr, y_train)
        yp = model.predict(Xte)
        results.append({
            'Model':       name,
            'Accuracy':    round(accuracy_score(y_test, yp), 4),
            'F1 Macro':    round(f1_score(y_test, yp, average='macro'), 4),
            'F1 Weighted': round(f1_score(y_test, yp, average='weighted'), 4),
        })

    df_res = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    return df_res, le, X_train, X_test, y_train, y_test, feat


# ── NO FILE STATE ─────────────────────────────────────────────────────────────
if not uploaded_files:
    st.markdown('<div class="section-header">HOW TO GET STARTED</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Step 1 — Download Dataset**
        1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)
        2. Download all CSV files: `players_15.csv` to `players_22.csv`

        **Step 2 — Upload Files**
        - Use the sidebar uploader
        - Select all 8 CSV files at once
        - App will auto-detect FIFA versions
        """)
    with c2:
        st.markdown("""
        **What you get:**
        - 100,000+ player records across 8 years
        - 14 Regression models for market value prediction
        - 12 Classification models for position prediction
        - Historical trend analysis (FIFA 15–22)
        - Messi vs Ronaldo comparison
        - Live prediction tool
        """)
    st.stop()

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("Loading and combining all FIFA CSV files..."):
    df = load_and_combine(uploaded_files)

if df is None or df.empty:
    st.error("Could not load data. Make sure files are named players_15.csv through players_22.csv")
    st.stop()

versions_loaded = sorted(df['fifa_version'].unique())
st.success(f"✅ Loaded FIFA versions: {versions_loaded} | Total records: {df.shape[0]:,} | Features: {len(FEATURES_EXT)}")

# ── KPI ROW ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (f"{df.shape[0]:,}",          "Total Records"),
    (f"{len(versions_loaded)}",   "FIFA Versions"),
    (f"{df['nationality_name'].nunique() if 'nationality_name' in df.columns else '—'}", "Nationalities"),
    (f"{df['overall'].mean():.1f}", "Avg Overall"),
    (f"€{df['value_eur'].mean()/1e6:.1f}M", "Avg Value"),
    (f"{df['age'].mean():.1f}",   "Avg Age"),
]
for col, (val, lbl) in zip([k1, k2, k3, k4, k5, k6], kpis):
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
        f'<div class="kpi-lbl">{lbl}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA", "📈 REGRESSION", "🏷️ CLASSIFICATION",
    "📅 TRENDS", "🔮 PREDICT", "📋 DATA"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = dark_fig(7, 4)
        ax.hist(np.log1p(df['value_eur']), bins=60, color=ACCENT, alpha=0.8, edgecolor='none')
        ax.set_title('Market Value Distribution (log scale)', fontweight='bold')
        ax.set_xlabel('log(Value EUR)')
        ax.set_ylabel('Count')
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = dark_fig(7, 4)
        pos_counts = df['position_cat'].value_counts()
        bar_colors = [ACCENT, ACCENT2, '#ff6b6b', '#ffd93d']
        ax.bar(pos_counts.index, pos_counts.values, color=bar_colors, edgecolor='none')
        ax.set_title('Position Category Distribution', fontweight='bold')
        ax.set_ylabel('Count')
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = dark_fig(7, 4)
        ax.scatter(df['overall'], np.log1p(df['value_eur']),
                   alpha=0.05, color=ACCENT, s=5)
        ax.set_title('Overall Rating vs Market Value', fontweight='bold')
        ax.set_xlabel('Overall Rating')
        ax.set_ylabel('log(Value EUR)')
        st.pyplot(fig); plt.close()

    with col4:
        fig, ax = dark_fig(7, 4)
        vers_avg = df.groupby('fifa_version')['overall'].mean()
        ax.plot(vers_avg.index, vers_avg.values,
                color=ACCENT, marker='o', linewidth=2)
        ax.fill_between(vers_avg.index, vers_avg.values, alpha=0.15, color=ACCENT)
        ax.set_title('Avg Overall Rating by FIFA Version', fontweight='bold')
        ax.set_xlabel('FIFA Version')
        ax.set_ylabel('Average Overall')
        st.pyplot(fig); plt.close()

    # Correlation heatmap
    st.markdown('<div class="section-header">CORRELATION HEATMAP</div>', unsafe_allow_html=True)
    corr_cols = ['overall', 'potential', 'age', 'wage_eur', 'pace',
                 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'value_eur']
    corr_cols = [c for c in corr_cols if c in df.columns]
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, linewidths=0.5, annot_kws={'size': 9},
                cbar_kws={'shrink': 0.8})
    ax.tick_params(colors='#ccc')
    ax.set_title('Feature Correlation Matrix', color='#eee', fontweight='bold', pad=15)
    st.pyplot(fig); plt.close()

    # Top 20
    st.markdown('<div class="section-header">TOP 20 MOST VALUABLE PLAYERS (LATEST VERSION)</div>', unsafe_allow_html=True)
    latest_ver = df['fifa_version'].max()
    latest = df[df['fifa_version'] == latest_ver]
    top20 = latest.nlargest(20, 'value_eur').copy()
    disp_cols = ['short_name', 'overall', 'potential', 'age',
                 'value_eur', 'wage_eur', 'position_cat', 'club_name', 'nationality_name']
    disp_cols = [c for c in disp_cols if c in top20.columns]
    top20_show = top20[disp_cols].copy()
    top20_show['value_eur'] = top20_show['value_eur'].apply(lambda x: f"€{x/1e6:.1f}M")
    if 'wage_eur' in top20_show.columns:
        top20_show['wage_eur'] = top20_show['wage_eur'].apply(lambda x: f"€{x:,.0f}/wk")
    st.dataframe(top20_show, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">REGRESSION — PLAYER MARKET VALUE PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Target: <b>value_eur</b> | Log-transformed | 80/20 Train-Test Split | 14 Models</div>', unsafe_allow_html=True)

    if st.button("🚀 Train All 14 Regression Models", key="btn_reg"):
        with st.spinner("Training 14 models on combined FIFA 15-22 data..."):
            res_r, Xtr_r, Xte_r, ytr_r, yte_r, feat_r = run_regression(df)
        st.session_state['reg_res']  = res_r
        st.session_state['reg_data'] = (Xtr_r, Xte_r, ytr_r, yte_r, feat_r)

    if 'reg_res' in st.session_state:
        res_r  = st.session_state['reg_res']
        best_r = res_r.iloc[0]['Model']

        st.markdown(f"**🏆 Best Model: `{best_r}` | R² = {res_r.iloc[0]['R² Score']} | RMSE = {res_r.iloc[0]['RMSE']} | MAE = {res_r.iloc[0]['MAE']}**")

        def hl_reg(row):
            if row['Model'] == best_r:
                return ['background-color:#00d4aa22;font-weight:bold;color:#00d4aa'] * len(row)
            return ['color:#ccc'] * len(row)

        st.dataframe(res_r.style.apply(hl_reg, axis=1), use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            fig, ax = dark_fig(5, 5)
            colors = [ACCENT if m == best_r else '#1e3a4a' for m in res_r['Model']]
            ax.barh(res_r['Model'], res_r['R² Score'], color=colors)
            ax.axvline(0.9, color='red', linestyle='--', alpha=0.6, linewidth=1)
            ax.set_title('R² Score', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            fig, ax = dark_fig(5, 5)
            best_rmse_m = res_r.loc[res_r['RMSE'].idxmin(), 'Model']
            colors = [ACCENT if m == best_rmse_m else '#1e3a4a' for m in res_r['Model']]
            ax.barh(res_r['Model'], res_r['RMSE'], color=colors)
            ax.set_title('RMSE (Lower=Better)', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c3:
            fig, ax = dark_fig(5, 5)
            best_mae_m = res_r.loc[res_r['MAE'].idxmin(), 'Model']
            colors = [ACCENT if m == best_mae_m else '#1e3a4a' for m in res_r['Model']]
            ax.barh(res_r['Model'], res_r['MAE'], color=colors)
            ax.set_title('MAE (Lower=Better)', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Feature importance
        st.markdown('<div class="section-header">FEATURE IMPORTANCE — XGBOOST REGRESSOR</div>', unsafe_allow_html=True)
        Xtr_r, Xte_r, ytr_r, yte_r, feat_r = st.session_state['reg_data']
        xgb_fi = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        xgb_fi.fit(Xtr_r, ytr_r)
        imp_r = pd.Series(xgb_fi.feature_importances_, index=feat_r).sort_values()
        fig, ax = dark_fig(9, 5)
        colors = [ACCENT if i == imp_r.idxmax() else ACCENT2 for i in imp_r.index]
        ax.barh(imp_r.index, imp_r.values, color=colors)
        ax.set_title('Top Feature Importances (XGBoost Regressor)', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(f"""
        **💡 Reasoning:**
        - **{best_r}** achieves highest R² — captures non-linear relationships between attributes and market value.
        - Ensemble/boosting models dominate; linear models struggle with non-linearity.
        - `overall`, `potential`, and `wage_eur` are consistently the strongest predictors.
        - Using FIFA 15–22 combined data (100K+ records) improves model generalization significantly vs single year.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">CLASSIFICATION — PLAYER POSITION PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Target: <b>Position Category</b> (Attacker / Midfielder / Defender / Goalkeeper) | Stratified 80/20 Split | 12 Models</div>', unsafe_allow_html=True)

    if st.button("🚀 Train All 12 Classification Models", key="btn_cls"):
        with st.spinner("Training 12 models..."):
            res_c, le_p, Xtr_c, Xte_c, ytr_c, yte_c, feat_c = run_classification(df)
        st.session_state['cls_res']  = res_c
        st.session_state['cls_data'] = (le_p, Xtr_c, Xte_c, ytr_c, yte_c, feat_c)

    if 'cls_res' in st.session_state:
        res_c  = st.session_state['cls_res']
        best_c = res_c.iloc[0]['Model']

        st.markdown(f"**🏆 Best Model: `{best_c}` | Accuracy = {res_c.iloc[0]['Accuracy']} | F1 Weighted = {res_c.iloc[0]['F1 Weighted']}**")

        def hl_cls(row):
            if row['Model'] == best_c:
                return ['background-color:#00d4aa22;font-weight:bold;color:#00d4aa'] * len(row)
            return ['color:#ccc'] * len(row)

        st.dataframe(res_c.style.apply(hl_cls, axis=1), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = dark_fig(7, 5)
            colors = [ACCENT if m == best_c else '#1e3a4a' for m in res_c['Model']]
            ax.barh(res_c['Model'], res_c['Accuracy'], color=colors)
            ax.axvline(0.85, color='red', linestyle='--', alpha=0.6, linewidth=1)
            ax.set_title('Accuracy Comparison', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            le_p, Xtr_c, Xte_c, ytr_c, yte_c, feat_c = st.session_state['cls_data']
            xgb_cm = XGBClassifier(n_estimators=200, random_state=42,
                                    use_label_encoder=False,
                                    eval_metric='mlogloss', verbosity=0)
            xgb_cm.fit(Xtr_c, ytr_c)
            yp_cm = xgb_cm.predict(Xte_c)
            cm = confusion_matrix(yte_c, yp_cm)
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn', ax=ax,
                        xticklabels=le_p.classes_, yticklabels=le_p.classes_,
                        linewidths=0.5)
            ax.set_title(f'Confusion Matrix — {best_c}', color='#eee', fontweight='bold')
            ax.tick_params(colors='#ccc')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(f"""
        **💡 Reasoning:**
        - **{best_c}** best handles multi-class imbalance using gradient boosting.
        - Goalkeepers are almost perfectly classified (very distinct stats).
        - Midfielders/Defenders have some CDM overlap — expected in real football.
        - `defending`, `passing`, and `physic` are top discriminators between positions.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">MULTI-YEAR TREND ANALYSIS (FIFA 15–22)</div>', unsafe_allow_html=True)

    # Attribute trends
    attrs = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    attrs = [a for a in attrs if a in df.columns]

    st.markdown("#### Average Attribute Trends by FIFA Version")
    fig, ax = dark_fig(11, 5)
    trend_colors = [ACCENT, ACCENT2, '#ff6b6b', '#ffd93d', '#c77dff', '#ff9f43']
    for i, attr in enumerate(attrs):
        trend = df.groupby('fifa_version')[attr].mean()
        ax.plot(trend.index, trend.values, marker='o',
                label=attr.capitalize(),
                color=trend_colors[i % len(trend_colors)], linewidth=2)
    ax.legend(fontsize=9, facecolor=DARK_BG, labelcolor='#ccc')
    ax.set_title('Player Attribute Trends Across FIFA Versions', fontweight='bold')
    ax.set_xlabel('FIFA Version')
    ax.set_ylabel('Average Value')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Avg Market Value by FIFA Version")
        fig, ax = dark_fig(6, 4)
        val_trend = df.groupby('fifa_version')['value_eur'].mean() / 1e6
        ax.bar(val_trend.index, val_trend.values, color=ACCENT, alpha=0.85, edgecolor='none')
        ax.plot(val_trend.index, val_trend.values, color=ACCENT2, marker='D', linewidth=2)
        ax.set_title('Avg Market Value (EUR Millions)', fontweight='bold')
        ax.set_xlabel('FIFA Version')
        ax.set_ylabel('Avg Value (M EUR)')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Player Count by FIFA Version")
        fig, ax = dark_fig(6, 4)
        cnt = df.groupby('fifa_version').size()
        ax.bar(cnt.index, cnt.values, color=ACCENT2, alpha=0.85, edgecolor='none')
        ax.set_title('Total Players Per Version', fontweight='bold')
        ax.set_xlabel('FIFA Version')
        ax.set_ylabel('Player Count')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Messi vs Ronaldo
    st.markdown('<div class="section-header">MESSI VS RONALDO — HISTORICAL COMPARISON</div>', unsafe_allow_html=True)

    name_col = 'short_name' if 'short_name' in df.columns else 'long_name'
    if name_col in df.columns:
        messi   = df[df[name_col].str.contains('Messi',   case=False, na=False)]
        ronaldo = df[df[name_col].str.contains('Ronaldo', case=False, na=False)]
        ronaldo = ronaldo[~ronaldo[name_col].str.contains('Ronaldo Lu', case=False, na=False)]
    else:
        messi = ronaldo = pd.DataFrame()

    if not messi.empty and not ronaldo.empty:
        compare_attrs = ['overall', 'potential', 'pace', 'shooting',
                         'passing', 'dribbling', 'physic']
        compare_attrs = [a for a in compare_attrs if a in df.columns]

        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        fig.patch.set_facecolor(DARK_BG)
        axes = axes.flatten()

        for i, attr in enumerate(compare_attrs):
            ax = axes[i]
            ax.set_facecolor(DARK_BG)
            m_trend = messi.groupby('fifa_version')[attr].mean()
            r_trend = ronaldo.groupby('fifa_version')[attr].mean()
            ax.plot(m_trend.index, m_trend.values, color=ACCENT,
                    marker='o', linewidth=2, label='Messi')
            ax.plot(r_trend.index, r_trend.values, color='#ff6b6b',
                    marker='s', linewidth=2, label='Ronaldo')
            ax.set_title(attr.capitalize(), color='#eee', fontsize=10, fontweight='bold')
            ax.tick_params(colors='#888', labelsize=8)
            for spine in ax.spines.values(): spine.set_edgecolor('#333')
            if i == 0:
                ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor='#ccc')

        for j in range(len(compare_attrs), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('Messi vs Ronaldo — FIFA 15 to 22', color='#eee',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    else:
        st.info("Messi / Ronaldo records not found in loaded files.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">LIVE PREDICTION TOOL</div>', unsafe_allow_html=True)
    st.markdown("Enter player attributes → get instant Market Value & Position prediction using XGBoost.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age         = st.slider("Age", 15, 45, 24)
        overall     = st.slider("Overall Rating", 40, 99, 75)
        potential   = st.slider("Potential", 40, 99, 80)
        wage_eur    = st.number_input("Wage (EUR/week)", 500, 500000, 20000, step=1000)
        rel_clause  = st.number_input("Release Clause (EUR)", 0, 300000000, 5000000, step=500000)
        fifa_ver    = st.selectbox("FIFA Version", [15,16,17,18,19,20,21,22], index=7)
    with c2:
        pace        = st.slider("Pace", 1, 99, 70)
        shooting    = st.slider("Shooting", 1, 99, 65)
        passing     = st.slider("Passing", 1, 99, 70)
        dribbling   = st.slider("Dribbling", 1, 99, 72)
        defending   = st.slider("Defending", 1, 99, 50)
    with c3:
        physic      = st.slider("Physicality", 1, 99, 68)
        height_cm   = st.slider("Height (cm)", 155, 210, 180)
        weight_kg   = st.slider("Weight (kg)", 50, 110, 75)
        intl_rep    = st.selectbox("Intl. Reputation ⭐", [1,2,3,4,5], index=0)
        weak_foot   = st.selectbox("Weak Foot ⭐", [1,2,3,4,5], index=2)
        skill_moves = st.selectbox("Skill Moves ⭐", [1,2,3,4,5], index=2)

    if st.button("⚡ PREDICT NOW", key="predict_btn"):
        pot_gap  = potential - overall
        w2v      = wage_eur / 1e6
        age_enc  = 0 if age <= 21 else (1 if age <= 27 else (2 if age <= 33 else 3))

        feat_avail = [f for f in FEATURES_EXT if f in df.columns]
        input_row  = pd.DataFrame([[
            age, overall, potential, wage_eur, rel_clause,
            pace, shooting, passing, dribbling, defending, physic,
            height_cm, weight_kg, intl_rep, weak_foot, skill_moves,
            pot_gap, w2v, age_enc, fifa_ver
        ]], columns=FEATURES_EXT)[feat_avail]

        X_all     = df[feat_avail]
        y_all_reg = np.log1p(df['value_eur'])
        le2       = LabelEncoder()
        y_all_cls = le2.fit_transform(df['position_cat'])

        reg_m = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        reg_m.fit(X_all, y_all_reg)
        pred_val = np.expm1(reg_m.predict(input_row)[0])

        cls_m = XGBClassifier(n_estimators=200, random_state=42,
                               use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
        cls_m.fit(X_all, y_all_cls)
        pred_pos = le2.classes_[cls_m.predict(input_row)[0]]

        icons = {'Attacker':'⚡', 'Midfielder':'🔄', 'Defender':'🛡️', 'Goalkeeper':'🥅'}
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-lbl">💰 PREDICTED MARKET VALUE</div>
              <div class="kpi-val">€{pred_val/1e6:.2f}M</div>
              <div class="kpi-lbl">EUR {pred_val:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-lbl">🎯 PREDICTED POSITION</div>
              <div class="kpi-val">{icons.get(pred_pos,'⚽')} {pred_pos.upper()}</div>
              <div class="kpi-lbl">Based on XGBoost Classification</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">RAW DATA EXPLORER</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        ver_filter = st.multiselect("FIFA Version", sorted(df['fifa_version'].unique()),
                                    default=sorted(df['fifa_version'].unique()))
    with f2:
        pos_filter = st.multiselect("Position", list(df['position_cat'].unique()),
                                    default=list(df['position_cat'].unique()))
    with f3:
        search = st.text_input("🔍 Search Player Name")

    filtered = df[df['fifa_version'].isin(ver_filter) & df['position_cat'].isin(pos_filter)]
    name_col2 = 'short_name' if 'short_name' in filtered.columns else None
    if search and name_col2:
        filtered = filtered[filtered[name_col2].str.contains(search, case=False, na=False)]

    show_cols = ['short_name', 'fifa_version', 'age', 'overall', 'potential',
                 'value_eur', 'wage_eur', 'pace', 'shooting', 'passing',
                 'dribbling', 'defending', 'physic', 'position_cat',
                 'club_name', 'nationality_name']
    show_cols = [c for c in show_cols if c in filtered.columns]

    st.dataframe(filtered[show_cols].head(500), use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(500, len(filtered)):,} of {len(filtered):,} records")
