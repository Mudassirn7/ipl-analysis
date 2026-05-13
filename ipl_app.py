import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os

st.set_page_config(
    page_title="IPL ML Predictor",
    layout="wide"
)

st.title("🏏 IPL ML Predictor")

warnings.filterwarnings("ignore")

# =========================
# Dataset Auto Download
# =========================

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

        st.success("✅ Dataset downloaded successfully!")

    except Exception as e:
        st.error(f"Dataset download failed: {e}")
        st.stop()

if not os.path.exists(CSV_FILE):
    st.error("IPL.csv not found after download.")
    st.stop()


# =========================
# Load Dataset
# =========================

@st.cache_data
def load_data():
    return pd.read_csv(CSV_FILE)

raw_df = load_data()

st.success("✅ Dataset Loaded Successfully")
st.write(f"Total Rows: {len(raw_df):,}")
st.write(f"Total Columns: {raw_df.shape[1]}")


import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
    print('✅ XGBoost available')
except ImportError:
    XGBOOST_AVAILABLE = False
    print('⚠️  XGBoost not installed — run: pip install xgboost')

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')
print('✅ All libraries imported successfully')




print('✅ Dataset Loaded Successfully')
print(f'Total Rows    : {len(raw_df):,}')
print(f'Total Columns : {raw_df.shape[1]}')
print(f'Columns       : {raw_df.columns.tolist()}')
raw_df.head()

print('=== Basic Info ===')
print(f'Duplicate Rows : {raw_df.duplicated().sum()}')

null_counts = raw_df.isnull().sum()
null_cols   = null_counts[null_counts > 0]

if len(null_cols) > 0:
    print('\nColumns with Null Values:')
    print(null_cols)
else:
    print('✅ No null values found')

print('\n=== Data Types ===')
print(raw_df.dtypes)

TEAM_NAME_MAPPING = {
    'Delhi Daredevils'  : 'Delhi Capitals',
    'Kings XI Punjab'   : 'Punjab Kings',
    'Deccan Chargers'   : 'Sunrisers Hyderabad',
    'Gujarat Lions'     : 'Gujarat Titans'
}

raw_df['batting_team'] = raw_df['batting_team'].replace(TEAM_NAME_MAPPING)
raw_df['bowling_team'] = raw_df['bowling_team'].replace(TEAM_NAME_MAPPING)

print('✅ Team names standardized')
print(f'Mappings applied       : {len(TEAM_NAME_MAPPING)}')
print(f'Unique teams after fix : {raw_df["batting_team"].nunique()}')

VENUE_MAPPING = {
    'Arun Jaitley Stadium, Delhi'                 : 'Arun Jaitley Stadium',
    'M Chinnaswamy Stadium'                       : 'M. Chinnaswamy Stadium',
    'M.Chinnaswamy Stadium'                       : 'M. Chinnaswamy Stadium',
    'MA Chidambaram Stadium'                      : 'M. A. Chidambaram Stadium',
    'MA Chidambaram Stadium, Chepauk'             : 'M. A. Chidambaram Stadium',
    'Punjab Cricket Association Stadium'          : 'PCA Stadium',
    'Punjab Cricket Association IS Bindra Stadium': 'PCA Stadium',
    'Rajiv Gandhi International Stadium'          : 'Rajiv Gandhi Intl. Cricket Stadium',
    'Rajiv Gandhi Intl. Cricket Stadium'          : 'Rajiv Gandhi Intl. Cricket Stadium'
}

raw_df['venue'] = raw_df['venue'].replace(VENUE_MAPPING)

print('✅ Venue names standardized')
print(f'Mappings applied : {len(VENUE_MAPPING)}')
print(f'Unique venues    : {raw_df["venue"].nunique()}')

IPL_TEAMS = [
    'Chennai Super Kings', 'Mumbai Indians',
    'Royal Challengers Bangalore', 'Kolkata Knight Riders',
    'Delhi Capitals', 'Sunrisers Hyderabad',
    'Rajasthan Royals', 'Punjab Kings',
    'Lucknow Super Giants', 'Gujarat Titans'
]

df = raw_df[
    (raw_df['batting_team'].isin(IPL_TEAMS)) &
    (raw_df['bowling_team'].isin(IPL_TEAMS))
].copy()

print(f'✅ Filtered to current 10 IPL teams')
print(f'Rows after filter : {len(df):,}')
print(f'Unique matches    : {df["match_id"].nunique()}')

# Exact decimal overs (e.g. over 10 ball 3 = 10.5)
df['overs_completed'] = df['over'] + (df['ball'] / 6)

# Current Run Rate = runs scored / overs bowled
df['current_run_rate'] = df['team_runs'] / df['overs_completed'].replace(0, 0.1)

# Final innings score — regression label
df['final_score'] = df.groupby(['match_id', 'innings'])['team_runs'].transform('max')

print('✅ New features created:')
print('  • overs_completed  — decimal over count (e.g. 10.3)')
print('  • current_run_rate — runs per over at this point')
print('  • final_score      — total innings runs (regression target)')
print()
df[['overs_completed', 'current_run_rate', 'final_score']].describe().round(2)

IPL_VENUES = sorted(df['venue'].dropna().unique().tolist())

TEAM_ENC  = {team: idx  for idx, team  in enumerate(IPL_TEAMS)}
VENUE_ENC = {venue: idx for idx, venue in enumerate(IPL_VENUES)}
TEAM_DEC  = {v: k for k, v in TEAM_ENC.items()}

print('✅ Label encodings created (integer encoding)')
print(f'Teams encoded  : {len(TEAM_ENC)}')
print(f'Venues encoded : {len(VENUE_ENC)}')
print()
print('Team → Index:')
for k, v in TEAM_ENC.items():
    print(f'  {v:2d} → {k}')

score_df = df[
    (df['overs_completed'] >= 6) &
    (df['overs_completed'] <= 16)
][[
    'batting_team', 'bowling_team', 'venue',
    'team_runs', 'team_wicket', 'overs_completed',
    'current_run_rate', 'final_score'
]].dropna().copy()

score_df.columns = [
    'batting_team', 'bowling_team', 'venue',
    'current_runs', 'wickets', 'overs', 'crr', 'final_score'
]

score_df['batting_team'] = score_df['batting_team'].map(TEAM_ENC)
score_df['bowling_team'] = score_df['bowling_team'].map(TEAM_ENC)
score_df['venue']        = score_df['venue'].map(VENUE_ENC)
score_df = score_df.dropna()

print(f'✅ Regression dataset ready')
print(f'Shape    : {score_df.shape}')
print(f'Features : batting_team, bowling_team, venue, current_runs, wickets, overs, crr')
print(f'Target   : final_score')
score_df.describe().round(2)

# 1st innings totals as targets
innings1 = df[df['innings'] == 1]
targets  = innings1.groupby('match_id')['team_runs'].max().reset_index()
targets.columns = ['match_id', 'target']

# Actual result from last ball of match
match_results = df[df['innings'] == 2].copy()
match_results = match_results.merge(targets, on='match_id')
match_end = match_results.sort_values(['match_id', 'over', 'ball']) \
                         .groupby('match_id').last().reset_index()
match_end['won_chase'] = (match_end['team_runs'] >= match_end['target']).astype(int)
winner_lookup = match_end[['match_id', 'won_chase']]

# Mid-game snapshots: overs 6–18 of 2nd innings
win_df = df[
    (df['innings'] == 2) &
    (df['overs_completed'] >= 6) &
    (df['overs_completed'] <= 18)
].copy()

win_df = win_df.merge(targets, on='match_id')
win_df = win_df.merge(winner_lookup, on='match_id')

win_df['balls_left']        = 120 - ((win_df['over'] * 6) + win_df['ball'])
win_df['balls_left']        = win_df['balls_left'].replace(0, 1)
win_df['runs_needed']       = win_df['target'] - win_df['team_runs']
win_df['required_run_rate'] = win_df['runs_needed'] * 6 / win_df['balls_left']
win_df['pct_target_done']   = win_df['team_runs'] / win_df['target'].replace(0, 1)
win_df['pct_overs_done']    = win_df['overs_completed'] / 20

win_df = win_df[[
    'batting_team', 'bowling_team', 'venue',
    'target', 'team_wicket', 'overs_completed',
    'current_run_rate', 'required_run_rate',
    'pct_target_done', 'pct_overs_done', 'won_chase'
]].dropna().copy()

win_df['batting_team'] = win_df['batting_team'].map(TEAM_ENC)
win_df['bowling_team'] = win_df['bowling_team'].map(TEAM_ENC)
win_df['venue']        = win_df['venue'].map(VENUE_ENC)
win_df = win_df.dropna()

print(f'✅ Classification dataset ready')
print(f'Shape           : {win_df.shape}')
print(f'Chase Win Rate  : {round(win_df["won_chase"].mean()*100, 1)}%')
print(f'Features        : batting_team, bowling_team, venue, target, team_wicket,')
print(f'                  overs_completed, CRR, RRR, pct_target_done, pct_overs_done')
print(f'Target          : won_chase  (0 = defending team wins, 1 = chasing team wins)')

# Regression split
Xr = score_df[['batting_team','bowling_team','venue','current_runs','wickets','overs','crr']]
yr = score_df['final_score']
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

# Classification split
Xc = win_df[['batting_team','bowling_team','venue','target','team_wicket',
             'overs_completed','current_run_rate','required_run_rate',
             'pct_target_done','pct_overs_done']]
yc = win_df['won_chase']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

print('✅ Train/Test Split done (80% train | 20% test | random_state=42)')
print(f'Regression     — Train: {len(Xr_train):,}  |  Test: {len(Xr_test):,}')
print(f'Classification — Train: {len(Xc_train):,}  |  Test: {len(Xc_test):,}')

REG_MODELS = {
    'Random Forest'     : RandomForestRegressor(n_estimators=100, max_depth=8,
                                                min_samples_leaf=10, random_state=42),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                                    learning_rate=0.1, random_state=42),
    'AdaBoost'          : AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'KNN'               : KNeighborsRegressor(n_neighbors=10),
    'Ridge Regression'  : Ridge(alpha=1.0),
    'Linear Regression' : LinearRegression(),
    'Decision Tree'     : DecisionTreeRegressor(max_depth=6, min_samples_leaf=15, random_state=42)
}

if XGBOOST_AVAILABLE:
    REG_MODELS['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=4,
                                          learning_rate=0.1, random_state=42, verbosity=0)

for name, model in REG_MODELS.items():
    model.fit(Xr_train, yr_train)
    print(f'✅ {name} trained')

print(f'\nTotal regression models trained: {len(REG_MODELS)}')

CLS_MODELS = {
    'Random Forest'       : RandomForestClassifier(n_estimators=100, max_depth=6,
                                                   min_samples_leaf=15, random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                       learning_rate=0.1, random_state=42),
    'AdaBoost'            : AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'KNN'                 : KNeighborsClassifier(n_neighbors=10),
    'Logistic Regression' : LogisticRegression(max_iter=1000),
    'Decision Tree'       : DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
}

if XGBOOST_AVAILABLE:
    CLS_MODELS['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                           random_state=42, verbosity=0, eval_metric='logloss')

for name, model in CLS_MODELS.items():
    model.fit(Xc_train, yc_train)
    print(f'✅ {name} trained')

print(f'\nTotal classification models trained: {len(CLS_MODELS)}')

reg_metrics = {}

for name, model in REG_MODELS.items():
    train_pred = model.predict(Xr_train)
    test_pred  = model.predict(Xr_test)
    train_r2   = r2_score(yr_train, train_pred)
    test_r2    = r2_score(yr_test,  test_pred)

    reg_metrics[name] = {
        'Train R²' : round(train_r2, 4),
        'Test R²'  : round(test_r2,  4),
        'RMSE'     : round(np.sqrt(mean_squared_error(yr_test, test_pred)), 2),
        'MAE'      : round(mean_absolute_error(yr_test, test_pred), 2),
        'Overfit'  : 'Yes' if abs(train_r2 - test_r2) > 0.10 else 'No'
    }

reg_report = pd.DataFrame(reg_metrics).T.sort_values('Test R²', ascending=False)
print('📊 Regression Model Report (sorted by Test R²):')
reg_report

cls_metrics = {}

for name, model in CLS_MODELS.items():
    train_pred = model.predict(Xc_train)
    test_pred  = model.predict(Xc_test)
    train_acc  = accuracy_score(yc_train, train_pred) * 100
    test_acc   = accuracy_score(yc_test,  test_pred)  * 100

    cls_metrics[name] = {
        'Train Acc %' : round(train_acc, 2),
        'Test Acc %'  : round(test_acc,  2),
        'Precision %' : round(precision_score(yc_test, test_pred, zero_division=0) * 100, 2),
        'Recall %'    : round(recall_score(yc_test,    test_pred, zero_division=0) * 100, 2),
        'F1 Score %'  : round(f1_score(yc_test,        test_pred, zero_division=0) * 100, 2),
        'Overfit'     : 'Yes' if abs(train_acc - test_acc) > 8 else 'No'
    }

cls_report = pd.DataFrame(cls_metrics).T.sort_values('Test Acc %', ascending=False)
print('🏆 Classification Model Report (sorted by Test Accuracy):')
cls_report

models_r  = list(reg_metrics.keys())
train_r2s = [reg_metrics[m]['Train R²'] for m in models_r]
test_r2s  = [reg_metrics[m]['Test R²']  for m in models_r]
rmses     = [reg_metrics[m]['RMSE']     for m in models_r]
maes      = [reg_metrics[m]['MAE']      for m in models_r]

x     = np.arange(len(models_r))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Regression Model Comparison', fontsize=14, fontweight='bold')

# R² comparison
axes[0].bar(x - width/2, train_r2s, width, label='Train R²', color='#1f77b4')
axes[0].bar(x + width/2, test_r2s,  width, label='Test R²',  color='#ff7f0e')
axes[0].set_title('R² Score (higher is better)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_r, rotation=45, ha='right', fontsize=8)
axes[0].set_ylim(0, 1.15)
axes[0].legend()
axes[0].set_ylabel('R² Score')
for i, v in enumerate(test_r2s):
    axes[0].text(i + width/2, v + 0.01, str(v), ha='center', fontsize=7)

# RMSE comparison
bars1 = axes[1].bar(models_r, rmses, color='#2ca02c')
axes[1].set_title('RMSE (lower is better)')
axes[1].set_xticks(range(len(models_r)))
axes[1].set_xticklabels(models_r, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('RMSE (runs)')
for bar, v in zip(bars1, rmses):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(v), ha='center', fontsize=7)

# MAE comparison
bars2 = axes[2].bar(models_r, maes, color='#d62728')
axes[2].set_title('MAE (lower is better)')
axes[2].set_xticks(range(len(models_r)))
axes[2].set_xticklabels(models_r, rotation=45, ha='right', fontsize=8)
axes[2].set_ylabel('MAE (runs)')
for bar, v in zip(bars2, maes):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(v), ha='center', fontsize=7)

plt.tight_layout()
plt.savefig('regression_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Saved: regression_comparison.png')

models_c   = list(cls_metrics.keys())
test_accs  = [cls_metrics[m]['Test Acc %']  for m in models_c]
precisions = [cls_metrics[m]['Precision %'] for m in models_c]
recalls    = [cls_metrics[m]['Recall %']    for m in models_c]
f1s        = [cls_metrics[m]['F1 Score %']  for m in models_c]

x     = np.arange(len(models_c))
width = 0.2

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Classification Model Comparison', fontsize=14, fontweight='bold')

# Accuracy bar
bars = axes[0].bar(models_c, test_accs, color='#1f77b4')
axes[0].set_title('Test Accuracy % (higher is better)')
axes[0].set_xticks(range(len(models_c)))
axes[0].set_xticklabels(models_c, rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(0, 115)
for bar, v in zip(bars, test_accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{v}%', ha='center', fontsize=7)

# Precision / Recall / F1 grouped
axes[1].bar(x - width, precisions, width, label='Precision %', color='#2ca02c')
axes[1].bar(x,         recalls,    width, label='Recall %',    color='#ff7f0e')
axes[1].bar(x + width, f1s,        width, label='F1 Score %',  color='#9467bd')
axes[1].set_title('Precision / Recall / F1 Score')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models_c, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('Score (%)')
axes[1].set_ylim(0, 120)
axes[1].legend()

plt.tight_layout()
plt.savefig('classification_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Saved: classification_comparison.png')

best_reg_name = max(reg_metrics, key=lambda x: reg_metrics[x]['Test R²'])
best_cls_name = max(cls_metrics, key=lambda x: cls_metrics[x]['F1 Score %'])

print('=' * 55)
print('        BEST MODEL SELECTION')
print('=' * 55)
print()
print(f'Best Regression Model   : {best_reg_name}')
print(f'  Test R²  : {reg_metrics[best_reg_name]["Test R²"]}')
print(f'  RMSE     : {reg_metrics[best_reg_name]["RMSE"]} runs')
print(f'  MAE      : {reg_metrics[best_reg_name]["MAE"]} runs')
print(f'  Overfit  : {reg_metrics[best_reg_name]["Overfit"]}')
print()
print(f'Best Classification Model : {best_cls_name}')
print(f'  Test Acc : {cls_metrics[best_cls_name]["Test Acc %"]}%')
print(f'  F1 Score : {cls_metrics[best_cls_name]["F1 Score %"]}%')
print(f'  Precision: {cls_metrics[best_cls_name]["Precision %"]}%')
print(f'  Recall   : {cls_metrics[best_cls_name]["Recall %"]}%')
print(f'  Overfit  : {cls_metrics[best_cls_name]["Overfit"]}')

avg_score = score_df.copy()
avg_score['team_name'] = avg_score['batting_team'].map(TEAM_DEC)

team_avg = avg_score.groupby('team_name')['final_score'].mean() \
                    .sort_values(ascending=False).reset_index()
team_avg.columns = ['Team', 'Avg Final Score']
team_avg['Avg Final Score'] = team_avg['Avg Final Score'].round(1)

plt.figure(figsize=(10, 4))
plt.bar(team_avg['Team'], team_avg['Avg Final Score'], color='#1f77b4')
plt.title('Average Final Score by Batting Team')
plt.xlabel('Team')
plt.ylabel('Avg Final Score (runs)')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(team_avg['Avg Final Score']):
    plt.text(i, v + 0.3, str(v), ha='center', fontsize=8)
plt.tight_layout()
plt.show()
team_avg

chase_df = win_df.copy()
chase_df['team_name'] = chase_df['batting_team'].map(TEAM_DEC)

chase_rate = chase_df.groupby('team_name')['won_chase'].mean() \
                     .sort_values(ascending=False).reset_index()
chase_rate['Win Rate %'] = (chase_rate['won_chase'] * 100).round(1)

colors_bar = ['#2ca02c' if w >= 50 else '#d62728' for w in chase_rate['Win Rate %']]
plt.figure(figsize=(10, 4))
plt.bar(chase_rate['team_name'], chase_rate['Win Rate %'], color=colors_bar)
plt.axhline(y=50, color='black', linestyle='--', linewidth=1, label='50% line')
plt.title('Chase Win Rate by Team (%)')
plt.xlabel('Team')
plt.ylabel('Win Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.legend()
for i, v in enumerate(chase_rate['Win Rate %']):
    plt.text(i, v + 0.5, f'{v}%', ha='center', fontsize=8)
plt.tight_layout()
plt.show()
chase_rate[['team_name', 'Win Rate %']].rename(columns={'team_name':'Team'})

team_counts = pd.concat([df['batting_team'], df['bowling_team']]) \
                .value_counts().reset_index()
team_counts.columns = ['Team', 'Rows']

plt.figure(figsize=(10, 4))
plt.bar(team_counts['Team'], team_counts['Rows'], color='#9467bd')
plt.title('Ball-by-Ball Data Volume per Team')
plt.xlabel('Team')
plt.ylabel('Number of Rows')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
team_counts

# ====== CHANGE MATCH SITUATION HERE ======
BATTING_TEAM = 'Chennai Super Kings'
BOWLING_TEAM = 'Mumbai Indians'
VENUE        = IPL_VENUES[0]
CURRENT_RUNS = 85
WICKETS      = 2
OVER_NUM     = 10
BALL_NUM     = 0
# =========================================

overs  = round(OVER_NUM + (BALL_NUM / 6), 2)
crr    = CURRENT_RUNS / max(overs, 0.1)
X_pred = np.array([[TEAM_ENC[BATTING_TEAM], TEAM_ENC[BOWLING_TEAM],
                    VENUE_ENC[VENUE], CURRENT_RUNS, WICKETS, overs, crr]])

print('🏏 Score Prediction')
print('=' * 45)
print(f'Batting : {BATTING_TEAM}  |  Bowling : {BOWLING_TEAM}')
print(f'Venue   : {VENUE}')
print(f'Score   : {CURRENT_RUNS}/{WICKETS} in {OVER_NUM}.{BALL_NUM} overs  (CRR: {round(crr,2)})')
print('=' * 45)

results = []
for name, model in REG_MODELS.items():
    pred = int(model.predict(X_pred)[0])
    results.append({'Model': name, 'Predicted Score': pred})
    print(f'{name:25s} → {pred} runs')

res_df = pd.DataFrame(results)
plt.figure(figsize=(10, 4))
plt.bar(res_df['Model'], res_df['Predicted Score'], color='#1f77b4')
plt.title(f'Predicted Final Scores — {BATTING_TEAM} vs {BOWLING_TEAM} ({CURRENT_RUNS}/{WICKETS} in {OVER_NUM} overs)')
plt.ylabel('Predicted Final Score (runs)')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(res_df['Predicted Score']):
    plt.text(i, v + 0.3, str(v), ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# ====== CHANGE MATCH SITUATION HERE ======
CHASING_TEAM   = 'Mumbai Indians'
DEFENDING_TEAM = 'Chennai Super Kings'
WIN_VENUE      = IPL_VENUES[0]
TARGET         = 180
CURRENT_SCORE  = 90
WICKETS_FALLEN = 3
WIN_OVER       = 10
WIN_BALL       = 0
# =========================================

overs2   = round(WIN_OVER + (WIN_BALL / 6), 2)
crr2     = CURRENT_SCORE / max(overs2, 0.1)
rrr      = (TARGET - CURRENT_SCORE) * 6 / max((120 - overs2 * 6), 1)
pct_done = CURRENT_SCORE / max(TARGET, 1)
pct_ov   = overs2 / 20

X_win = np.array([[TEAM_ENC[CHASING_TEAM], TEAM_ENC[DEFENDING_TEAM],
                   VENUE_ENC[WIN_VENUE], TARGET, WICKETS_FALLEN,
                   overs2, crr2, rrr, pct_done, pct_ov]])

print('🏆 Win Prediction')
print('=' * 55)
print(f'Chasing : {CHASING_TEAM}  |  Defending : {DEFENDING_TEAM}')
print(f'Target  : {TARGET}  |  Score : {CURRENT_SCORE}/{WICKETS_FALLEN} in {WIN_OVER}.{WIN_BALL} overs')
print(f'CRR: {round(crr2,2)}  |  RRR: {round(rrr,2)}')
print('=' * 55)

results_cls = []
for name, model in CLS_MODELS.items():
    pred   = model.predict(X_win)[0]
    prob   = model.predict_proba(X_win)[0]
    winner = CHASING_TEAM if pred == 1 else DEFENDING_TEAM
    conf   = round(max(prob) * 100, 1)
    results_cls.append({'Model': name, 'Winner': winner, 'Confidence %': conf})
    print(f'{name:25s} → {winner:30s} | {conf}%')

res_cls_df  = pd.DataFrame(results_cls)
colors_bar  = ['#2ca02c' if w == CHASING_TEAM else '#d62728' for w in res_cls_df['Winner']]
green_patch = mpatches.Patch(color='#2ca02c', label=f'{CHASING_TEAM} wins')
red_patch   = mpatches.Patch(color='#d62728', label=f'{DEFENDING_TEAM} wins')

plt.figure(figsize=(10, 4))
plt.bar(res_cls_df['Model'], res_cls_df['Confidence %'], color=colors_bar)
plt.legend(handles=[green_patch, red_patch])
plt.title('Win Prediction Confidence by Model')
plt.ylabel('Confidence (%)')
plt.ylim(0, 115)
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(res_cls_df['Confidence %']):
    plt.text(i, v + 0.5, f'{v}%', ha='center', fontsize=8)
plt.tight_layout()
plt.show()

best_reg_name = max(reg_metrics, key=lambda x: reg_metrics[x]['Test R²'])
best_cls_name = max(cls_metrics, key=lambda x: cls_metrics[x]['F1 Score %'])

print('=' * 58)
print('      IPL PREDICTOR — FULL PIPELINE SUMMARY')
print('=' * 58)
print(f'Raw Dataset Rows         : {len(raw_df):,}')
print(f'After Team Filter        : {len(df):,}')
print(f'Unique Matches           : {df["match_id"].nunique()}')
print(f'Regression Dataset Rows  : {len(score_df):,}')
print(f'Classification Dataset   : {len(win_df):,}')
print(f'Unique Venues            : {len(IPL_VENUES)}')
print()
print(f'Regression Models        : {len(REG_MODELS)}')
for n in REG_MODELS: print(f'  • {n}')
print()
print(f'Classification Models    : {len(CLS_MODELS)}')
for n in CLS_MODELS: print(f'  • {n}')
print()
print(f'Best Regression Model    : {best_reg_name}')
print(f'  R²={reg_metrics[best_reg_name]["Test R²"]}  RMSE={reg_metrics[best_reg_name]["RMSE"]}  MAE={reg_metrics[best_reg_name]["MAE"]}')
print()
print(f'Best Classification Model: {best_cls_name}')
print(f'  Acc={cls_metrics[best_cls_name]["Test Acc %"]}%  F1={cls_metrics[best_cls_name]["F1 Score %"]}%')
print('=' * 58)
