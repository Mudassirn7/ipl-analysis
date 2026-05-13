# =========================================================
# IPL PREDICTOR - STREAMLIT APP
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import warnings
import gdown

warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score
)

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
    color: black;
}

.title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: #ff6b00;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 25px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background: #fff3e0;
    border: 2px solid #ff6b00;
    text-align: center;
}

.big-score {
    font-size: 70px;
    font-weight: bold;
    color: #ff6b00;
}

.winner {
    font-size: 45px;
    font-weight: bold;
    color: green;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# GOOGLE DRIVE FILE
# =========================================================

FILE_ID = "1mr2IIjhMOtRp0ZDlVLw_IFxmAY_ExGUL"

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    output = "ipl.csv"

    gdown.download(url, output, quiet=False)

    df = pd.read_csv(output, low_memory=False)

    # remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df

# =========================================================
# FEATURE ENGINEERING
# =========================================================

@st.cache_data(show_spinner=False)
def build_features(df):

    # -----------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # -----------------------------------------------------

    required_cols = [
        "match_id",
        "innings",
        "batting_team",
        "bowling_team",
        "over",
        "ball"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.write(df.columns.tolist())
        st.stop()

    # -----------------------------------------------------
    # VENUE FIX
    # -----------------------------------------------------

    if "venue" not in df.columns:
        df["venue"] = "Unknown Venue"

    # -----------------------------------------------------
    # RUNS COLUMN FIX
    # -----------------------------------------------------

    if "runs_total" not in df.columns:

        if "total_runs" in df.columns:
            df["runs_total"] = df["total_runs"]

        elif "batter_runs" in df.columns:
            df["runs_total"] = df["batter_runs"]

        else:
            st.error("No runs column found.")
            st.stop()

    # -----------------------------------------------------
    # BALL NUMBER
    # -----------------------------------------------------

    if "ball_no" not in df.columns:
        df["ball_no"] = (
            (df["over"] * 6)
            + df["ball"]
            + 1
        )

    # -----------------------------------------------------
    # WICKET COLUMN
    # -----------------------------------------------------

    if "striker_out" not in df.columns:

        if "player_dismissed" in df.columns:

            df["striker_out"] = (
                df["player_dismissed"]
                .notna()
                .astype(int)
            )

        elif "team_wicket" in df.columns:

            df["striker_out"] = (
                df["team_wicket"]
                .fillna(0)
                .astype(int)
            )

        elif "bowler_wicket" in df.columns:

            df["striker_out"] = (
                df["bowler_wicket"]
                .fillna(0)
                .astype(int)
            )

        else:
            df["striker_out"] = 0

    # -----------------------------------------------------
    # ENCODING
    # -----------------------------------------------------

    all_teams = pd.concat([
        df["batting_team"],
        df["bowling_team"]
    ]).dropna().unique()

    all_venues = df["venue"].dropna().unique()

    team_enc = {
        t: i for i, t in enumerate(sorted(all_teams))
    }

    venue_enc = {
        v: i for i, v in enumerate(sorted(all_venues))
    }

    df["batting_team_enc"] = (
        df["batting_team"]
        .map(team_enc)
        .fillna(0)
        .astype(int)
    )

    df["bowling_team_enc"] = (
        df["bowling_team"]
        .map(team_enc)
        .fillna(0)
        .astype(int)
    )

    df["venue_enc"] = (
        df["venue"]
        .map(venue_enc)
        .fillna(0)
        .astype(int)
    )

    # =====================================================
    # REGRESSION DATASET
    # =====================================================

    reg_rows = []

    inn1 = df[df["innings"] == 1]

    for match_id, mdf in inn1.groupby("match_id"):

        mdf = mdf.sort_values(["over", "ball"])

        final_score = mdf["runs_total"].sum()

        if final_score < 50:
            continue

        snap = mdf[mdf["ball_no"] <= 60]

        if len(snap) < 10:
            continue

        runs_so_far = snap["runs_total"].sum()

        wickets_so_far = snap["striker_out"].sum()

        overs_so_far = len(snap) / 6

        last5 = mdf[
            (mdf["ball_no"] > 30)
            &
            (mdf["ball_no"] <= 60)
        ]

        last5_runs = last5["runs_total"].sum()

        last5_wickets = last5["striker_out"].sum()

        run_rate = (
            runs_so_far
            / max(overs_so_far, 0.1)
        )

        reg_rows.append([
            mdf["batting_team_enc"].iloc[0],
            mdf["bowling_team_enc"].iloc[0],
            mdf["venue_enc"].iloc[0],
            runs_so_far,
            wickets_so_far,
            overs_so_far,
            last5_runs,
            last5_wickets,
            round(run_rate, 2),
            final_score
        ])

    reg_df = pd.DataFrame(reg_rows, columns=[
        "batting_team",
        "bowling_team",
        "venue",
        "runs_so_far",
        "wickets_so_far",
        "overs_so_far",
        "last5_runs",
        "last5_wickets",
        "run_rate",
        "final_score"
    ])

    # =====================================================
    # CLASSIFICATION DATASET
    # =====================================================

    cls_rows = []

    for match_id, mdf in df.groupby("match_id"):

        inn1m = mdf[mdf["innings"] == 1]
        inn2m = mdf[mdf["innings"] == 2]

        if inn1m.empty or inn2m.empty:
            continue

        target = inn1m["runs_total"].sum() + 1

        snap2 = inn2m[inn2m["ball_no"] <= 60]

        if len(snap2) < 6:
            continue

        cs_at10 = snap2["runs_total"].sum()

        wk_at10 = snap2["striker_out"].sum()

        ov_at10 = len(snap2) / 6

        rrr_at10 = (
            (target - cs_at10)
            / max(20 - ov_at10, 0.1)
        )

        crr_at10 = (
            cs_at10
            / max(ov_at10, 0.1)
        )

        pp = inn1m[inn1m["over"] < 6]

        powerplay_runs = pp["runs_total"].sum()

        # -------------------------------------------------
        # WINNER LABEL
        # -------------------------------------------------

        if "match_won_by" in mdf.columns:

            won_by = (
                str(
                    mdf["match_won_by"]
                    .dropna()
                    .iloc[0]
                )
            )

            inn1_team = (
                inn1m["batting_team"]
                .iloc[0]
            )

            label = (
                1 if inn1_team in won_by
                else 0
            )

        else:
            label = 0

        cls_rows.append([
            inn1m["batting_team_enc"].iloc[0],
            inn1m["bowling_team_enc"].iloc[0],
            inn1m["venue_enc"].iloc[0],
            target,
            cs_at10,
            ov_at10,
            wk_at10,
            round(rrr_at10, 2),
            round(crr_at10, 2),
            powerplay_runs,
            label
        ])

    cls_df = pd.DataFrame(cls_rows, columns=[
        "batting_team",
        "bowling_team",
        "venue",
        "target",
        "cs_at10",
        "ov_at10",
        "wk_at10",
        "rrr_at10",
        "crr_at10",
        "powerplay_runs",
        "winner"
    ])

    return (
        reg_df,
        cls_df,
        team_enc,
        venue_enc,
        all_teams,
        all_venues
    )

# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource(show_spinner=False)
def train_models(reg_df, cls_df):

    # -----------------------------------------------------
    # REGRESSION
    # -----------------------------------------------------

    Xr = reg_df.drop(
        "final_score",
        axis=1
    )

    yr = reg_df["final_score"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr,
        yr,
        test_size=0.2,
        random_state=42
    )

    reg_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    reg_model.fit(Xr_train, yr_train)

    pred_r = reg_model.predict(Xr_test)

    reg_r2 = r2_score(
        yr_test,
        pred_r
    )

    reg_rmse = (
        mean_squared_error(
            yr_test,
            pred_r
        ) ** 0.5
    )

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    Xc = cls_df.drop(
        "winner",
        axis=1
    )

    yc = cls_df["winner"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc,
        yc,
        test_size=0.2,
        random_state=42
    )

    cls_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    cls_model.fit(Xc_train, yc_train)

    pred_c = cls_model.predict(Xc_test)

    cls_acc = accuracy_score(
        yc_test,
        pred_c
    )

    return (
        reg_model,
        cls_model,
        reg_r2,
        reg_rmse,
        cls_acc
    )

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🏏 IPL PREDICTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real IPL Data 2008–2024</div>',
    unsafe_allow_html=True
)

# =========================================================
# LOAD + TRAIN
# =========================================================

with st.spinner("Loading IPL dataset..."):

    raw_df = load_data()

with st.spinner("Building features..."):

    (
        reg_df,
        cls_df,
        team_enc,
        venue_enc,
        all_teams,
        all_venues
    ) = build_features(raw_df)

with st.spinner("Training ML models..."):

    (
        reg_model,
        cls_model,
        reg_r2,
        reg_rmse,
        cls_acc
    ) = train_models(
        reg_df,
        cls_df
    )

st.success("Models trained successfully!")

# =========================================================
# TEAM LIST
# =========================================================

IPL_TEAMS = sorted(all_teams.tolist())

IPL_VENUES = sorted(all_venues.tolist())

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🎯 Score Predictor",
    "🏆 Win Predictor",
    "📊 Model Report"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    col1, col2 = st.columns(2)

    with col1:

        bat_team = st.selectbox(
            "Batting Team",
            IPL_TEAMS
        )

        bowl_team = st.selectbox(
            "Bowling Team",
            [t for t in IPL_TEAMS if t != bat_team]
        )

        venue = st.selectbox(
            "Venue",
            IPL_VENUES
        )

    with col2:

        current_runs = st.number_input(
            "Current Runs",
            0,
            250,
            85
        )

        wickets = st.slider(
            "Wickets Fallen",
            0,
            9,
            2
        )

        overs = st.slider(
            "Overs Completed",
            1,
            20,
            10
        )

        last5_runs = st.number_input(
            "Runs in Last 5 Overs",
            0,
            100,
            45
        )

        last5_wickets = st.slider(
            "Wickets in Last 5 Overs",
            0,
            5,
            1
        )

    if st.button("Predict Final Score"):

        rr = (
            current_runs
            / max(overs, 0.1)
        )

        X = np.array([[
            team_enc[bat_team],
            team_enc[bowl_team],
            venue_enc[venue],
            current_runs,
            wickets,
            overs,
            last5_runs,
            last5_wickets,
            rr
        ]])

        pred = int(
            reg_model.predict(X)[0]
        )

        st.markdown(f"""
        <div class="result-box">
            <div>Predicted Final Score</div>
            <div class="big-score">{pred}</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2
# =========================================================

with tab2:

    col1, col2 = st.columns(2)

    with col1:

        chase_team = st.selectbox(
            "Chasing Team",
            IPL_TEAMS
        )

        defend_team = st.selectbox(
            "Defending Team",
            [t for t in IPL_TEAMS if t != chase_team]
        )

        venue2 = st.selectbox(
            "Venue",
            IPL_VENUES,
            key="venue2"
        )

    with col2:

        target = st.number_input(
            "Target",
            50,
            300,
            180
        )

        chase_score = st.number_input(
            "Current Score",
            0,
            300,
            90
        )

        overs2 = st.slider(
            "Overs Completed",
            1,
            20,
            10,
            key="overs2"
        )

        wickets2 = st.slider(
            "Wickets Fallen",
            0,
            9,
            3
        )

        pp_runs = st.number_input(
            "Powerplay Runs",
            0,
            100,
            50
        )

    if st.button("Predict Winner"):

        rrr = (
            (target - chase_score)
            / max(20 - overs2, 0.1)
        )

        crr = (
            chase_score
            / max(overs2, 0.1)
        )

        X2 = np.array([[
            team_enc[defend_team],
            team_enc[chase_team],
            venue_enc[venue2],
            target,
            chase_score,
            overs2,
            wickets2,
            rrr,
            crr,
            pp_runs
        ]])

        pred = cls_model.predict(X2)[0]

        winner = (
            defend_team
            if pred == 1
            else chase_team
        )

        st.markdown(f"""
        <div class="result-box">
            <div>Predicted Winner</div>
            <div class="winner">{winner}</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.subheader("Regression Model")

    st.write(
        f"R² Score: {round(reg_r2, 4)}"
    )

    st.write(
        f"RMSE: {round(reg_rmse, 2)}"
    )

    st.subheader("Classification Model")

    st.write(
        f"Accuracy: {round(cls_acc * 100, 2)}%"
    )

    st.subheader("Dataset Stats")

    st.write(
        f"Deliveries: {len(raw_df):,}"
    )

    st.write(
        f"Regression Samples: {len(reg_df):,}"
    )

    st.write(
        f"Classification Samples: {len(cls_df):,}"
    )

    st.dataframe(
        raw_df.head(20),
        use_container_width=True
    )
