# =========================================================
# IPL PREDICTOR — FULL INTERACTIVE VERSION
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import warnings
import gdown
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
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
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, .stApp {
    background-color: #ffffff;
    color: black;
}

.main-title {
    text-align: center;
    font-size: 70px;
    font-weight: 900;
    color: #ff6b00;
}

.sub-title {
    text-align: center;
    color: gray;
    font-size: 20px;
    margin-bottom: 25px;
}

.box {
    background: #fff4e6;
    padding: 25px;
    border-radius: 18px;
    border: 2px solid #ff6b00;
}

.big-score {
    font-size: 90px;
    font-weight: 900;
    color: #ff6b00;
    text-align: center;
}

.winner {
    font-size: 55px;
    font-weight: 900;
    color: green;
    text-align: center;
}

.metric-box {
    background: #f8f9fa;
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #ddd;
}

.small-text {
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CURRENT IPL TEAMS ONLY
# =========================================================

CURRENT_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad"
]

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

    # clean columns
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
    # TEAM NAME CLEANING
    # -----------------------------------------------------

    team_name_map = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru"
    }

    df["batting_team"] = df["batting_team"].replace(team_name_map)

    df["bowling_team"] = df["bowling_team"].replace(team_name_map)

    # -----------------------------------------------------
    # KEEP ONLY CURRENT IPL TEAMS
    # -----------------------------------------------------

    df = df[
        df["batting_team"].isin(CURRENT_TEAMS)
    ]

    df = df[
        df["bowling_team"].isin(CURRENT_TEAMS)
    ]

    # -----------------------------------------------------
    # FIX VENUE
    # -----------------------------------------------------

    if "venue" not in df.columns:
        df["venue"] = "Unknown Venue"

    # -----------------------------------------------------
    # FIX RUNS
    # -----------------------------------------------------

    if "runs_total" not in df.columns:

        if "total_runs" in df.columns:
            df["runs_total"] = df["total_runs"]

        elif "batter_runs" in df.columns:
            df["runs_total"] = df["batter_runs"]

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

        if "team_wicket" in df.columns:
            df["striker_out"] = df["team_wicket"]

        else:
            df["striker_out"] = 0

    # -----------------------------------------------------
    # ENCODING
    # -----------------------------------------------------

    all_teams = sorted(CURRENT_TEAMS)

    all_venues = sorted(df["venue"].dropna().unique())

    team_enc = {
        t: i for i, t in enumerate(all_teams)
    }

    venue_enc = {
        v: i for i, v in enumerate(all_venues)
    }

    df["batting_team_enc"] = df["batting_team"].map(team_enc)

    df["bowling_team_enc"] = df["bowling_team"].map(team_enc)

    df["venue_enc"] = df["venue"].map(venue_enc)

    # =====================================================
    # REGRESSION DATASET
    # =====================================================

    reg_rows = []

    inn1 = df[df["innings"] == 1]

    for match_id, mdf in inn1.groupby("match_id"):

        mdf = mdf.sort_values(["over", "ball"])

        final_score = mdf["runs_total"].sum()

        if final_score < 70:
            continue

        snap = mdf[mdf["ball_no"] <= 60]

        if len(snap) < 30:
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
            run_rate,
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

        if len(snap2) < 30:
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

        pp_runs = inn1m[inn1m["over"] < 6]["runs_total"].sum()

        # winner
        if "match_won_by" in mdf.columns:

            won_by = str(
                mdf["match_won_by"]
                .dropna()
                .iloc[0]
            )

            inn1_team = inn1m["batting_team"].iloc[0]

            label = 1 if inn1_team in won_by else 0

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
            rrr_at10,
            crr_at10,
            pp_runs,
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

    return reg_df, cls_df, team_enc, venue_enc, all_teams, all_venues

# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource(show_spinner=False)
def train_models(reg_df, cls_df):

    # regression
    Xr = reg_df.drop("final_score", axis=1)
    yr = reg_df["final_score"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr,
        yr,
        test_size=0.2,
        random_state=42
    )

    reg_model = GradientBoostingRegressor(
        n_estimators=200,
        random_state=42
    )

    reg_model.fit(Xr_train, yr_train)

    pred_r = reg_model.predict(Xr_test)

    reg_r2 = r2_score(yr_test, pred_r)

    # classification
    Xc = cls_df.drop("winner", axis=1)
    yc = cls_df["winner"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc,
        yc,
        test_size=0.2,
        random_state=42
    )

    cls_model = GradientBoostingClassifier(
        n_estimators=200,
        random_state=42
    )

    cls_model.fit(Xc_train, yc_train)

    pred_c = cls_model.predict(Xc_test)

    cls_acc = accuracy_score(yc_test, pred_c)

    return reg_model, cls_model, reg_r2, cls_acc

# =========================================================
# LOAD EVERYTHING
# =========================================================

with st.spinner("Loading IPL dataset..."):

    raw_df = load_data()

with st.spinner("Building features..."):

    reg_df, cls_df, team_enc, venue_enc, all_teams, all_venues = build_features(raw_df)

with st.spinner("Training AI models..."):

    reg_model, cls_model, reg_r2, cls_acc = train_models(reg_df, cls_df)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🏏 IPL PREDICTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Interactive AI-Based Match Prediction System</div>',
    unsafe_allow_html=True
)

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🎯 Score Predictor",
    "🏆 Win Predictor",
    "📊 Analytics"
])

# =========================================================
# TAB 1 — SCORE PREDICTOR
# =========================================================

with tab1:

    c1, c2 = st.columns(2)

    with c1:

        bat_team = st.selectbox(
            "Batting Team",
            all_teams
        )

        bowl_team = st.selectbox(
            "Bowling Team",
            [t for t in all_teams if t != bat_team]
        )

        venue = st.selectbox(
            "Venue",
            all_venues
        )

    with c2:

        runs = st.number_input(
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

    if st.button("🎯 Predict Final Score"):

        rr = runs / overs

        X = np.array([[
            team_enc[bat_team],
            team_enc[bowl_team],
            venue_enc[venue],
            runs,
            wickets,
            overs,
            last5_runs,
            last5_wickets,
            rr
        ]])

        prediction = int(reg_model.predict(X)[0])

        projected_same_rr = int(rr * 20)

        acceleration = prediction - projected_same_rr

        st.markdown(f"""
        <div class="box">
            <div style="text-align:center;">
                <h2>Predicted Final Score</h2>
                <div class="big-score">{prediction}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------
        # INTERACTIVE MATCH ANALYSIS
        # -------------------------------------------------

        st.markdown("## 📈 Live Match Analysis")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "Current Run Rate",
                round(rr, 2)
            )

        with m2:
            st.metric(
                "Projected at Same RR",
                projected_same_rr
            )

        with m3:
            st.metric(
                "AI Expected Finish",
                prediction
            )

        # -------------------------------------------------
        # GRAPH
        # -------------------------------------------------

        overs_arr = np.arange(overs, 21)

        same_rr_scores = [
            runs + (rr * (x - overs))
            for x in overs_arr
        ]

        ai_scores = np.linspace(
            runs,
            prediction,
            len(overs_arr)
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=overs_arr,
            y=same_rr_scores,
            mode='lines+markers',
            name='Same Run Rate',
            line=dict(color='orange', width=4)
        ))

        fig.add_trace(go.Scatter(
            x=overs_arr,
            y=ai_scores,
            mode='lines+markers',
            name='AI Predicted Finish',
            line=dict(color='green', width=4)
        ))

        fig.update_layout(
            title="Projected Innings Progression",
            xaxis_title="Overs",
            yaxis_title="Score",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Commentary style insights

        st.markdown("## 🏏 Match Insights")

        if rr >= 10:
            st.success("🔥 Batting side is scoring at an explosive rate.")

        elif rr >= 8:
            st.info("✅ Batting side is maintaining a strong scoring pace.")

        else:
            st.warning("⚠️ Run rate is below modern T20 standards.")

        if wickets <= 3:
            st.success("💪 Plenty of wickets remaining for acceleration.")

        elif wickets >= 7:
            st.error("❌ Batting side under heavy pressure.")

# =========================================================
# TAB 2 — WIN PREDICTOR
# =========================================================

with tab2:

    c1, c2 = st.columns(2)

    with c1:

        chase_team = st.selectbox(
            "Chasing Team",
            all_teams
        )

        defend_team = st.selectbox(
            "Defending Team",
            [t for t in all_teams if t != chase_team]
        )

        venue2 = st.selectbox(
            "Venue",
            all_venues,
            key="venue2"
        )

    with c2:

        target = st.number_input(
            "Target",
            50,
            300,
            180
        )

        current_score = st.number_input(
            "Current Chase Score",
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
            "Wickets Lost",
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

    if st.button("🏆 Predict Winner"):

        rrr = (
            (target - current_score)
            / max(20 - overs2, 0.1)
        )

        crr = (
            current_score
            / overs2
        )

        X2 = np.array([[
            team_enc[defend_team],
            team_enc[chase_team],
            venue_enc[venue2],
            target,
            current_score,
            overs2,
            wickets2,
            rrr,
            crr,
            pp_runs
        ]])

        pred = cls_model.predict(X2)[0]

        probs = cls_model.predict_proba(X2)[0]

        chase_prob = round(probs[0] * 100, 1)

        defend_prob = round(probs[1] * 100, 1)

        winner = (
            defend_team
            if pred == 1
            else chase_team
        )

        st.markdown(f"""
        <div class="box">
            <h2 style="text-align:center;">Predicted Winner</h2>
            <div class="winner">{winner}</div>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------
        # WIN PROBABILITY BAR
        # -------------------------------------------------

        st.markdown("## 📊 Live Win Probability")

        st.progress(int(chase_prob))

        colA, colB = st.columns(2)

        with colA:
            st.metric(
                chase_team,
                f"{chase_prob}%"
            )

        with colB:
            st.metric(
                defend_team,
                f"{defend_prob}%"
            )

        # -------------------------------------------------
        # MATCH STATUS
        # -------------------------------------------------

        st.markdown("## ⚡ Chase Analysis")

        need_runs = target - current_score

        balls_left = (20 - overs2) * 6

        st.write(f"🏏 Runs Needed: **{need_runs}**")
        st.write(f"🎯 Required Run Rate: **{round(rrr,2)}**")
        st.write(f"⚡ Current Run Rate: **{round(crr,2)}**")
        st.write(f"🟢 Balls Remaining: **{int(balls_left)}**")

        if crr > rrr:
            st.success("🔥 Chasing side is ahead of the required pace.")

        else:
            st.warning("⚠️ Chasing side needs acceleration.")

        if wickets2 <= 3:
            st.success("💪 Enough wickets left for aggressive batting.")

        elif wickets2 >= 7:
            st.error("❌ Pressure situation — wickets almost finished.")

# =========================================================
# TAB 3 — ANALYTICS
# =========================================================

with tab3:

    st.subheader("📊 Model Performance")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Regression R² Score",
            round(reg_r2, 3)
        )

    with c2:

        st.metric(
            "Classification Accuracy",
            f"{round(cls_acc * 100, 2)}%"
        )

    st.markdown("---")

    st.subheader("📂 Dataset Information")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Total Deliveries",
            f"{len(raw_df):,}"
        )

    with d2:
        st.metric(
            "Regression Samples",
            len(reg_df)
        )

    with d3:
        st.metric(
            "Classification Samples",
            len(cls_df)
        )

    st.markdown("---")

    st.subheader("📋 Raw Dataset Preview")

    st.dataframe(
        raw_df.head(20),
        use_container_width=True
    )
