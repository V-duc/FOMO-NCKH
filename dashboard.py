"""
dashboard.py — FOMO Investor Detection Dashboard

Data source: snorkel_labels.csv + fomo_features.csv + enriched_trades_train.csv
Threshold:   > 0.71 = High | 0.2-0.71 = Medium | < 0.2 = Low
Không dùng XGBoost model — chỉ visualize Snorkel output.

Chạy:
    streamlit run dashboard.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import timedelta, date

warnings.filterwarnings("ignore")

from constants import OUTPUT_DIR

SNORKEL_FILE  = f"{OUTPUT_DIR}/snorkel_labels.csv"
FEATURES_FILE = f"{OUTPUT_DIR}/fomo_features.csv"
ENRICHED_FILE = f"{OUTPUT_DIR}/enriched_trades_train.csv"

# ── Snorkel-based thresholds ──────────────────────────────────────────────
FOMO_HIGH_THRESH = 0.71
FOMO_LOW_THRESH  = 0.20

DURATION_OPTIONS = {
    "5 ngày"  : 5,
    "10 ngày" : 10,
    "30 ngày" : 30,
    "60 ngày" : 60,
    "90 ngày" : 90,
    "180 ngày": 180,
    "360 ngày": 360,
    "Tất cả"  : None,
}

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FOMO Investor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* Chỉ override những thứ cần thiết — để Streamlit dark theme tự xử lý phần còn lại */
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    .block-container { padding-top: 1rem; }

    /* Buttons */
    [data-testid="stButton"] button {
        background-color: #e74c3c;
        color: white;
        border-radius: 6px;
        border: none;
    }
    [data-testid="stButton"] button:hover {
        background-color: #c0392b;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────
def assign_fomo_level(score):
    if score >= FOMO_HIGH_THRESH:  return "High"
    elif score >= FOMO_LOW_THRESH: return "Medium"
    else:                          return "Low"

def level_emoji(level):
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(level, "⚪")

def level_color(level):
    return {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}.get(level, "gray")


# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    snorkel  = pd.read_csv(SNORKEL_FILE,  parse_dates=["timestamp"])
    features = pd.read_csv(FEATURES_FILE, parse_dates=["timestamp"])
    enriched = pd.read_csv(ENRICHED_FILE, parse_dates=["timestamp"])

    # Classify từ fomo_prob Snorkel
    snorkel["fomo_level"] = snorkel["fomo_prob"].apply(assign_fomo_level)

    # Merge snorkel + features theo tx_id
    df = snorkel[["tx_id", "investor_id", "timestamp",
                  "fomo_prob", "fomo_level",
                  "lf_votes", "all_abstain"]].copy()

    # Merge market context từ enriched (BUY only)
    enriched_buy = enriched[enriched["side"] == "BUY"].copy()
    market_cols  = ["tx_id", "asset_id", "totalValue",
                    "rsi_14", "return_1d", "return_5d",
                    "volatility_5d", "volatility_10d",
                    "price_above_ma20", "market_price", "ma_20d"]
    market_cols  = [c for c in market_cols if c in enriched_buy.columns]
    df = df.merge(enriched_buy[market_cols], on="tx_id", how="left")

    # Merge behavioral features từ fomo_features
    feat_cols = [c for c in features.columns
                 if c not in ["tx_id", "investor_id", "timestamp",
                               "fomo_prob", "momentum_acceleration"]]
    df = df.merge(features[["tx_id"] + feat_cols], on="tx_id", how="left")

    return df

full_df = load_all()

# Reference groups Q1/Q4
@st.cache_data
def build_reference_groups(df):
    radar_feats = [f for f in [
        "rsi_14", "volatility_10d", "capital_acceleration_ratio",
        "investor_alignment_with_crowd", "consecutive_buy_streak",
        "rolling_buy_ratio_last_5"
    ] if f in df.columns and df[f].notna().sum() > 0]

    q1 = df[df["fomo_level"] == "Low"][radar_feats].median()
    q4 = df[df["fomo_level"] == "High"][radar_feats].median()
    return q1, q4, radar_feats

q1_ref, q4_ref, radar_feats = build_reference_groups(full_df)

# Investor summary
@st.cache_data
def build_investor_summary(df):
    s = df.groupby("investor_id").agg(
        n_tx      = ("tx_id",      "count"),
        avg_prob  = ("fomo_prob",  "mean"),
        max_prob  = ("fomo_prob",  "max"),
        n_high    = ("fomo_level", lambda x: (x=="High").sum()),
        n_medium  = ("fomo_level", lambda x: (x=="Medium").sum()),
        n_low     = ("fomo_level", lambda x: (x=="Low").sum()),
        first_tx  = ("timestamp",  "min"),
        last_tx   = ("timestamp",  "max"),
    ).reset_index()
    s["fomo_level"] = s["avg_prob"].apply(assign_fomo_level)
    s["high_ratio"] = s["n_high"] / s["n_tx"]
    return s.sort_values("avg_prob", ascending=False).reset_index(drop=True)

investor_summary = build_investor_summary(full_df)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🔍 Bộ lọc")

# ── Init session state ────────────────────────────────────────────────────
if "cur_idx" not in st.session_state:
    st.session_state.cur_idx = 0
if "selected_id" not in st.session_state:
    st.session_state.selected_id = ""

# ── 1. Search investor ────────────────────────────────────────────────────
st.sidebar.subheader("👤 Chọn Investor")

search_id = st.sidebar.text_input(
    "Tìm Investor ID:", placeholder="Nhập ID...",
    key="search_input"
)

level_filter = st.sidebar.multiselect(
    "Lọc theo FOMO Level:",
    options=["High", "Medium", "Low"],
    default=[],
    key="level_filter"
)

filtered_inv = investor_summary.copy()
if level_filter:
    filtered_inv = filtered_inv[
        filtered_inv["fomo_level"].isin(level_filter)
    ]

avail_ids = filtered_inv["investor_id"].tolist()
if not avail_ids:
    avail_ids = investor_summary["investor_id"].tolist()

# Nếu user nhập ID hợp lệ → cập nhật cur_idx
if search_id and search_id in avail_ids:
    st.session_state.cur_idx = avail_ids.index(search_id)

# Đảm bảo cur_idx trong bounds
st.session_state.cur_idx = max(0, min(
    st.session_state.cur_idx, len(avail_ids) - 1
))

# Navigation buttons
c_p, c_n = st.sidebar.columns(2)
with c_p:
    if st.button("⬅️ Prev", key="btn_prev"):
        st.session_state.cur_idx = max(0, st.session_state.cur_idx - 1)
        st.rerun()
with c_n:
    if st.button("Next ➡️", key="btn_next"):
        st.session_state.cur_idx = min(
            len(avail_ids) - 1, st.session_state.cur_idx + 1
        )
        st.rerun()

selected_id = avail_ids[st.session_state.cur_idx]
st.session_state.selected_id = selected_id

st.sidebar.markdown("---")

# ── 2. Time filter ────────────────────────────────────────────────────────
st.sidebar.subheader("⏱️ Khoảng thời gian")

inv_all  = full_df[full_df["investor_id"] == selected_id].copy()
min_date = inv_all["timestamp"].min().date() if len(inv_all) > 0 else date(2020, 11, 1)
max_date = inv_all["timestamp"].max().date() if len(inv_all) > 0 else date(2022, 7, 31)

# Reset start_date khi đổi investor
date_key     = f"sd_{selected_id}"
duration_key = f"dur_{selected_id}"

start_date = st.sidebar.date_input(
    "Start date:",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
    key=date_key,
)
duration_label = st.sidebar.selectbox(
    "Khoảng thời gian:",
    options=list(DURATION_OPTIONS.keys()),
    index=list(DURATION_OPTIONS.keys()).index("Tất cả"),
    key=duration_key,
)
duration_days = DURATION_OPTIONS[duration_label]
end_date = (start_date + timedelta(days=duration_days)
            if duration_days else max_date)

st.sidebar.markdown("---")

# ── 3. Investor list (thu nhỏ xuống dưới) ────────────────────────────────
st.sidebar.markdown(f"**{len(filtered_inv):,} investors**")
disp = filtered_inv[["investor_id", "fomo_level", "avg_prob"]].copy()
disp["Level"] = disp["fomo_level"].apply(lambda x: f"{level_emoji(x)} {x}")
disp["Score"] = disp["avg_prob"].round(4)
disp = disp[["investor_id", "Level", "Score"]]
disp.index = range(1, len(disp) + 1)
st.sidebar.dataframe(disp, height=200, use_container_width=True)

inv_data = inv_all[
    (inv_all["timestamp"].dt.date >= start_date) &
    (inv_all["timestamp"].dt.date <= end_date)
].copy()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
st.title("📊 FOMO Investor Dashboard")
st.markdown(
    f"**Investor:** `{selected_id}` | "
    f"**Period:** {start_date} → {end_date} | "
    f"**{len(inv_data):,} lệnh**"
)
st.markdown("---")

if len(inv_data) == 0:
    st.warning("Không có lệnh nào trong khoảng thời gian này.")
    st.stop()

inv_sorted  = inv_data.sort_values("timestamp")
avg_prob    = inv_data["fomo_prob"].mean()
max_prob    = inv_data["fomo_prob"].max()
level_      = assign_fomo_level(avg_prob)
certainty   = abs(avg_prob - 0.5) * 2
n_tx        = len(inv_data)
high_ratio  = (inv_data["fomo_level"] == "High").mean()
n_high      = (inv_data["fomo_level"] == "High").sum()
n_medium    = (inv_data["fomo_level"] == "Medium").sum()
n_low       = (inv_data["fomo_level"] == "Low").sum()

# ── Metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("FOMO Score (avg)", f"{avg_prob:.4f}",
          f"{avg_prob - full_df['fomo_prob'].mean():.4f} vs market")
c2.metric("FOMO Level", f"{level_emoji(level_)} {level_}")
c3.metric("LF Votes (avg)",
          f"{inv_data['lf_votes'].mean():.1f}" if "lf_votes" in inv_data.columns else "N/A")
c4.metric("Số lệnh", f"{n_tx:,}")
c5.metric("% High FOMO", f"{high_ratio:.1%}",
          f"{high_ratio - (full_df['fomo_level']=='High').mean():.1%} vs market")
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# ROW 1 — Gauge + Radar
# ════════════════════════════════════════════════════════════════════════════
col_g, col_r = st.columns([1, 2])

with col_g:
    st.subheader("🎯 FOMO Score")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_prob,
        title={"text": f"Avg Snorkel Prob<br>"
                       f"<span style='font-size:0.8em'>Max: {max_prob:.4f}</span>"},
        gauge={
            "axis" : {"range": [0, 1]},
            "bar"  : {"color": level_color(level_)},
            "steps": [
                {"range": [0,               FOMO_LOW_THRESH],  "color": "#d5f5e3"},
                {"range": [FOMO_LOW_THRESH, FOMO_HIGH_THRESH], "color": "#fef9e7"},
                {"range": [FOMO_HIGH_THRESH, 1],               "color": "#fadbd8"},
            ],
            "threshold": {
                "line" : {"color": "black", "width": 3},
                "value": FOMO_HIGH_THRESH
            }
        }
    ))
    fig_g.update_layout(height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig_g, use_container_width=True)

with col_r:
    st.subheader("🧬 Behavioral DNA")
    st.caption("So sánh vs Q1 Rational và Q4 High FOMO (market-wide)")

    inv_radar = inv_data[radar_feats].median()

    # Normalize theo range thật — RSI cố định 0-100
    FIXED_RANGES = {
        "rsi_14"                       : (0, 100),
        "rolling_buy_ratio_last_5"     : (0, 1),
        "investor_alignment_with_crowd": (0, 1),
        "consecutive_buy_streak"       : (0, 20),
        "capital_acceleration_ratio"   : (0, 5),
    }
    if "volatility_10d" in full_df.columns:
        FIXED_RANGES["volatility_10d"] = (0, full_df["volatility_10d"].quantile(0.99))

    def norm(s):
        result = []
        for feat in radar_feats:
            val = float(s[feat]) if feat in s.index else 0
            if feat in FIXED_RANGES:
                lo_f, hi_f = FIXED_RANGES[feat]
            else:
                lo_f = float(full_df[feat].quantile(0.01))
                hi_f = float(full_df[feat].quantile(0.99))
            result.append(float(np.clip((val - lo_f) / (hi_f - lo_f + 1e-8), 0, 1)))
        return result

    cats  = radar_feats + [radar_feats[0]]
    inv_n_raw = norm(inv_radar)
    q1_n_raw  = norm(q1_ref)
    q4_n_raw  = norm(q4_ref)
    inv_n = inv_n_raw + [inv_n_raw[0]]
    q1_n  = q1_n_raw  + [q1_n_raw[0]]
    q4_n  = q4_n_raw  + [q4_n_raw[0]]

    fig_r = go.Figure()
    for vals, name, color, opacity, fillcolor in [
        (inv_n, f"Investor {selected_id}", "#e74c3c", 0.85, "rgba(231,76,60,0.45)"),
        (q4_n,  "Q4 High FOMO",           "#f39c12", 0.85, "rgba(243,156,18,0.35)"),
        (q1_n,  "Q1 Rational",            "#27ae60", 0.85, "rgba(39,174,96,0.35)"),
    ]:
        fig_r.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name=name,
            line=dict(color=color, width=3),
            fill="toself", opacity=opacity,
            fillcolor=fillcolor,
        ))
    fig_r.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="#333", tickfont=dict(color="#aaa")),
            angularaxis=dict(gridcolor="#333", tickfont=dict(color="#fff")),
            bgcolor="#0e1117",
        ),
        height=420,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                    font=dict(color="#fff"))
    )
    st.plotly_chart(fig_r, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# ROW 2 — Buy Proximity to Peak + Herding Scatter
# ════════════════════════════════════════════════════════════════════════════
col_pk, col_hd = st.columns(2)

with col_pk:
    st.subheader("📍 Buy Proximity to Peak")
    st.caption("Vị trí giá mua so với MA20 — chứng minh 'mua đuổi'")

    if "price_above_ma20" in inv_data.columns:
        prox = inv_data["price_above_ma20"].dropna()
        if len(prox) > 0:
            fig_pk = go.Figure()

            # Tạo bins thủ công để control màu per bin
            bins    = np.arange(
                float(prox.min()),
                float(prox.max()) + 0.025, 0.025
            )
            counts, edges = np.histogram(prox, bins=bins)
            bin_centers   = (edges[:-1] + edges[1:]) / 2
            bar_colors    = [
                "#e74c3c" if c > 1.05
                else "#f39c12" if c > 1.0
                else "#27ae60"
                for c in bin_centers
            ]

            fig_pk.add_trace(go.Bar(
                x=bin_centers,
                y=counts,
                marker_color=bar_colors,
                width=0.022,
                name="Buy proximity",
            ))
            fig_pk.add_vline(x=1.0,  line_dash="dash",
                             line_color="black",
                             annotation_text="MA20")
            fig_pk.add_vline(x=1.05, line_dash="dot",
                             line_color="#e74c3c",
                             annotation_text="+5%")
            pct_above = (prox > 1.0).mean() * 100
            fig_pk.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1a2e", 
                height=300,
                xaxis_title="Price / MA20",
                yaxis_title="Số lệnh", showlegend=False,
            )
            st.caption(f"⚠️ {pct_above:.1f}% lệnh mua khi giá > MA20")
            st.plotly_chart(fig_pk, use_container_width=True)
    else:
        st.info("Không có dữ liệu price_above_ma20.")

with col_hd:
    st.subheader("🐑 Herding Sensitivity")
    st.caption("Giá trị lệnh vs mức độ đám đông — đo tính bầy đàn")

    herd_cols = ["totalValue", "investor_alignment_with_crowd",
                 "fomo_prob", "fomo_level", "timestamp"]
    herd_cols = [c for c in herd_cols if c in inv_data.columns]
    herd_df   = inv_data[herd_cols].dropna()

    if len(herd_df) > 0 and "totalValue" in herd_df.columns:
        fig_hd = px.scatter(
            herd_df,
            x="investor_alignment_with_crowd",
            y="totalValue",
            color="fomo_level",
            color_discrete_map={
                "High": "#e74c3c",
                "Medium": "#f39c12",
                "Low": "#27ae60"
            },
            size="fomo_prob", size_max=15,
            hover_data=["timestamp", "fomo_prob"],
            labels={
                "investor_alignment_with_crowd":
                    "Crowd Alignment (1=mua khi đám đông mua)",
                "totalValue": "Giá trị lệnh (€)"
            },
        )
        fig_hd.update_layout(height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig_hd, use_container_width=True)
    else:
        st.info("Không đủ dữ liệu herding.")


# ════════════════════════════════════════════════════════════════════════════
# ROW 3 — Timeline FOMO Score
# ════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Timeline FOMO Score")

fig_tl = go.Figure()
fig_tl.add_trace(go.Scatter(
    x=inv_sorted["timestamp"],
    y=inv_sorted["fomo_prob"],
    mode="lines+markers",
    name="Snorkel fomo_prob",
    line=dict(color="#3498db", width=1.5),
    marker=dict(
        color=[level_color(l) for l in inv_sorted["fomo_level"]],
        size=8, line=dict(width=1, color="white")
    ),
    hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>"
))
fig_tl.add_hline(y=FOMO_HIGH_THRESH, line_dash="dash",
                 line_color="#e74c3c",
                 annotation_text=f"High ({FOMO_HIGH_THRESH})",
                 annotation_position="right")
fig_tl.add_hline(y=FOMO_LOW_THRESH, line_dash="dash",
                 line_color="#27ae60",
                 annotation_text=f"Low ({FOMO_LOW_THRESH})",
                 annotation_position="right")
fig_tl.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", 
    height=300, yaxis=dict(range=[0, 1.05]),
    xaxis_title="Thời gian",
    yaxis_title="FOMO Prob (Snorkel)",
    hovermode="x unified"
)
st.plotly_chart(fig_tl, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# ROW 4 — Feature Trajectory
# ════════════════════════════════════════════════════════════════════════════
st.subheader("📉 Feature Trajectory")
st.caption("Quỹ đạo chỉ số kỹ thuật theo thời gian — 'Tại sao họ mua'")

traj_feats = [f for f in [
    "rsi_14", "volatility_10d", "consecutive_buy_streak",
    "rolling_buy_ratio_last_5", "capital_acceleration_ratio",
    "return_5d", "volatility_ratio"
] if f in full_df.columns and f in inv_sorted.columns and inv_sorted[f].notna().sum() > 0]

if not traj_feats:
    st.warning("Không có feature trajectory nào có dữ liệu.")
elif len(inv_sorted) == 1:
    st.info("Investor này chỉ có 1 lệnh — không thể vẽ trajectory.")
else:
    sel_traj = st.multiselect(
        "Chọn feature:", options=traj_feats,
        default=traj_feats[:3],
    )
    if sel_traj:
        fig_tr    = go.Figure()
        colors_tr = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
        has_data  = False

        for i, feat in enumerate(sel_traj):
            s = inv_sorted[feat].dropna()
            if len(s) < 2:
                st.caption(f"⚠️ {feat}: không đủ data (n={len(s)})")
                continue

            if s.max() == s.min():
                norm_s = pd.Series([0.5] * len(s), index=s.index)
                st.caption(f"ℹ️ {feat}: constant = {s.iloc[0]:.4f}")
            else:
                norm_s = (s - s.min()) / (s.max() - s.min())

            fig_tr.add_trace(go.Scatter(
                x=inv_sorted.loc[s.index, "timestamp"],
                y=norm_s, mode="lines+markers", name=feat,
                line=dict(color=colors_tr[i % len(colors_tr)], width=2),
                customdata=s.values,
                hovertemplate=f"<b>{feat}</b>: %{{customdata:.4f}}<extra></extra>",
            ))
            has_data = True

        if has_data:
            fig_tr.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                height=320,
                yaxis_title="Normalized (0-1)",
                xaxis_title="Thời gian",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                            font=dict(color="#fff")),
                xaxis=dict(gridcolor="#333"),
                yaxis=dict(gridcolor="#333"),
            )
            st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.warning("Không có feature nào đủ data để vẽ trajectory.")


# ════════════════════════════════════════════════════════════════════════════
# ROW 5 — Transaction table + Pie
# ════════════════════════════════════════════════════════════════════════════
col_tx, col_pie = st.columns([2, 1])

with col_tx:
    st.subheader("📋 Chi tiết lệnh")
    show = ["timestamp", "fomo_prob", "fomo_level"]
    if "lf_votes" in inv_sorted.columns:
        show.append("lf_votes")
    if "rsi_14" in inv_sorted.columns:
        show.append("rsi_14")
    if "return_5d" in inv_sorted.columns:
        show.append("return_5d")

    tx_disp = inv_sorted[show].copy()
    tx_disp["fomo_level"] = tx_disp["fomo_level"].apply(
        lambda x: f"{level_emoji(x)} {x}"
    )
    tx_disp["fomo_prob"] = tx_disp["fomo_prob"].round(4)
    for col in ["rsi_14", "return_5d"]:
        if col in tx_disp.columns:
            tx_disp[col] = tx_disp[col].round(4)
    tx_disp.index = range(1, len(tx_disp) + 1)
    st.dataframe(tx_disp, use_container_width=True, height=350)

with col_pie:
    st.subheader("🥧 Phân bố Level")
    level_c = inv_data["fomo_level"].value_counts().reindex(
        ["High", "Medium", "Low"], fill_value=0
    )
    fig_pie = go.Figure(go.Pie(
        labels=[f"{level_emoji(l)} {l}" for l in level_c.index],
        values=level_c.values,
        marker_colors=["#e74c3c", "#f39c12", "#27ae60"],
        hole=0.4,
        textinfo="label+percent+value",
    ))
    fig_pie.update_layout(height=350, showlegend=False, paper_bgcolor="#0e1117",
                          margin=dict(t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)


# ── Summary ───────────────────────────────────────────────────────────────
st.markdown("---")
st.info(f"""
**📋 Tóm tắt — Investor `{selected_id}`**
Period: **{start_date}** → **{end_date}** ({duration_label})

- **FOMO Level:** {level_emoji(level_)} **{level_}**  
  avg = {avg_prob:.4f} | max = {max_prob:.4f}
- **Phân bố:** 🔴 High: {n_high} | 🟡 Medium: {n_medium} | 🟢 Low: {n_low}
- **% High FOMO:** {high_ratio:.1%} vs market {(full_df['fomo_level']=='High').mean():.1%}
- **Threshold:** High > {FOMO_HIGH_THRESH} | Low < {FOMO_LOW_THRESH}
""")


# ── Market overview ────────────────────────────────────────────────────────
with st.expander("🌐 Top 10 FOMO Investors (toàn thị trường)"):
    top10 = investor_summary.head(10).copy()
    top10["Level"]  = top10["fomo_level"].apply(lambda x: f"{level_emoji(x)} {x}")
    top10["Score"]  = top10["avg_prob"].round(4)
    top10["High %"] = (top10["high_ratio"]*100).round(1).astype(str) + "%"
    top10["N lệnh"] = top10["n_tx"]
    top10.index = range(1, 11)
    st.dataframe(
        top10[["investor_id", "Level", "Score", "High %", "N lệnh"]],
        use_container_width=True
    )