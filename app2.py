# =============================================================================
#  Stock Price Prediction Dashboard — Polynomial Regression
#  Framework : Streamlit + Plotly + Scikit-Learn
#  Author    : Expert Data Scientist / UI-UX Developer
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolyStock · Prediction Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg-deep:      #080c14;
    --bg-card:      #0d1424;
    --bg-card2:     #111a2e;
    --border:       #1e2d4a;
    --accent-gold:  #f0c040;
    --accent-green: #30d980;
    --accent-red:   #ff5f6d;
    --accent-blue:  #4d9fff;
    --text-primary: #e8edf8;
    --text-muted:   #6a7b99;
    --font-display: 'DM Serif Display', serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'DM Mono', monospace;
}

/* ── App shell ── */
.stApp { background: var(--bg-deep); color: var(--text-primary); font-family: var(--font-body); }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: var(--text-primary); }

/* ── Hero heading ── */
.hero-title {
    font-family: var(--font-display);
    font-size: clamp(2.2rem, 4vw, 3.4rem);
    letter-spacing: -0.5px;
    line-height: 1.15;
    background: linear-gradient(120deg, var(--accent-gold) 0%, #ff9940 60%, var(--accent-red) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.6rem; }
.metric-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.5rem;
    flex: 1 1 160px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.gold::before  { background: var(--accent-gold); }
.metric-card.green::before { background: var(--accent-green); }
.metric-card.blue::before  { background: var(--accent-blue); }
.metric-card.red::before   { background: var(--accent-red); }
.metric-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); margin-bottom: 0.4rem; font-family: var(--font-mono); }
.metric-value { font-size: 1.6rem; font-weight: 600; font-family: var(--font-display); }
.metric-value.gold  { color: var(--accent-gold); }
.metric-value.green { color: var(--accent-green); }
.metric-value.blue  { color: var(--accent-blue); }
.metric-value.red   { color: var(--accent-red); }
.metric-delta { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.2rem; font-family: var(--font-mono); }

/* ── Section headings ── */
.section-heading {
    font-family: var(--font-display);
    font-size: 1.3rem;
    color: var(--text-primary);
    border-left: 3px solid var(--accent-gold);
    padding-left: 0.75rem;
    margin: 1.6rem 0 0.8rem;
}

/* ── Info box ── */
.info-box {
    background: #0d1e38;
    border: 1px solid #1e3560;
    border-left: 3px solid var(--accent-blue);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: #8eadd4;
    font-family: var(--font-mono);
    margin-bottom: 1rem;
}

/* ── Welcome screen ── */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 55vh;
    text-align: center;
    padding: 2rem;
}
.welcome-icon { font-size: 4.5rem; line-height: 1; margin-bottom: 1.2rem; }
.welcome-title { font-family: var(--font-display); font-size: 2.2rem; color: var(--text-primary); margin-bottom: 0.6rem; }
.welcome-body { color: var(--text-muted); font-size: 0.95rem; max-width: 480px; line-height: 1.7; }

/* ── Data table ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Slider label ── */
.stSlider label { color: var(--text-primary) !important; font-family: var(--font-mono) !important; font-size: 0.8rem !important; letter-spacing: 1px !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 0.5rem">
        <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;
                    background:linear-gradient(120deg,#f0c040,#ff9940);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">
            PolyStock
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                    color:#6a7b99;letter-spacing:2px;text-transform:uppercase;
                    margin-top:2px;">
            Prediction Engine · v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── File uploader ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                color:#6a7b99;letter-spacing:1.5px;text-transform:uppercase;
                margin-bottom:0.4rem;">
        01 · Upload Data
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Upload historical stock CSV",
        type=["csv"],
        help="Expected columns: Date, Open, High, Low, Close, Shares Traded, Turnover(in cr ruppes)",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Model controls ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                color:#6a7b99;letter-spacing:1.5px;text-transform:uppercase;
                margin-bottom:0.4rem;">
        02 · Model Tuning
    </div>
    """, unsafe_allow_html=True)

    poly_degree = st.slider(
        label="POLYNOMIAL DEGREE",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        help="Higher degree → more flexible curve (risk of overfitting)",
    )

    # ── Degree hint ────────────────────────────────────────────────────────────
    degree_hints = {
        2: ("🟢", "Smooth quadratic — low variance, slight underfitting risk"),
        3: ("🟡", "Cubic — balanced flexibility and generalisability"),
        4: ("🟠", "Quartic — high flexibility, watch for overfitting"),
        5: ("🔴", "Quintic — very flexible, overfitting likely on 30 pts"),
    }
    icon, hint = degree_hints[poly_degree]
    st.markdown(f"""
    <div style="background:#0d1424;border:1px solid #1e2d4a;border-radius:8px;
                padding:0.7rem 0.9rem;font-family:'DM Mono',monospace;
                font-size:0.72rem;color:#8eadd4;line-height:1.5;margin-top:0.4rem;">
        {icon} &nbsp;{hint}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Expected columns guide ─────────────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                color:#6a7b99;letter-spacing:1.5px;text-transform:uppercase;
                margin-bottom:0.6rem;">
        03 · Expected CSV Schema
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:0.73rem;
                color:#8eadd4;line-height:1.9;">
        📅 Date<br>
        📈 Open<br>
        📈 High<br>
        📉 Low<br>
        💰 Close <span style="color:#f0c040;">(target)</span><br>
        📊 Shares Traded<br>
        💹 Turnover(in cr ruppes)
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-title">Stock Price Prediction</div>
<div class="hero-sub">Polynomial Regression · Powered by Scikit-Learn & Plotly</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  WELCOME STATE (no file uploaded yet)
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded_file is None:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">📂</div>
        <div class="welcome-title">Upload Your Stock Data</div>
        <div class="welcome-body">
            Drop a <strong style="color:#f0c040;">CSV file</strong> in the sidebar to instantly
            fit a Polynomial Regression model on the last 30 trading days and predict
            tomorrow's closing price — interactively.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()   # halt execution gracefully — nothing more to render


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
REQUIRED_COLS = {"Date", "Open", "High", "Low", "Close"}

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV, parse dates, sort chronologically."""
    df = pd.read_csv(file_bytes)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Parse Date — try multiple common formats
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, infer_datetime_format=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Coerce numeric columns
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    df.dropna(subset=["Date", "Close"], inplace=True)
    return df


with st.spinner("Parsing your data…"):
    try:
        df_raw = load_data(uploaded_file, uploaded_file.name)
    except Exception as e:
        st.error(f"❌ Could not read the file: {e}")
        st.stop()

# ── Validate required columns ──────────────────────────────────────────────────
missing_cols = REQUIRED_COLS - set(df_raw.columns)
if missing_cols:
    st.error(f"❌ Missing required columns: **{', '.join(missing_cols)}**\n\n"
             f"Found: {list(df_raw.columns)}")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  ·  Last-30-day window
# ═══════════════════════════════════════════════════════════════════════════════
df_30 = df_raw.tail(30).copy()
df_30.reset_index(drop=True, inplace=True)

# Day number: 1 … 30 (the regression feature X)
df_30["Day"] = np.arange(1, len(df_30) + 1, dtype=float)

X = df_30[["Day"]].values          # shape (n, 1)
y = df_30["Close"].values           # shape (n,)


# ═══════════════════════════════════════════════════════════════════════════════
#  POLYNOMIAL REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
poly    = PolynomialFeatures(degree=poly_degree, include_bias=True)
X_poly  = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# ── Smooth fitted curve (100 pts for a clean line) ────────────────────────────
X_smooth       = np.linspace(1, len(df_30), 300).reshape(-1, 1)
X_smooth_poly  = poly.transform(X_smooth)
y_curve        = model.predict(X_smooth_poly)

# ── Predict Day 31 ────────────────────────────────────────────────────────────
next_day        = np.array([[len(df_30) + 1]])
next_day_poly   = poly.transform(next_day)
predicted_price = float(model.predict(next_day_poly)[0])

# ── Model evaluation ──────────────────────────────────────────────────────────
y_fitted = model.predict(X_poly)
r2       = r2_score(y, y_fitted)
mae      = mean_absolute_error(y, y_fitted)

# ── Price delta vs last close ─────────────────────────────────────────────────
last_close   = float(y[-1])
price_delta  = predicted_price - last_close
delta_pct    = (price_delta / last_close) * 100


# ═══════════════════════════════════════════════════════════════════════════════
#  METRIC CARDS
# ═══════════════════════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)

with col1:
    direction = "↑" if price_delta >= 0 else "↓"
    clr = "green" if price_delta >= 0 else "red"
    st.markdown(f"""
    <div class="metric-card {clr}">
        <div class="metric-label">Predicted Close · Day 31</div>
        <div class="metric-value {clr}">₹{predicted_price:,.2f}</div>
        <div class="metric-delta">{direction} ₹{abs(price_delta):.2f} ({delta_pct:+.2f}%) vs last close</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card gold">
        <div class="metric-label">Last Close Price</div>
        <div class="metric-value gold">₹{last_close:,.2f}</div>
        <div class="metric-delta">{df_30['Date'].iloc[-1].strftime('%d %b %Y')}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card blue">
        <div class="metric-label">R² Score</div>
        <div class="metric-value blue">{r2:.4f}</div>
        <div class="metric-delta">Degree-{poly_degree} polynomial fit</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card {'green' if mae < last_close * 0.02 else 'red'}">
        <div class="metric-label">Mean Abs. Error</div>
        <div class="metric-value {'green' if mae < last_close * 0.02 else 'red'}">₹{mae:.2f}</div>
        <div class="metric-delta">On training window (30 days)</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PLOTLY CHART
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-heading">Regression Curve &amp; Next-Day Forecast</div>',
            unsafe_allow_html=True)

# Build x-axis labels: dates for days 1-30, "Day 31" label for prediction
date_labels = [d.strftime("%d %b") for d in df_30["Date"]]

fig = go.Figure()

# 1 ── Actual close prices ──────────────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=df_30["Day"],
    y=y,
    mode="markers+lines",
    name="Actual Close",
    line=dict(color="#4d9fff", width=1.4, dash="dot"),
    marker=dict(
        size=7,
        color="#4d9fff",
        line=dict(color="#0d1424", width=1.5),
    ),
    hovertemplate=(
        "<b>%{customdata}</b><br>"
        "Close: ₹%{y:,.2f}<extra></extra>"
    ),
    customdata=date_labels,
))

# 2 ── Polynomial regression curve ─────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=X_smooth.flatten(),
    y=y_curve,
    mode="lines",
    name=f"Poly Fit (degree {poly_degree})",
    line=dict(color="#f0c040", width=2.5),
    hoverinfo="skip",
))

# 3 ── Predicted Day 31 point ───────────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=[len(df_30) + 1],
    y=[predicted_price],
    mode="markers+text",
    name="Predicted (Day 31)",
    marker=dict(
        symbol="star",
        size=22,
        color="#30d980",
        line=dict(color="#0d1424", width=2),
    ),
    text=[f"  ₹{predicted_price:,.2f}"],
    textposition="middle right",
    textfont=dict(family="DM Mono", size=13, color="#30d980"),
    hovertemplate="<b>Day 31 Prediction</b><br>₹%{y:,.2f}<extra></extra>",
))

# 4 ── Vertical dashed separator ───────────────────────────────────────────────
fig.add_vline(
    x=len(df_30) + 0.5,
    line=dict(color="#1e3560", width=1.5, dash="dash"),
    annotation_text="Forecast →",
    annotation_font=dict(color="#6a7b99", size=11, family="DM Mono"),
    annotation_position="top right",
)

# 5 ── Shaded forecast zone ────────────────────────────────────────────────────
fig.add_vrect(
    x0=len(df_30) + 0.5,
    x1=len(df_30) + 1.8,
    fillcolor="#30d980",
    opacity=0.04,
    line_width=0,
)

# ── Tick labels: map day number → date string ──────────────────────────────────
tick_vals = list(df_30["Day"][::5]) + [int(df_30["Day"].iloc[-1])]
tick_text = [df_30.loc[df_30["Day"] == v, "Date"].values[0] for v in tick_vals]
tick_text = [pd.Timestamp(t).strftime("%d %b") for t in tick_text]

# ── Layout ─────────────────────────────────────────────────────────────────────
fig.update_layout(
    plot_bgcolor="#080c14",
    paper_bgcolor="#080c14",
    font=dict(family="DM Sans", color="#8eadd4", size=12),
    margin=dict(l=60, r=40, t=30, b=60),
    height=440,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.01,
        xanchor="left", x=0,
        font=dict(family="DM Mono", size=11, color="#8eadd4"),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        gridcolor="#111a2e",
        linecolor="#1e2d4a",
        tickfont=dict(family="DM Mono", size=10),
        title=dict(text="Trading Date", font=dict(size=11, color="#6a7b99")),
    ),
    yaxis=dict(
        gridcolor="#111a2e",
        linecolor="#1e2d4a",
        tickformat="₹,.0f",
        tickfont=dict(family="DM Mono", size=10),
        title=dict(text="Close Price (₹)", font=dict(size=11, color="#6a7b99")),
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0d1424",
        bordercolor="#1e2d4a",
        font=dict(family="DM Mono", size=11, color="#e8edf8"),
    ),
)

st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECONDARY CHARTS  ·  Two columns
# ═══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1], gap="medium")

# ── Residuals plot ─────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-heading">Residuals</div>', unsafe_allow_html=True)
    residuals = y - y_fitted

    fig_res = go.Figure()
    fig_res.add_trace(go.Bar(
        x=df_30["Day"],
        y=residuals,
        marker_color=np.where(residuals >= 0, "#30d980", "#ff5f6d"),
        name="Residual",
        hovertemplate="Day %{x}: ₹%{y:,.2f}<extra></extra>",
    ))
    fig_res.add_hline(y=0, line=dict(color="#6a7b99", width=1))
    fig_res.update_layout(
        plot_bgcolor="#080c14", paper_bgcolor="#080c14",
        height=270, margin=dict(l=50, r=20, t=10, b=40),
        xaxis=dict(gridcolor="#111a2e", tickfont=dict(family="DM Mono", size=9)),
        yaxis=dict(gridcolor="#111a2e", tickfont=dict(family="DM Mono", size=9),
                   tickformat="₹,.0f"),
        showlegend=False,
    )
    st.plotly_chart(fig_res, use_container_width=True)

# ── Volume / Turnover bar chart ────────────────────────────────────────────────
with right:
    turnover_col = next(
        (c for c in df_raw.columns if "turnover" in c.lower() or "volume" in c.lower()
         or "shares" in c.lower()),
        None
    )
    if turnover_col:
        st.markdown(f'<div class="section-heading">{turnover_col}</div>',
                    unsafe_allow_html=True)

        tv = pd.to_numeric(
            df_30[turnover_col].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=df_30["Day"],
            y=tv,
            marker_color="#4d9fff",
            opacity=0.75,
            hovertemplate="Day %{x}: %{y:,.0f}<extra></extra>",
        ))
        fig_vol.update_layout(
            plot_bgcolor="#080c14", paper_bgcolor="#080c14",
            height=270, margin=dict(l=50, r=20, t=10, b=40),
            xaxis=dict(gridcolor="#111a2e", tickfont=dict(family="DM Mono", size=9)),
            yaxis=dict(gridcolor="#111a2e", tickfont=dict(family="DM Mono", size=9)),
            showlegend=False,
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.markdown('<div class="section-heading">OHLC Range</div>',
                    unsafe_allow_html=True)
        fig_ohlc = go.Figure(go.Candlestick(
            x=df_30["Date"],
            open=df_30["Open"], high=df_30["High"],
            low=df_30["Low"],   close=df_30["Close"],
            increasing_line_color="#30d980",
            decreasing_line_color="#ff5f6d",
        ))
        fig_ohlc.update_layout(
            plot_bgcolor="#080c14", paper_bgcolor="#080c14",
            height=270, margin=dict(l=50, r=20, t=10, b=40),
            xaxis_rangeslider_visible=False, showlegend=False,
            xaxis=dict(gridcolor="#111a2e"),
            yaxis=dict(gridcolor="#111a2e", tickformat="₹,.0f"),
        )
        st.plotly_chart(fig_ohlc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-heading">Last 30 Trading Days · Raw Data</div>',
            unsafe_allow_html=True)

display_df = df_30.copy()
display_df["Date"] = display_df["Date"].dt.strftime("%d %b %Y")
display_df = display_df.drop(columns=["Day"], errors="ignore")

# Reorder: Date first
cols_order = ["Date", "Open", "High", "Low", "Close"] + \
    [c for c in display_df.columns if c not in {"Date","Open","High","Low","Close"}]
display_df = display_df[[c for c in cols_order if c in display_df.columns]]

st.dataframe(
    display_df.style.format(
        {c: "₹{:,.2f}" for c in ["Open","High","Low","Close"] if c in display_df.columns}
    ).set_properties(**{
        "background-color": "#0d1424",
        "color": "#e8edf8",
        "border": "1px solid #1e2d4a",
        "font-family": "DM Mono, monospace",
        "font-size": "0.8rem",
    }),
    use_container_width=True,
    height=320,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            font-family:'DM Mono',monospace;font-size:0.7rem;color:#3a4d6a;
            padding:0.2rem 0 0.8rem;">
    <span>PolyStock · Polynomial Regression Dashboard</span>
    <span>Degree: {poly_degree} · R²: {r2:.4f} · MAE: ₹{mae:.2f} · 30-day window</span>
    <span>Built with Streamlit · Plotly · Scikit-Learn</span>
</div>
""", unsafe_allow_html=True)