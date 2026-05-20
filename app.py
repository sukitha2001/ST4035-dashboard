import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import statsmodels.api as sm
from scipy.stats import norm
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inv_probit(x):
    return norm.cdf(x)


def inv_cloglog(x):
    return 1 - np.exp(-np.exp(x))


def generate_data(n, minority_prop, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X1 = np.random.normal(0, 1, size=n)
    X2 = np.random.binomial(1, 0.5, size=n)
    beta1 = 1.0
    beta2 = -0.5

    def logit(p):
        return np.log(p / (1 - p))

    beta0 = logit(minority_prop / 100.0) - beta2 * 0.5
    eta = beta0 + beta1 * X1 + beta2 * X2
    p = sigmoid(eta)
    y = np.random.binomial(1, p)
    X = pd.DataFrame({"X1": X1, "X2": X2})
    return X, pd.Series(y, name="y")


def apply_balancing(X, y, method):
    if method == "None":
        return X.reset_index(drop=True), y.reset_index(drop=True)
    elif method == "Oversampling":
        ros = RandomOverSampler(sampling_strategy=1.0, random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
    elif method == "Undersampling":
        rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
    elif method == "SMOTE":
        smote = SMOTE(sampling_strategy=0.85, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
    else:
        return X, y


link_map = {
    "Logit": sm.families.links.logit(),
    "Probit": sm.families.links.probit(),
    "Complementary log-log": sm.families.links.cloglog(),
}


st.set_page_config(
    page_title="Binary Logistic Regression Explorer",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #f0f4ff; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1440px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 320px !important;
    min-width: 320px !important;
    max-width: 320px !important;
    height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    transform: none !important;
    visibility: visible !important;
    background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%) !important;
    border-right: none;
    box-shadow: 4px 0 24px rgba(0,0,0,0.18);
    z-index: 1000 !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #6366f1 !important;
    border-color: #6366f1 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 6px 12px;
    margin-bottom: 4px;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(99,102,241,0.35) !important;
}

/* ── Content shift for fixed sidebar ── */
section[data-testid="stAppViewContainer"],
div[data-testid="stAppViewContainer"],
.main, main {
    margin-left: 340px !important;
    width: calc(100% - 340px) !important;
    padding-left: 0 !important;
}
section[data-testid="stAppViewContainer"] > main,
div[data-testid="stAppViewContainer"] > main,
section[data-testid="stAppViewContainer"] > main > div,
div[data-testid="stAppViewContainer"] > main > div,
.main .block-container,
main > div {
    margin-left: 0 !important;
    padding-left: 0 !important;
}

/* ── Responsive fallback for smaller widths */
@media (max-width: 1100px) {
    section[data-testid="stSidebar"] {
        position: relative !important;
        width: auto !important;
        min-width: auto !important;
        height: auto !important;
        box-shadow: none !important;
    }
    section[data-testid="stAppViewContainer"] {
        margin-left: 0 !important;
        width: auto !important;
    }
}

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 60%, #a855f7 100%);
    border-radius: 20px;
    padding: 36px 40px 32px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(79,70,229,0.28);
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.page-header::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 260px; height: 260px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 6px;
    letter-spacing: -0.03em;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
    font-weight: 400;
}

/* ── Stat chips in the header ── */
.chip-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 20px; }
.chip {
    background: rgba(255,255,255,0.14);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #ffffff;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.chip-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.chip-blue   { background-color: #60a5fa; }
.chip-orange { background-color: #fb923c; }
.chip-green  { background-color: #34d399; }
.chip-purple { background-color: #c084fc; }

.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
.metric-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 22px 24px 18px;
    border: 1px solid #e8eaf6;
    box-shadow: 0 2px 12px rgba(79,70,229,0.06);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(79,70,229,0.14);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    border-radius: 16px 16px 0 0;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
    letter-spacing: -0.03em;
    line-height: 1;
}
.metric-sub {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 6px;
}

/* ── Section header ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #334155;
    letter-spacing: -0.01em;
    padding: 0 0 12px;
    border-bottom: 2px solid #e8eaf6;
    margin-bottom: 20px;
}

/* ── Chart card ── */
.chart-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px 20px 8px;
    border: 1px solid #e8eaf6;
    box-shadow: 0 2px 12px rgba(79,70,229,0.05);
    margin-bottom: 20px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.chart-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(79,70,229,0.10);
}

/* ── Footer summary strip ── */
.summary-strip {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-top: 8px;
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.summary-item { text-align: center; }
.summary-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 6px;
}
.summary-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}

/* ── Hide Streamlit chrome ── */
header, footer, #MainMenu { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# Plotly config
config_plot = {"displayModeBar": True, "responsive": True}
PLOT_FONT   = dict(size=14, family="Inter, sans-serif", color="#334155")
TITLE_FONT  = dict(size=16, family="Inter, sans-serif", color="#1e293b")
LEGEND_FONT = dict(size=12, family="Inter, sans-serif")

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(248,250,252,1)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=PLOT_FONT,
    title_font=TITLE_FONT,
    legend=dict(font=LEGEND_FONT, bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(font=dict(size=13, family="Inter, sans-serif"), bgcolor="#1e293b", font_color="#f1f5f9"),
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    margin=dict(t=60, l=20, r=20, b=20),
)

# Sidebar controls 
st.sidebar.markdown("## Controls")
st.sidebar.markdown("---")
st.sidebar.markdown("**Data**")
N = st.sidebar.slider("Sample size (n)", min_value=100, max_value=2000, step=50, value=500)
minority_pct = st.sidebar.slider("Minority class %", min_value=5, max_value=50, step=5, value=10)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modelling**")
balancing = st.sidebar.radio("Balancing method", ("None", "Oversampling", "Undersampling", "SMOTE"))
link_choice = st.sidebar.radio("Link function", ("Logit", "Probit", "Complementary log-log"))


# Data generation
X_orig, y_orig = generate_data(N, minority_pct, seed=42)
orig_counts = y_orig.value_counts().to_dict()
orig_min = int(orig_counts.get(1, 0))
orig_maj = int(orig_counts.get(0, 0))

X_bal, y_bal = apply_balancing(X_orig, y_orig, balancing)
bal_counts = y_bal.value_counts().to_dict()
bal_min = int(bal_counts.get(1, 0))
bal_maj = int(bal_counts.get(0, 0))
ratio_str = "1 : Inf" if bal_min == 0 else f"1 : {bal_maj / bal_min:.2f}"

# Model fitting 
X_design = sm.add_constant(X_bal)
try:
    model = sm.GLM(y_bal, X_design, family=sm.families.Binomial(link=link_map[link_choice])).fit()
    probs = model.predict(X_design)
except Exception as e:
    st.error(f"Model fitting failed: {e}")
    probs = np.zeros(len(y_bal))

preds = (probs >= 0.5).astype(int)
fpr_arr, tpr_arr, _ = roc_curve(y_bal, probs)
roc_auc = auc(fpr_arr, tpr_arr)
accuracy = accuracy_score(y_bal, preds)
tn, fp, fn, tp = confusion_matrix(y_bal, preds).ravel()
tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
fnr_val = fn / (tp + fn) if (tp + fn) > 0 else 0
tnr_val = tn / (tn + fp) if (tn + fp) > 0 else 0
fpr_val = fp / (tn + fp) if (tn + fp) > 0 else 0

# Page header
st.markdown("""
<div class="page-header">
    <div class="page-title">📊 Binary Logistic Regression Explorer</div>
</div>
""", unsafe_allow_html=True)

# KPI cards
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">Total Samples</div>
        <div class="metric-value">{len(y_bal):,}</div>
        <div class="metric-sub">after balancing</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Minority Class (1)</div>
        <div class="metric-value">{bal_min:,}</div>
        <div class="metric-sub">{bal_min/len(y_bal)*100:.1f}% of total</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Majority Class (0)</div>
        <div class="metric-value">{bal_maj:,}</div>
        <div class="metric-sub">{bal_maj/len(y_bal)*100:.1f}% of total</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Imbalance Ratio</div>
        <div class="metric-value" style="font-size:1.5rem">{ratio_str}</div>
        <div class="metric-sub">minority : majority</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Link & Balancing
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Link Function</div>
        <div class="metric-value">{link_choice}</div>
        <div class="metric-sub">current link</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Balancing</div>
        <div class="metric-value">{balancing}</div>
        <div class="metric-sub">resampling method</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-header">📈 Data Distribution &amp; Link Functions</div>', unsafe_allow_html=True)
r1c1, r1c2 = st.columns(2)

with r1c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Class 0", x=["Original", "Balanced"], y=[orig_maj, bal_maj],
                         marker_color="#6366f1", marker_line_width=0, opacity=0.9))
    fig.add_trace(go.Bar(name="Class 1", x=["Original", "Balanced"], y=[orig_min, bal_min],
                         marker_color="#f59e0b", marker_line_width=0, opacity=0.9))

    layout_extra = {}
    if balancing == "SMOTE":
        smote_target = 0.85 * orig_maj
        layout_extra["shapes"] = [dict(
            type="line",
            x0=-0.5, x1=1.5,
            y0=smote_target, y1=smote_target,
            line=dict(color="#ef4444", width=2, dash="dash"),
            layer="above",
        )]
        layout_extra["annotations"] = [dict(
            x=1.45, y=smote_target,
            xref="x", yref="y",
            text=f"<b>SMOTE target</b><br>85% of majority<br>({int(smote_target):,})",
            showarrow=True, arrowhead=2, arrowcolor="#ef4444",
            ax=50, ay=-28,
            font=dict(size=11, color="#ef4444", family="Inter"),
            align="left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ef4444",
            borderwidth=1,
            borderpad=4,
        )]

    fig.update_layout(**{**CHART_LAYOUT,
        "barmode": "group",
        "title_text": "Class Distribution — Before & After Balancing",
        "legend": dict(font=LEGEND_FONT, bgcolor="rgba(0,0,0,0)", orientation="h",
                       yanchor="bottom", y=1.02, xanchor="right", x=1),
        **layout_extra,
    })
    st.plotly_chart(fig, use_container_width=True, config=config_plot)

with r1c2:
    eta = np.linspace(-4, 4, 400)
    styles = {
        "Logit":                  {"dash": "solid",   "color": "#6366f1"},
        "Probit":                 {"dash": "dash",    "color": "#10b981"},
        "Complementary log-log":  {"dash": "dot",     "color": "#ef4444"},
    }
    fig2 = go.Figure()
    for name, yvals in [("Logit", sigmoid(eta)), ("Probit", inv_probit(eta)),
                         ("Complementary log-log", inv_cloglog(eta))]:
        fig2.add_trace(go.Scatter(x=eta, y=yvals, mode="lines", name=name,
                                  line=dict(width=4 if link_choice == name else 1.5,
                                            dash=styles[name]["dash"], color=styles[name]["color"])))
    fig2.update_layout(**{**CHART_LAYOUT,
        "title_text": "Inverse Link Functions  P(Y=1 | η)",
        "xaxis_title": "η (linear predictor)",
        "yaxis_title": "P(Y=1)",
        "legend": dict(font=LEGEND_FONT, bgcolor="rgba(0,0,0,0)", orientation="h",
                       yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        "margin": dict(t=60, l=20, r=20, b=60),
    })
    st.plotly_chart(fig2, use_container_width=True, config=config_plot)

# ROC & Confusion Matrix
st.markdown('<div class="section-header">🎯 Model Diagnostics</div>', unsafe_allow_html=True)
r2c1, r2c2 = st.columns(2)

with r2c1:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=fpr_arr, y=tpr_arr, mode="lines", fill="tozeroy",
                              fillcolor="rgba(99,102,241,0.08)",
                              name=f"ROC  (AUC = {roc_auc:.3f})",
                              line=dict(color="#6366f1", width=2.5)))
    fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random classifier",
                              line=dict(color="#94a3b8", dash="dash", width=1.5)))
    fig3.update_layout(**{**CHART_LAYOUT,
        "title_text": "ROC Curve",
        "xaxis_title": "False Positive Rate (1 − Specificity)",
        "yaxis_title": "True Positive Rate (Sensitivity)",
        "legend": dict(font=LEGEND_FONT, bgcolor="rgba(0,0,0,0)", orientation="h",
                       yanchor="bottom", y=1.02, xanchor="right", x=1),
    })
    st.plotly_chart(fig3, use_container_width=True, config=config_plot)

with r2c2:
    z = [[tpr_val, fnr_val], [fpr_val, tnr_val]]
    fig4 = go.Figure(data=go.Heatmap(
        z=z, x=["Pred = 1", "Pred = 0"], y=["Actual = 1", "Actual = 0"],
        colorscale=[[0, "#ede9fe"], [0.5, "#818cf8"], [1, "#3730a3"]],
        zmin=0, zmax=1, showscale=True,
        hovertemplate="Rate: %{z:.2f}<extra></extra>",
    ))
    labels = [["TPR", "FNR"], ["FPR", "TNR"]]
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            fig4.add_annotation(
                x=["Pred = 1", "Pred = 0"][j], y=["Actual = 1", "Actual = 0"][i],
                text=f"<b>{labels[i][j]}</b><br>{val:.2f}",
                showarrow=False,
                font=dict(color="white" if val > 0.45 else "#1e293b", size=15, family="Inter"),
            )
    fig4.update_layout(**{**CHART_LAYOUT,
        "title_text": "Confusion Matrix (Normalised Rates)",
    })
    st.plotly_chart(fig4, use_container_width=True, config=config_plot)

# Footer summary strip
st.markdown(f"""
<div class="summary-strip">
    <div class="summary-item">
        <div class="summary-label">AUC</div>
        <div class="summary-value">{roc_auc:.3f}</div>
    </div>
    <div class="summary-item">
        <div class="summary-label">Accuracy</div>
        <div class="summary-value">{accuracy:.3f}</div>
    </div>
    <div class="summary-item">
        <div class="summary-label">Sensitivity</div>
        <div class="summary-value">{tpr_val:.3f}</div>
    </div>
    <div class="summary-item">
        <div class="summary-label">Specificity</div>
        <div class="summary-value">{tnr_val:.3f}</div>
    </div>
    <div class="summary-item">
        <div class="summary-label">Link Function</div>
        <div class="summary-value" style="font-size:1rem;padding-top:4px">{link_choice}</div>
    </div>
    <div class="summary-item">
        <div class="summary-label">Balancing</div>
        <div class="summary-value" style="font-size:1rem;padding-top:4px">{balancing}</div>
    </div>
</div>
""", unsafe_allow_html=True)
