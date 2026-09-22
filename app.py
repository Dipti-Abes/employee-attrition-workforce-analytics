"""
Employee Attrition & Workforce Intelligence System
An AI-Powered HR Analytics Dashboard for Employee Attrition Analysis,
Risk Prediction and Workforce Insights

IBM HR Analytics Dataset — Streamlit Application
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="Employee Attrition & Workforce Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ───────────────────────────────
st.markdown("""
<style>
/* ── Main area ── */
.main { background-color: #f0f2f6; }

/* ── Sidebar collapse/expand button — always visible ── */
[data-testid="stSidebarCollapsedControl"] {
    background-color: #e94560 !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    fill: white !important;
    color: white !important;
}
button[data-testid="baseButton-header"] {
    color: white !important;
}

/* ── Sidebar dark theme ── */
[data-testid="stSidebar"] {
    background-color: #1a1a2e !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown strong,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption p,
[data-testid="stSidebar"] span[data-testid="stWidgetLabel"],
[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] {
    color: #e0e0e0 !important;
}
/* Radio option text */
[data-testid="stSidebar"] [data-testid="stRadio"] span p,
[data-testid="stSidebar"] [data-testid="stRadio"] div p {
    color: #e0e0e0 !important;
    font-size: 0.88rem !important;
}
/* Selected radio highlight */
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] ~ div p {
    color: #e94560 !important;
    font-weight: 600 !important;
}
/* Selectbox & slider text */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] span {
    color: #1f2328 !important;
}
/* HR dividers */
[data-testid="stSidebar"] hr {
    border-color: #2a2a4a !important;
}

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px; padding: 20px 16px; text-align: center;
    border-left: 4px solid #e94560;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15); margin-bottom: 8px;
}
.kpi-card .kpi-value { font-size: 2rem; font-weight: 700; color: #e94560; line-height: 1.1; }
.kpi-card .kpi-label { font-size: 0.75rem; color: #a8b2d8; margin-top: 4px;
    text-transform: uppercase; letter-spacing: 0.8px; }
.kpi-card .kpi-sub { font-size: 0.70rem; color: #7a859e; margin-top: 2px; }

/* ── Section header ── */
.section-header {
    background: linear-gradient(90deg, #0f3460, #16213e);
    color: white; padding: 10px 18px; border-radius: 8px;
    font-size: 1.1rem; font-weight: 600; margin: 16px 0 12px 0;
}

/* ── Insight box ── */
.insight-box {
    background: #eef2ff; border-left: 4px solid #3b5bdb;
    border-radius: 6px; padding: 10px 14px; margin: 6px 0;
    font-size: 0.87rem; color: #1a1a2e;
}

/* ── Risk result boxes ── */
.risk-high {
    background: linear-gradient(135deg, #ff6b6b, #c0392b);
    color: white; padding: 18px; border-radius: 12px;
    text-align: center; font-size: 1.3rem; font-weight: 700;
}
.risk-low {
    background: linear-gradient(135deg, #51cf66, #27ae60);
    color: white; padding: 18px; border-radius: 12px;
    text-align: center; font-size: 1.3rem; font-weight: 700;
}
.risk-med {
    background: linear-gradient(135deg, #ffa94d, #e67e22);
    color: white; padding: 18px; border-radius: 12px;
    text-align: center; font-size: 1.3rem; font-weight: 700;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── HELPER: KPI CARD ─────────────────────────
def kpi_card(value, label, sub=""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True
    )

def section_header(text):
    st.markdown(f'<div class="section-header">📊 {text}</div>', unsafe_allow_html=True)

def insight_box(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


# ── DATA LOADING & PREPROCESSING ─────────────
@st.cache_data
def load_data():
    """Load the IBM HR Attrition dataset."""
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df


@st.cache_data
def preprocess_data(df):
    """
    Clean and preprocess. Drop constant columns (EmployeeCount, Over18, StandardHours).
    Returns: df_clean (human-readable), df_encoded (ML-ready), le_dict, cat_cols.
    """
    df_clean = df.copy()
    constant_cols = ["EmployeeCount", "Over18", "StandardHours"]
    df_clean.drop(columns=constant_cols, inplace=True)
    df_clean["AttritionFlag"] = (df_clean["Attrition"] == "Yes").astype(int)

    cat_cols = [c for c in df_clean.select_dtypes(include="object").columns if c != "Attrition"]

    le_dict = {}
    df_encoded = df_clean.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le

    return df_clean, df_encoded, le_dict, cat_cols


@st.cache_resource
def train_models(_df_encoded):
    """
    Train 4 classifiers with class_weight='balanced' to handle imbalance.
    Returns trained pipelines, metrics, train/test splits, feature names.
    Note: _df_encoded prefixed with _ so Streamlit doesn't hash the DataFrame arg.
    """
    df_encoded = _df_encoded
    drop_cols = ["Attrition", "AttritionFlag", "EmployeeNumber"]
    feature_cols = [c for c in df_encoded.columns if c not in drop_cols]
    X = df_encoded[feature_cols].values
    y = df_encoded["AttritionFlag"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(class_weight="balanced", max_depth=6, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                max_depth=8, random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
            ))
        ]),
    }

    results = {}
    trained = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
            "y_pred": y_pred,
            "y_prob":  y_prob,
        }
        trained[name] = pipe

    return trained, results, X_train, X_test, y_train, y_test, feature_cols


def _safe_kmeans_fit(km, X):
    """
    Runs KMeans/MiniBatchKMeans fit_predict while patching around the
    threadpoolctl 'NoneType has no attribute split' bug present in some
    Anaconda installations (broken threadpoolctl version with MKL).
    Patches threadpool_info and threadpool_limits directly on the
    sklearn.cluster._kmeans module for the duration of this call only.
    """
    import sklearn.cluster._kmeans as _km_mod

    # Save originals
    _orig_info   = _km_mod.threadpool_info
    _orig_limits = _km_mod.threadpool_limits

    # No-op replacements
    def _noop_info():
        return []

    class _DummyLimits:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    try:
        _km_mod.threadpool_info   = _noop_info
        _km_mod.threadpool_limits = _DummyLimits
        labels = km.fit_predict(X)
    finally:
        _km_mod.threadpool_info   = _orig_info
        _km_mod.threadpool_limits = _orig_limits
    return labels, km.inertia_


@st.cache_data
def run_clustering(_df_encoded):
    """
    MiniBatchKMeans clustering on scaled numeric workforce features.
    Determines optimal k via elbow + silhouette (k=2..6).
    Returns cluster labels, silhouette scores, inertias, pca coords.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import MiniBatchKMeans
    df_encoded = _df_encoded

    cluster_features = [
        "Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
        "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
        "JobLevel", "DistanceFromHome", "NumCompaniesWorked"
    ]
    cluster_features = [f for f in cluster_features if f in df_encoded.columns]
    X_cl = df_encoded[cluster_features].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cl)

    inertias, sil_scores = [], []
    k_range = range(2, 7)
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
        labels, inertia = _safe_kmeans_fit(km, X_scaled)
        inertias.append(inertia)
        sil_scores.append(silhouette_score(X_scaled, labels))

    best_k = list(k_range)[np.argmax(sil_scores)]
    km_final = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels, _ = _safe_kmeans_fit(km_final, X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    return cluster_labels, sil_scores, inertias, list(k_range), coords, best_k, cluster_features


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════
def page_overview(df, df_filtered):
    st.markdown("## 🏢 Executive Overview Dashboard")
    st.caption("High-level KPIs and workforce snapshot based on active filters.")

    total = len(df_filtered)
    if total == 0:
        st.warning("No records match the current filters. Please adjust your selections.")
        return

    left_n   = df_filtered[df_filtered["Attrition"] == "Yes"].shape[0]
    stayed_n = df_filtered[df_filtered["Attrition"] == "No"].shape[0]
    attr_rate = (left_n / total * 100) if total > 0 else 0
    avg_income = df_filtered["MonthlyIncome"].mean()
    avg_age    = df_filtered["Age"].mean()
    avg_tenure = df_filtered["YearsAtCompany"].mean()
    avg_satisfaction = df_filtered["JobSatisfaction"].mean()

    # ── KPI ROW ──
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1: kpi_card(f"{total:,}", "Total Employees")
    with c2: kpi_card(f"{left_n:,}", "Employees Left", "Attrition = Yes")
    with c3: kpi_card(f"{stayed_n:,}", "Employees Retained", "Attrition = No")
    with c4: kpi_card(f"{attr_rate:.1f}%", "Attrition Rate")
    with c5: kpi_card(f"${avg_income:,.0f}", "Avg Monthly Income")
    with c6: kpi_card(f"{avg_age:.1f}", "Avg Age", "years")
    with c7: kpi_card(f"{avg_tenure:.1f}", "Avg Tenure", "years at company")

    st.markdown("---")

    # ── ROW 2: Attrition donut + Dept distribution ──
    section_header("Workforce Composition")
    col1, col2, col3 = st.columns(3)

    with col1:
        att_counts = df_filtered["Attrition"].value_counts().reset_index()
        att_counts.columns = ["Attrition", "Count"]
        fig = px.pie(att_counts, names="Attrition", values="Count",
                     hole=0.55, color="Attrition",
                     color_discrete_map={"Yes": "#e94560", "No": "#51cf66"},
                     title="Attrition Distribution")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dept_counts = df_filtered["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]
        fig = px.bar(dept_counts, x="Count", y="Department", orientation="h",
                     color="Count", color_continuous_scale="Blues",
                     title="Employees by Department")
        fig.update_layout(coloraxis_showscale=False, height=300,
                          margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        gender_counts = df_filtered["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig = px.pie(gender_counts, names="Gender", values="Count",
                     hole=0.5, color="Gender",
                     color_discrete_map={"Male": "#4c9be8", "Female": "#e96fa3"},
                     title="Gender Distribution")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        jrole = df_filtered["JobRole"].value_counts().reset_index()
        jrole.columns = ["JobRole", "Count"]
        fig = px.bar(jrole, x="Count", y="JobRole", orientation="h",
                     color="Count", color_continuous_scale="Purples",
                     title="Employees by Job Role")
        fig.update_layout(coloraxis_showscale=False, height=340,
                          margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        ot = df_filtered["OverTime"].value_counts().reset_index()
        ot.columns = ["OverTime", "Count"]
        fig = px.pie(ot, names="OverTime", values="Count", hole=0.5,
                     color="OverTime",
                     color_discrete_map={"Yes": "#f59f00", "No": "#74c0fc"},
                     title="OverTime Distribution")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, height=340, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        bt = df_filtered["BusinessTravel"].value_counts().reset_index()
        bt.columns = ["BusinessTravel", "Count"]
        fig = px.bar(bt, x="BusinessTravel", y="Count",
                     color="Count", color_continuous_scale="Oranges",
                     title="Business Travel Frequency")
        fig.update_layout(coloraxis_showscale=False, height=340,
                          margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Age & Income distributions ──
    section_header("Demographic Distributions")
    col7, col8 = st.columns(2)
    with col7:
        fig = px.histogram(df_filtered, x="Age", nbins=20, color="Attrition",
                           barmode="overlay",
                           color_discrete_map={"Yes": "#e94560", "No": "#51cf66"},
                           title="Age Distribution by Attrition",
                           labels={"Age": "Age (years)"})
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        fig = px.histogram(df_filtered, x="MonthlyIncome", nbins=25, color="Attrition",
                           barmode="overlay",
                           color_discrete_map={"Yes": "#e94560", "No": "#51cf66"},
                           title="Monthly Income Distribution by Attrition",
                           labels={"MonthlyIncome": "Monthly Income ($)"})
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — WORKFORCE ANALYTICS
# ═══════════════════════════════════════════════════════════════
def page_workforce(df, df_filtered):
    st.markdown("## 👥 Workforce Analytics")
    st.caption("Detailed workforce statistics and demographic breakdowns.")

    if len(df_filtered) == 0:
        st.warning("No records match the current filters.")
        return

    section_header("Satisfaction Levels Overview")
    sat_cols = {
        "JobSatisfaction": "Job Satisfaction",
        "EnvironmentSatisfaction": "Environment Satisfaction",
        "RelationshipSatisfaction": "Relationship Satisfaction",
        "WorkLifeBalance": "Work-Life Balance",
        "JobInvolvement": "Job Involvement",
    }
    rating_labels = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
    col1, col2 = st.columns(2)
    pairs = list(sat_cols.items())
    for i, (col_name, label) in enumerate(pairs[:4]):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            data = df_filtered[col_name].value_counts().sort_index().reset_index()
            data.columns = ["Rating", "Count"]
            data["Label"] = data["Rating"].map(rating_labels).fillna(data["Rating"].astype(str))
            fig = px.bar(data, x="Label", y="Count",
                         color="Count", color_continuous_scale="Blues",
                         title=label)
            fig.update_layout(coloraxis_showscale=False, height=280,
                               margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    section_header("Job & Compensation Analysis")
    col3, col4 = st.columns(2)
    with col3:
        jl = df_filtered.groupby("JobLevel")["MonthlyIncome"].mean().reset_index()
        jl.columns = ["JobLevel", "AvgIncome"]
        fig = px.bar(jl, x="JobLevel", y="AvgIncome",
                     color="AvgIncome", color_continuous_scale="Greens",
                     title="Average Monthly Income by Job Level",
                     labels={"AvgIncome": "Avg Monthly Income ($)", "JobLevel": "Job Level"})
        fig.update_layout(coloraxis_showscale=False, height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        ms_count = df_filtered["MaritalStatus"].value_counts().reset_index()
        ms_count.columns = ["MaritalStatus", "Count"]
        fig = px.pie(ms_count, names="MaritalStatus", values="Count",
                     hole=0.45, title="Marital Status Distribution")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        fig = px.box(df_filtered, x="Department", y="MonthlyIncome",
                     color="Department", title="Monthly Income by Department",
                     labels={"MonthlyIncome": "Monthly Income ($)"})
        fig.update_layout(showlegend=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        fig = px.scatter(df_filtered, x="TotalWorkingYears", y="MonthlyIncome",
                         color="Attrition",
                         color_discrete_map={"Yes": "#e94560", "No": "#51cf66"},
                         opacity=0.6,
                         title="Experience vs Monthly Income",
                         labels={"TotalWorkingYears": "Total Working Years",
                                 "MonthlyIncome": "Monthly Income ($)"})
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    section_header("Education & Training")
    col7, col8 = st.columns(2)
    edu_map = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
    with col7:
        ef = df_filtered["EducationField"].value_counts().reset_index()
        ef.columns = ["EducationField", "Count"]
        fig = px.bar(ef, x="Count", y="EducationField", orientation="h",
                     color="Count", color_continuous_scale="Tealgrn",
                     title="Education Field Distribution")
        fig.update_layout(coloraxis_showscale=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        tr = df_filtered["TrainingTimesLastYear"].value_counts().sort_index().reset_index()
        tr.columns = ["TrainingTimes", "Count"]
        fig = px.bar(tr, x="TrainingTimes", y="Count",
                     color="Count", color_continuous_scale="Oryel",
                     title="Training Times Last Year",
                     labels={"TrainingTimes": "Training Times"})
        fig.update_layout(coloraxis_showscale=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — ATTRITION ANALYSIS
# ═══════════════════════════════════════════════════════════════
def page_attrition(df, df_filtered):
    st.markdown("## 📉 Attrition Analysis")
    st.caption("Detailed breakdown of attrition patterns across workforce dimensions.")

    if len(df_filtered) == 0:
        st.warning("No records match the current filters.")
        return

    def attrition_rate_df(group_col):
        grp = df_filtered.groupby(group_col)["AttritionFlag"].agg(["sum", "count"]).reset_index()
        grp.columns = [group_col, "Left", "Total"]
        grp["AttritionRate"] = (grp["Left"] / grp["Total"] * 100).round(1)
        return grp

    section_header("Attrition by Department & Role")
    col1, col2 = st.columns(2)
    with col1:
        d = attrition_rate_df("Department")
        fig = px.bar(d, x="Department", y="AttritionRate",
                     color="AttritionRate", color_continuous_scale="Reds",
                     text="AttritionRate",
                     title="Attrition Rate by Department (%)",
                     labels={"AttritionRate": "Attrition Rate (%)"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        d = attrition_rate_df("JobRole")
        d = d.sort_values("AttritionRate", ascending=True)
        fig = px.bar(d, x="AttritionRate", y="JobRole", orientation="h",
                     color="AttritionRate", color_continuous_scale="Reds",
                     text="AttritionRate",
                     title="Attrition Rate by Job Role (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    section_header("Attrition by Work Factors")
    col3, col4, col5 = st.columns(3)
    with col3:
        d = attrition_rate_df("OverTime")
        fig = px.bar(d, x="OverTime", y="AttritionRate",
                     color="OverTime",
                     color_discrete_map={"Yes": "#e94560", "No": "#51cf66"},
                     text="AttritionRate",
                     title="Attrition by OverTime (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        d = attrition_rate_df("BusinessTravel")
        fig = px.bar(d, x="BusinessTravel", y="AttritionRate",
                     color="AttritionRate", color_continuous_scale="OrRd",
                     text="AttritionRate",
                     title="Attrition by Business Travel (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        d = attrition_rate_df("JobLevel")
        fig = px.bar(d, x="JobLevel", y="AttritionRate",
                     color="AttritionRate", color_continuous_scale="RdPu",
                     text="AttritionRate",
                     title="Attrition by Job Level (%)",
                     labels={"JobLevel": "Job Level", "AttritionRate": "Attrition Rate (%)"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    section_header("Attrition by Satisfaction Ratings")
    col6, col7, col8, col9 = st.columns(4)
    sat_vars = [
        ("JobSatisfaction", "Job Satisfaction"),
        ("EnvironmentSatisfaction", "Environment Satisfaction"),
        ("WorkLifeBalance", "Work-Life Balance"),
        ("JobInvolvement", "Job Involvement"),
    ]
    for idx, (var, lbl) in enumerate(sat_vars):
        target = [col6, col7, col8, col9][idx]
        with target:
            d = attrition_rate_df(var)
            fig = px.line(d, x=var, y="AttritionRate", markers=True,
                          color_discrete_sequence=["#e94560"],
                          title=f"Attrition vs {lbl}",
                          labels={var: lbl, "AttritionRate": "Attrition Rate (%)"})
            fig.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    section_header("Attrition by Age, Income, and Tenure")
    col10, col11, col12 = st.columns(3)
    with col10:
        df_temp = df_filtered.copy()
        df_temp["AgeGroup"] = pd.cut(df_temp["Age"],
                                      bins=[17, 25, 30, 35, 40, 45, 60],
                                      labels=["18-25", "26-30", "31-35", "36-40", "41-45", "46+"])
        d = df_temp.groupby("AgeGroup", observed=True)["AttritionFlag"].agg(["sum", "count"]).reset_index()
        d.columns = ["AgeGroup", "Left", "Total"]
        d["AttritionRate"] = (d["Left"] / d["Total"] * 100).round(1)
        fig = px.bar(d, x="AgeGroup", y="AttritionRate",
                     color="AttritionRate", color_continuous_scale="Reds",
                     text="AttritionRate",
                     title="Attrition by Age Group (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col11:
        df_temp2 = df_filtered.copy()
        df_temp2["IncomeGroup"] = pd.cut(df_temp2["MonthlyIncome"],
                                          bins=[0, 3000, 6000, 10000, 20000],
                                          labels=["<3K", "3K-6K", "6K-10K", ">10K"])
        d2 = df_temp2.groupby("IncomeGroup", observed=True)["AttritionFlag"].agg(["sum", "count"]).reset_index()
        d2.columns = ["IncomeGroup", "Left", "Total"]
        d2["AttritionRate"] = (d2["Left"] / d2["Total"] * 100).round(1)
        fig = px.bar(d2, x="IncomeGroup", y="AttritionRate",
                     color="AttritionRate", color_continuous_scale="Oranges",
                     text="AttritionRate",
                     title="Attrition by Income Band (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col12:
        df_temp3 = df_filtered.copy()
        df_temp3["TenureGroup"] = pd.cut(df_temp3["YearsAtCompany"],
                                          bins=[-1, 2, 5, 10, 15, 40],
                                          labels=["0-2", "3-5", "6-10", "11-15", "16+"])
        d3 = df_temp3.groupby("TenureGroup", observed=True)["AttritionFlag"].agg(["sum", "count"]).reset_index()
        d3.columns = ["TenureGroup", "Left", "Total"]
        d3["AttritionRate"] = (d3["Left"] / d3["Total"] * 100).round(1)
        fig = px.bar(d3, x="TenureGroup", y="AttritionRate",
                     color="AttritionRate", color_continuous_scale="RdYlGn_r",
                     text="AttritionRate",
                     title="Attrition by Tenure Group (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    section_header("Distance From Home vs Attrition")
    col13, col14 = st.columns(2)
    with col13:
        fig = px.box(df_filtered, x="Attrition", y="DistanceFromHome",
                     color="Attrition",
                     color_discrete_map={"Yes": "#e94560", "No": "#51cf66"},
                     title="Distance From Home by Attrition Status",
                     labels={"DistanceFromHome": "Distance From Home (km)"})
        fig.update_layout(showlegend=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col14:
        fig = px.box(df_filtered, x="Attrition", y="YearsAtCompany",
                     color="Attrition",
                     color_discrete_map={"Yes": "#e94560", "No": "#51cf66"},
                     title="Years at Company by Attrition Status")
        fig.update_layout(showlegend=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Key observed patterns (calculated, not hard-coded) ──
    section_header("Observed Attrition Patterns")
    ot_rate = df_filtered.groupby("OverTime")["AttritionFlag"].mean() * 100
    bt_rate = df_filtered.groupby("BusinessTravel")["AttritionFlag"].mean() * 100
    dept_rate = df_filtered.groupby("Department")["AttritionFlag"].mean() * 100
    role_rate = df_filtered.groupby("JobRole")["AttritionFlag"].mean() * 100

    if len(ot_rate) >= 2 and "Yes" in ot_rate.index and "No" in ot_rate.index:
        insight_box(
            f"Employees working overtime show an observed attrition rate of "
            f"{ot_rate.get('Yes', 0):.1f}% vs {ot_rate.get('No', 0):.1f}% for those not working overtime "
            f"in this filtered dataset. Note: correlation ≠ causation."
        )
    if len(dept_rate) > 0:
        top_dept = dept_rate.idxmax()
        insight_box(
            f"'{top_dept}' has the highest observed attrition rate "
            f"({dept_rate.max():.1f}%) among the filtered departments."
        )
    if len(role_rate) > 0:
        top_role = role_rate.idxmax()
        insight_box(
            f"'{top_role}' shows the highest attrition rate "
            f"({role_rate.max():.1f}%) among the visible job roles."
        )


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — CORRELATION & FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════
def page_correlation(df_clean, df_filtered):
    st.markdown("## 🔬 Correlation & Feature Analysis")
    st.caption(
        "Numerical correlations and feature distributions. "
        "Correlation indicates statistical association — it does not prove causation."
    )

    if len(df_filtered) < 5:
        st.warning("Insufficient data for correlation analysis with current filters.")
        return

    num_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    drop_ids = ["EmployeeNumber", "AttritionFlag"]
    num_cols = [c for c in num_cols if c not in drop_ids]

    section_header("Correlation Heatmap")
    corr = df_filtered[num_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Pearson Correlation Matrix — Numerical Features"
    )
    fig.update_layout(height=600, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    ⚠️ <strong>Important:</strong> Correlation measures linear association between variables.
    A high correlation with attrition indicates a statistical relationship in this dataset —
    it does not imply that changing that variable would cause or prevent attrition.
    Always interpret correlations in the context of domain knowledge and business judgment.
    </div>
    """, unsafe_allow_html=True)

    section_header("Top Features Correlated with Attrition")
    corr_with_target = df_filtered[num_cols + ["AttritionFlag"]].corr()["AttritionFlag"].drop("AttritionFlag")
    corr_with_target = corr_with_target.abs().sort_values(ascending=False).head(15).reset_index()
    corr_with_target.columns = ["Feature", "AbsCorrelation"]
    corr_with_target["Correlation"] = df_filtered[num_cols + ["AttritionFlag"]].corr()["AttritionFlag"].drop("AttritionFlag").loc[corr_with_target["Feature"]].values

    fig = px.bar(corr_with_target, x="AbsCorrelation", y="Feature", orientation="h",
                 color="Correlation", color_continuous_scale="RdBu_r",
                 title="Top Features by Absolute Correlation with Attrition",
                 labels={"AbsCorrelation": "|Correlation|"})
    fig.update_layout(height=420, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    section_header("Distribution Comparison: Left vs Stayed")
    key_vars = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
                "DistanceFromHome", "JobSatisfaction", "WorkLifeBalance"]
    key_vars = [v for v in key_vars if v in df_filtered.columns]

    selected_var = st.selectbox("Select variable to compare:", key_vars)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_filtered, x=selected_var, color="Attrition",
                           barmode="overlay", nbins=25,
                           color_discrete_map={"Yes": "#e94560", "No": "#51cf66"},
                           opacity=0.75,
                           title=f"Distribution of {selected_var} — Left vs Stayed")
        fig.update_layout(height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df_filtered, x="Attrition", y=selected_var, color="Attrition",
                     color_discrete_map={"Yes": "#e94560", "No": "#51cf66"},
                     title=f"Box Plot: {selected_var} by Attrition Status",
                     points="outliers")
        fig.update_layout(showlegend=False, height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Descriptive stats ──
    section_header("Descriptive Statistics: Left vs Stayed")
    left_df   = df_filtered[df_filtered["Attrition"] == "Yes"][num_cols]
    stayed_df = df_filtered[df_filtered["Attrition"] == "No"][num_cols]
    stats = pd.DataFrame({
        "Left (mean)":   left_df.mean().round(2),
        "Stayed (mean)": stayed_df.mean().round(2),
        "Diff":          (left_df.mean() - stayed_df.mean()).round(2),
    })
    st.dataframe(stats.style.background_gradient(cmap="RdYlGn_r", subset=["Diff"]),
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — ML MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
def page_ml_performance(trained_models, results, X_test, y_test):
    st.markdown("## 🤖 Machine Learning Model Performance")
    st.caption(
        "Four classifiers trained on 80% of data; evaluated on a stratified 20% hold-out set. "
        "Class imbalance handled via class_weight='balanced'."
    )

    # ── Model comparison table ──
    section_header("Model Comparison Table")
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in res.items() if k not in ("y_pred", "y_prob")}
        for name, res in results.items()
    }).T.reset_index()
    metrics_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    # Highlight best model (by F1 for attrition — a risk problem)
    best_model_name = metrics_df.loc[metrics_df["F1-Score"].idxmax(), "Model"]
    st.dataframe(
        metrics_df.style
            .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                           color="#d4edda")
            .format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]}),
        use_container_width=True
    )
    st.success(
        f"✅ **Selected Model: {best_model_name}** — chosen based on highest F1-Score, "
        f"balancing Precision and Recall for attrition risk identification."
    )

    st.markdown("""
    <div class="insight-box">
    <strong>Why not just maximize Accuracy?</strong><br>
    With ~16% attrition rate, a model predicting "No attrition" for everyone achieves ~84% accuracy.
    For a risk-identification problem, Recall (catching actual at-risk employees) and F1-Score
    (balancing precision and recall) are more meaningful evaluation metrics.
    </div>
    """, unsafe_allow_html=True)

    # ── Metric comparison bar chart ──
    section_header("Metric Comparison Chart")
    metrics_melt = metrics_df.melt(id_vars="Model",
                                    value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                                    var_name="Metric", value_name="Score")
    fig = px.bar(metrics_melt, x="Metric", y="Score", color="Model", barmode="group",
                 title="Model Performance Comparison",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400, margin=dict(t=40, b=10), yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrices ──
    section_header("Confusion Matrices")
    cols = st.columns(4)
    model_names = list(results.keys())
    for i, name in enumerate(model_names):
        with cols[i]:
            cm = confusion_matrix(y_test, results[name]["y_pred"])
            fig = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=["No (0)", "Yes (1)"], y=["No (0)", "Yes (1)"],
                color_continuous_scale="Blues",
                title=name
            )
            fig.update_layout(coloraxis_showscale=False, height=260,
                               margin=dict(t=50, b=5, l=5, r=5))
            st.plotly_chart(fig, use_container_width=True)

    # ── ROC Curves ──
    section_header("ROC Curves")
    from sklearn.metrics import roc_curve
    fig_roc = go.Figure()
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={res['ROC-AUC']:.3f})"
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
        line=dict(dash="dash", color="grey")
    ))
    fig_roc.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420, margin=dict(t=50, b=10)
    )
    st.plotly_chart(fig_roc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 6 — EMPLOYEE RISK PREDICTION
# ═══════════════════════════════════════════════════════════════
def page_prediction(df_clean, trained_models, results, feature_cols):
    st.markdown("## 🎯 Employee Attrition Risk Prediction")
    st.caption(
        "Enter employee profile details to obtain a model-estimated attrition risk. "
        "This prediction is a statistical estimate — not a definitive outcome."
    )

    # Best model = highest F1
    model_metrics = {n: r["F1-Score"] for n, r in results.items()}
    best_model_name = max(model_metrics, key=model_metrics.get)
    best_model = trained_models[best_model_name]

    st.info(f"🔑 Using **{best_model_name}** (highest F1-Score = {model_metrics[best_model_name]:.4f})")

    # ── Input form ──
    st.markdown("### Enter Employee Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 65, 35)
        monthly_income = st.number_input("Monthly Income ($)", 1009, 19999, 5000, step=100)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        total_working_years = st.slider("Total Working Years", 0, 40, 10)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_role = st.slider("Years in Current Role", 0, 18, 3)
        years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)

    with col2:
        dept_options = sorted(df_clean["Department"].unique().tolist())
        department = st.selectbox("Department", dept_options)
        role_options = sorted(df_clean[df_clean["Department"] == department]["JobRole"].unique().tolist())
        job_role = st.selectbox("Job Role", role_options)
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        business_travel = st.selectbox("Business Travel",
                                        sorted(df_clean["BusinessTravel"].unique().tolist()))
        marital_status = st.selectbox("Marital Status",
                                       sorted(df_clean["MaritalStatus"].unique().tolist()))
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col3:
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        work_life_balance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
        job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
        distance_from_home = st.slider("Distance From Home (km)", 1, 29, 5)
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        training_times = st.slider("Training Times Last Year", 0, 6, 3)
        num_companies = st.slider("Num Companies Worked", 0, 9, 2)

    if st.button("🔍 Predict Attrition Risk", use_container_width=True):
        # Build input matching feature_cols
        # We need to replicate the exact columns used in training.
        # Map categorical values via mode-based defaults for unused fields.
        education_val = int(df_clean["Education"].median())
        education_field_val = df_clean["EducationField"].mode()[0]
        daily_rate = int(df_clean["DailyRate"].median())
        hourly_rate = int(df_clean["HourlyRate"].median())
        monthly_rate = int(df_clean["MonthlyRate"].median())
        percent_hike = int(df_clean["PercentSalaryHike"].median())
        performance_rating = int(df_clean["PerformanceRating"].median())
        employee_number = int(df_clean["EmployeeNumber"].median())
        years_with_mgr = min(years_at_company, int(df_clean["YearsWithCurrManager"].median()))

        input_dict = {
            "Age": age,
            "BusinessTravel": business_travel,
            "DailyRate": daily_rate,
            "Department": department,
            "DistanceFromHome": distance_from_home,
            "Education": education_val,
            "EducationField": education_field_val,
            "EmployeeNumber": employee_number,
            "EnvironmentSatisfaction": env_satisfaction,
            "Gender": gender,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies,
            "OverTime": overtime,
            "PercentSalaryHike": percent_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promotion,
            "YearsWithCurrManager": years_with_mgr,
        }

        # Label-encode categorical fields using the same training encoders
        # We need encoders — load them via session state
        le_dict = st.session_state.get("le_dict", {})
        for col, le in le_dict.items():
            if col in input_dict:
                val = str(input_dict[col])
                if val in le.classes_:
                    input_dict[col] = int(le.transform([val])[0])
                else:
                    input_dict[col] = 0  # fallback for unseen labels

        # Build ordered feature array
        input_row = np.array([[input_dict.get(f, 0) for f in feature_cols]])

        prob = best_model.predict_proba(input_row)[0][1]
        pred = best_model.predict(input_row)[0]

        st.markdown("---")
        st.markdown("### Prediction Result")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if prob >= 0.65:
                css_class = "risk-high"
                risk_label = "⚠️ HIGH ATTRITION RISK"
            elif prob >= 0.4:
                css_class = "risk-med"
                risk_label = "⚡ MODERATE ATTRITION RISK"
            else:
                css_class = "risk-low"
                risk_label = "✅ LOW ATTRITION RISK"

            st.markdown(
                f'<div class="{css_class}">{risk_label}<br>'
                f'<span style="font-size:1.6rem">{prob*100:.1f}% Estimated Probability</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Model-Estimated Attrition Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 40],  "color": "#51cf66"},
                        {"range": [40, 65], "color": "#ffa94d"},
                        {"range": [65, 100],"color": "#e94560"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75, "value": prob * 100
                    },
                    "bar": {"color": "#1a1a2e"},
                }
            ))
            fig.update_layout(height=320, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("#### Interpretation")
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Model Used | {best_model_name} |
            | Predicted Class | {"Attrition Likely" if pred == 1 else "Likely to Stay"} |
            | Estimated Probability | {prob*100:.1f}% |
            | Risk Level | {"High" if prob >= 0.65 else ("Moderate" if prob >= 0.4 else "Low")} |
            """)
            st.markdown("""
            <div class="insight-box">
            <strong>⚠️ Disclaimer:</strong> This estimate is generated by a machine learning model
            trained on historical IBM HR data. It represents a statistical pattern — not a certainty.
            Model predictions should inform (not replace) HR professional judgment.
            Individual circumstances may differ significantly from dataset patterns.
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 7 — WORKFORCE SEGMENTATION
# ═══════════════════════════════════════════════════════════════
def page_segmentation(df_clean, df_encoded):
    st.markdown("## 🗂️ Workforce Segmentation")
    st.caption(
        "K-Means clustering applied to key workforce features to identify natural employee segments. "
        "Segment labels are derived from the data — not predetermined."
    )

    cluster_labels, sil_scores, inertias, k_range, coords, best_k, cluster_features = \
        run_clustering(df_encoded)

    # ── Optimal k charts ──
    section_header("Optimal Number of Clusters")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(x=k_range, y=inertias, markers=True,
                      title="Elbow Method — Inertia vs k",
                      labels={"x": "Number of Clusters (k)", "y": "Inertia"})
        fig.update_traces(line_color="#e94560", marker_color="#1a1a2e")
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(x=k_range, y=sil_scores,
                     title="Silhouette Scores vs k",
                     labels={"x": "Number of Clusters (k)", "y": "Silhouette Score"},
                     color=sil_scores, color_continuous_scale="Greens")
        fig.update_layout(coloraxis_showscale=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ Optimal k = **{best_k}** selected based on highest Silhouette Score "
               f"({max(sil_scores):.3f})")

    # Attach cluster labels
    df_viz = df_clean.copy()
    df_viz["Cluster"] = cluster_labels
    df_viz["Cluster"] = df_viz["Cluster"].astype(str)
    df_viz["PCA_1"] = coords[:, 0]
    df_viz["PCA_2"] = coords[:, 1]

    # ── PCA scatter ──
    section_header("Cluster Visualization — PCA 2D Projection")
    fig = px.scatter(
        df_viz, x="PCA_1", y="PCA_2", color="Cluster",
        hover_data=["Age", "MonthlyIncome", "Department", "JobRole", "Attrition"],
        title="Employee Clusters (PCA Projection)",
        color_discrete_sequence=px.colors.qualitative.Set1,
        labels={"PCA_1": "Principal Component 1", "PCA_2": "Principal Component 2"}
    )
    fig.update_layout(height=450, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Cluster profiles ──
    section_header("Cluster Profiles")
    profile_cols = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
                    "JobSatisfaction", "WorkLifeBalance", "JobLevel"]
    profile_cols = [c for c in profile_cols if c in df_viz.columns]

    profile = df_viz.groupby("Cluster")[profile_cols].mean().round(2)
    profile["AttritionRate(%)"] = (df_viz.groupby("Cluster")["AttritionFlag"].mean() * 100).round(1)
    profile["EmployeeCount"] = df_viz.groupby("Cluster").size()

    st.dataframe(profile.style.background_gradient(cmap="YlOrRd", subset=["AttritionRate(%)"]),
                 use_container_width=True)

    # ── Attrition rate by cluster ──
    col3, col4 = st.columns(2)
    with col3:
        cluster_attr = df_viz.groupby("Cluster")["AttritionFlag"].mean().reset_index()
        cluster_attr.columns = ["Cluster", "AttritionRate"]
        cluster_attr["AttritionRate"] *= 100
        fig = px.bar(cluster_attr, x="Cluster", y="AttritionRate",
                     color="AttritionRate", color_continuous_scale="Reds",
                     title="Observed Attrition Rate by Cluster (%)",
                     text="AttritionRate")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(df_viz, x="Cluster", y="MonthlyIncome", color="Cluster",
                     title="Monthly Income Distribution by Cluster")
        fig.update_layout(showlegend=False, height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Auto-describe each cluster ──
    section_header("Cluster Descriptions (Data-Driven)")
    for cid in sorted(df_viz["Cluster"].unique()):
        cdf = df_viz[df_viz["Cluster"] == cid]
        n = len(cdf)
        attr_r = cdf["AttritionFlag"].mean() * 100
        avg_inc = cdf["MonthlyIncome"].mean()
        avg_age = cdf["Age"].mean()
        avg_tenure = cdf["YearsAtCompany"].mean()
        top_dept = cdf["Department"].mode()[0] if len(cdf) > 0 else "N/A"
        insight_box(
            f"**Cluster {cid}** — {n} employees | "
            f"Avg Age: {avg_age:.0f} | Avg Monthly Income: ${avg_inc:,.0f} | "
            f"Avg Tenure: {avg_tenure:.1f} yrs | Dominant Dept: {top_dept} | "
            f"Observed Attrition Rate: {attr_r:.1f}% — "
            f"{'Higher risk segment' if attr_r > 20 else 'Lower risk segment'} (based on this dataset)."
        )


# ═══════════════════════════════════════════════════════════════
# PAGE 8 — WORKFORCE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════
def page_intelligence(df_clean, df_filtered):
    st.markdown("## 💡 Workforce Intelligence")
    st.caption(
        "Automatically calculated insights from the filtered dataset. "
        "All findings are evidence-based and phrased to distinguish observation from causation."
    )

    if len(df_filtered) < 10:
        st.warning("Insufficient data for intelligence analysis with current filters.")
        return

    section_header("Key Attrition Drivers — Observed Patterns")

    # ── OverTime ──
    ot = df_filtered.groupby("OverTime")["AttritionFlag"].agg(["sum", "count"])
    if len(ot) == 2:
        ot_yes = ot.loc["Yes"]["sum"] / ot.loc["Yes"]["count"] * 100 if "Yes" in ot.index else 0
        ot_no  = ot.loc["No"]["sum"]  / ot.loc["No"]["count"]  * 100 if "No"  in ot.index else 0
        insight_box(
            f"Employees working overtime show an observed attrition rate of {ot_yes:.1f}%, "
            f"compared to {ot_no:.1f}% for those not working overtime. "
            f"The difference is {abs(ot_yes - ot_no):.1f} percentage points. "
            f"Note: this is an observed association in this dataset, not a proven causal relationship."
        )

    # ── Business Travel ──
    bt = df_filtered.groupby("BusinessTravel")["AttritionFlag"].mean() * 100
    if len(bt) > 0:
        bt_sorted = bt.sort_values(ascending=False)
        insight_box(
            f"Business travel frequency and attrition rate: "
            + " | ".join([f"{k}: {v:.1f}%" for k, v in bt_sorted.items()])
            + ". Employees who travel frequently show a comparatively higher observed attrition rate."
        )

    # ── Department ──
    dept = df_filtered.groupby("Department")["AttritionFlag"].agg(["sum", "count"])
    dept["Rate"] = dept["sum"] / dept["count"] * 100
    dept = dept.sort_values("Rate", ascending=False)
    if len(dept) > 0:
        insight_box(
            "Department-wise observed attrition rates: "
            + " | ".join([f"{idx}: {row['Rate']:.1f}% ({int(row['sum'])}/{int(row['count'])})"
                          for idx, row in dept.iterrows()])
        )

    # ── Job Role ──
    role = df_filtered.groupby("JobRole")["AttritionFlag"].mean() * 100
    role = role.sort_values(ascending=False)
    top3_roles = role.head(3)
    insight_box(
        f"Top 3 job roles with highest observed attrition: "
        + ", ".join([f"{r} ({v:.1f}%)" for r, v in top3_roles.items()])
    )

    # ── Satisfaction ──
    for sat_col, sat_name in [
        ("JobSatisfaction", "job satisfaction"),
        ("EnvironmentSatisfaction", "environment satisfaction"),
        ("WorkLifeBalance", "work-life balance"),
    ]:
        if sat_col in df_filtered.columns:
            sat = df_filtered.groupby(sat_col)["AttritionFlag"].mean() * 100
            if len(sat) >= 2:
                lowest_sat = sat.idxmin()
                highest_sat_rate = sat.max()
                lowest_rate = sat.min()
                insight_box(
                    f"For {sat_name}: employees with the lowest rating (1) show "
                    f"{sat.get(1, 0):.1f}% attrition rate vs {sat.min():.1f}% "
                    f"for the highest rating. Lower {sat_name} is associated with higher attrition "
                    f"in this dataset."
                )

    # ── Income ──
    left_income   = df_filtered[df_filtered["Attrition"] == "Yes"]["MonthlyIncome"].mean()
    stayed_income = df_filtered[df_filtered["Attrition"] == "No"]["MonthlyIncome"].mean()
    insight_box(
        f"Average monthly income of employees who left: ${left_income:,.0f} | "
        f"Employees who stayed: ${stayed_income:,.0f}. "
        f"Employees who left earned ${abs(stayed_income - left_income):,.0f} "
        f"{'less' if left_income < stayed_income else 'more'} on average "
        f"(observed association, not a controlled experiment)."
    )

    # ── Tenure ──
    left_tenure   = df_filtered[df_filtered["Attrition"] == "Yes"]["YearsAtCompany"].mean()
    stayed_tenure = df_filtered[df_filtered["Attrition"] == "No"]["YearsAtCompany"].mean()
    insight_box(
        f"Average tenure of employees who left: {left_tenure:.1f} years | "
        f"Employees who stayed: {stayed_tenure:.1f} years. "
        f"Shorter tenure is associated with higher attrition in this dataset."
    )

    # ── Distance ──
    left_dist   = df_filtered[df_filtered["Attrition"] == "Yes"]["DistanceFromHome"].mean()
    stayed_dist = df_filtered[df_filtered["Attrition"] == "No"]["DistanceFromHome"].mean()
    insight_box(
        f"Average distance from home: employees who left ({left_dist:.1f} km) vs "
        f"employees who stayed ({stayed_dist:.1f} km). "
        f"Longer commutes show a slight observed association with attrition in this dataset."
    )

    # ── Stock Option ──
    so = df_filtered.groupby("StockOptionLevel")["AttritionFlag"].mean() * 100
    insight_box(
        "Stock option level vs attrition rate: "
        + " | ".join([f"Level {k}: {v:.1f}%" for k, v in so.items()])
        + ". Employees with no stock options show higher attrition rates in this dataset."
    )

    section_header("Summary Statistics — Filtered Dataset")
    col1, col2 = st.columns(2)
    with col1:
        total = len(df_filtered)
        left = df_filtered["AttritionFlag"].sum()
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total Employees | {total:,} |
        | Attrition Count | {left:,} |
        | Attrition Rate | {left/total*100:.1f}% |
        | Avg Monthly Income | ${df_filtered['MonthlyIncome'].mean():,.0f} |
        | Avg Age | {df_filtered['Age'].mean():.1f} years |
        | Avg Tenure | {df_filtered['YearsAtCompany'].mean():.1f} years |
        | % Working OverTime | {(df_filtered['OverTime'] == 'Yes').mean()*100:.1f}% |
        """)
    with col2:
        dept_counts = df_filtered["Department"].value_counts()
        role_counts = df_filtered["JobRole"].value_counts()
        st.markdown(f"""
        | Department | Employees |
        |------------|-----------|
        """ + "\n".join([f"| {d} | {n:,} |" for d, n in dept_counts.items()])
        )


# ═══════════════════════════════════════════════════════════════
# PAGE 9 — MODEL EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
def page_explainability(trained_models, results, feature_cols):
    st.markdown("## 🔍 Model Explainability")
    st.caption(
        "Feature importance from tree-based models indicates how much each feature "
        "contributed to the model's decisions — not causal influence."
    )

    st.markdown("""
    <div class="insight-box">
    <strong>About Feature Importance:</strong> Feature importance values reflect how much each variable
    was used by the model to split data and reduce impurity. High importance means the feature was
    predictive in the training data. It does <em>not</em> mean the feature causes attrition.
    Correlation in historical data may reflect confounding factors or selection biases.
    </div>
    """, unsafe_allow_html=True)

    section_header("Random Forest Feature Importance")
    rf_model = trained_models.get("Random Forest")
    if rf_model:
        rf_clf = rf_model.named_steps["clf"]
        importances = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": rf_clf.feature_importances_
        }).sort_values("Importance", ascending=False).head(20)

        fig = px.bar(importances, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues_r",
                     title="Top 20 Feature Importances — Random Forest")
        fig.update_layout(coloraxis_showscale=False, height=520, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Top 5 insights
        section_header("Top Feature Insights")
        for _, row in importances.head(5).iterrows():
            insight_box(
                f"**{row['Feature']}** — Importance score: {row['Importance']:.4f}. "
                f"This feature ranked highly in the Random Forest model's decision-making process."
            )

    section_header("Gradient Boosting Feature Importance")
    gb_model = trained_models.get("Gradient Boosting")
    if gb_model:
        gb_clf = gb_model.named_steps["clf"]
        imp_gb = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": gb_clf.feature_importances_
        }).sort_values("Importance", ascending=False).head(20)

        fig2 = px.bar(imp_gb, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Purples_r",
                      title="Top 20 Feature Importances — Gradient Boosting")
        fig2.update_layout(coloraxis_showscale=False, height=520, margin=dict(t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    section_header("Logistic Regression Coefficients")
    lr_model = trained_models.get("Logistic Regression")
    if lr_model:
        lr_clf = lr_model.named_steps["clf"]
        coef_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": lr_clf.coef_[0]
        }).sort_values("Coefficient", key=abs, ascending=False).head(20)

        fig3 = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                      color="Coefficient", color_continuous_scale="RdBu_r",
                      title="Top 20 Logistic Regression Coefficients (Standardized)")
        fig3.update_layout(coloraxis_showscale=False, height=520, margin=dict(t=40, b=10))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "Positive coefficients indicate features associated with higher attrition probability. "
            "Negative coefficients are associated with lower attrition probability. "
            "Values are on standardized scale."
        )


# ═══════════════════════════════════════════════════════════════
# PAGE 10 — ABOUT PROJECT
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════
def main():
    # ── Load & preprocess data ──
    df_raw = load_data()
    df_clean, df_encoded, le_dict, cat_cols = preprocess_data(df_raw)

    # Store encoders in session state for use in prediction page
    if "le_dict" not in st.session_state:
        st.session_state["le_dict"] = le_dict

    # ── Train ML models (cached) ──
    with st.spinner("Training ML models (cached after first run)..."):
        trained_models, results, X_train, X_test, y_train, y_test, feature_cols = \
            train_models(df_encoded)

    # ═══════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:10px 0 6px 0">
            <div style="font-size:2.2rem">👥</div>
            <div style="font-size:1.0rem;font-weight:700;color:#e94560;margin-top:4px">
                Workforce Intelligence
            </div>
            <div style="font-size:0.72rem;color:#a8b2d8;margin-top:2px">
                IBM HR Analytics Dashboard
            </div>
        </div>
        <hr style="border-color:#2a2a4a;margin:8px 0 12px 0">
        """, unsafe_allow_html=True)

        # ── Navigation ──
        st.markdown(
            '<p style="color:#a8b2d8;font-size:0.82rem;font-weight:600;'
            'text-transform:uppercase;letter-spacing:1px;margin:0 0 6px 0">'
            '🗺️ Navigation</p>',
            unsafe_allow_html=True
        )
        page = st.radio(
            "Go to",
            [
                "Executive Overview",
                "Workforce Analytics",
                "Attrition Analysis",
                "Correlation & Features",
                "ML Model Performance",
                "Employee Risk Prediction",
                "Workforce Segmentation",
                "Workforce Intelligence",
                "Model Explainability",
            ],
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border-color:#2a2a4a;margin:10px 0'>", unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#a8b2d8;font-size:0.82rem;font-weight:600;'
            'text-transform:uppercase;letter-spacing:1px;margin:0 0 4px 0">'
            '🔽 Dataset Filters</p>',
            unsafe_allow_html=True
        )
        st.caption("Filters apply to Overview, Workforce Analytics, Attrition Analysis, Correlation & Intelligence pages.")

        # ── Filters ──
        all_depts  = ["All"] + sorted(df_clean["Department"].unique().tolist())
        all_roles  = ["All"] + sorted(df_clean["JobRole"].unique().tolist())
        all_genders = ["All"] + sorted(df_clean["Gender"].unique().tolist())
        all_travels = ["All"] + sorted(df_clean["BusinessTravel"].unique().tolist())
        all_ot     = ["All"] + sorted(df_clean["OverTime"].unique().tolist())
        all_levels = ["All"] + sorted([str(x) for x in df_clean["JobLevel"].unique()])

        sel_dept    = st.selectbox("Department",    all_depts)
        sel_role    = st.selectbox("Job Role",      all_roles)
        sel_gender  = st.selectbox("Gender",        all_genders)
        sel_travel  = st.selectbox("Business Travel", all_travels)
        sel_ot      = st.selectbox("OverTime",      all_ot)
        sel_level   = st.selectbox("Job Level",     all_levels)

        age_range = st.slider("Age Range", int(df_clean["Age"].min()),
                               int(df_clean["Age"].max()),
                               (int(df_clean["Age"].min()), int(df_clean["Age"].max())))

        if st.button("🔄 Reset Filters", use_container_width=True):
            st.rerun()

        st.markdown("<hr style='border-color:#2a2a4a;margin:10px 0'>", unsafe_allow_html=True)
        total_n = len(df_clean)
        attr_n  = df_clean["AttritionFlag"].sum()
        st.markdown(f"""
        <div style="font-size:0.75rem;color:#a8b2d8">
        📊 Dataset: <b style="color:#e94560">{total_n:,}</b> employees<br>
        📉 Attrition: <b style="color:#e94560">{attr_n:,}</b> ({attr_n/total_n*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)

    # ── Apply Filters ──
    df_filtered = df_clean.copy()
    if sel_dept   != "All": df_filtered = df_filtered[df_filtered["Department"]    == sel_dept]
    if sel_role   != "All": df_filtered = df_filtered[df_filtered["JobRole"]       == sel_role]
    if sel_gender != "All": df_filtered = df_filtered[df_filtered["Gender"]        == sel_gender]
    if sel_travel != "All": df_filtered = df_filtered[df_filtered["BusinessTravel"] == sel_travel]
    if sel_ot     != "All": df_filtered = df_filtered[df_filtered["OverTime"]      == sel_ot]
    if sel_level  != "All": df_filtered = df_filtered[df_filtered["JobLevel"]      == int(sel_level)]
    df_filtered = df_filtered[
        (df_filtered["Age"] >= age_range[0]) & (df_filtered["Age"] <= age_range[1])
    ]

    # ── Route Pages ──
    if page == "Executive Overview":
        page_overview(df_clean, df_filtered)

    elif page == "Workforce Analytics":
        page_workforce(df_clean, df_filtered)

    elif page == "Attrition Analysis":
        page_attrition(df_clean, df_filtered)

    elif page == "Correlation & Features":
        page_correlation(df_clean, df_filtered)

    elif page == "ML Model Performance":
        page_ml_performance(trained_models, results, X_test, y_test)

    elif page == "Employee Risk Prediction":
        page_prediction(df_clean, trained_models, results, feature_cols)

    elif page == "Workforce Segmentation":
        page_segmentation(df_clean, df_encoded)

    elif page == "Workforce Intelligence":
        page_intelligence(df_clean, df_filtered)

    elif page == "Model Explainability":
        page_explainability(trained_models, results, feature_cols)


if __name__ == "__main__":
    main()
