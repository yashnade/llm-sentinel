# dashboard/app.py
import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np

DB_PATH = "evaluation/eval_results.db"

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f2937;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #374151;
        font-weight: 600;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Info boxes */
    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9fafb;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM evaluations", conn)
    conn.close()
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], unit="s", errors="coerce")
    return df


# ==============================================================
# ⚙️ Page Configuration
# ==============================================================
st.set_page_config(
    layout="wide",
    page_title="Model Evaluation Dashboard",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 Model Evaluation Dashboard")
    st.markdown("Monitor and analyze your model performance metrics in real-time")

# Load data
df = load_data()
if df.empty:
    st.error("⚠️ No evaluation data found. Please run evaluations first.")
    st.stop()

# ==============================================================
# 🎛️ Sidebar Filters
# ==============================================================
st.sidebar.header("🎛️ Filters")
st.sidebar.markdown("---")

models = sorted(df["model_name"].unique())
model_sel = st.sidebar.multiselect(
    "📦 Select Models",
    models,
    default=models,
    help="Choose one or more models to analyze"
)

sample_ids = sorted(df["sample_id"].unique())
sample_sel = st.sidebar.multiselect(
    "🔖 Select Sample IDs",
    sample_ids,
    default=sample_ids,
    help="Filter by specific sample IDs"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use filters to focus on specific models or samples")

# Filter data
filtered = df[df["model_name"].isin(model_sel) & df["sample_id"].isin(sample_sel)]

# ==============================================================
# 📈 Key Metrics Summary
# ==============================================================
if not filtered.empty:
    st.markdown("### 📈 Key Metrics Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_evals = len(filtered)
        st.metric(
            label="Total Evaluations",
            value=f"{total_evals:,}",
            delta=None
        )
    
    with col2:
        avg_faithfulness = filtered["faithfulness"].mean()
        st.metric(
            label="Avg Faithfulness",
            value=f"{avg_faithfulness:.2f}",
            delta=None
        )
    
    with col3:
        avg_relevance = filtered["relevance"].mean()
        st.metric(
            label="Avg Relevance",
            value=f"{avg_relevance:.2f}",
            delta=None
        )
    
    with col4:
        avg_latency = filtered["latency"].mean()
        st.metric(
            label="Avg Latency",
            value=f"{avg_latency:.2f}s",
            delta=None
        )
    
    st.markdown("---")

# ==============================================================
# 📊 Average Scores by Model
# ==============================================================
st.markdown("### 📊 Average Scores by Model")

if not filtered.empty:
    numeric_cols = ["faithfulness", "relevance", "latency"]
    agg = (
        filtered.groupby("model_name")[numeric_cols]
        .mean()
        .reset_index()
    )

    if not agg.empty:
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f9fafb')
        
        width = 0.35
        x = np.arange(len(agg))
        
        bars1 = ax.bar(x - width/2, agg["faithfulness"], width=width, 
                       label="Faithfulness", color="#667eea", alpha=0.8, edgecolor='white', linewidth=1.5)
        bars2 = ax.bar(x + width/2, agg["relevance"], width=width, 
                       label="Relevance", color="#764ba2", alpha=0.8, edgecolor='white', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(agg["model_name"], rotation=45, ha="right", fontsize=10, fontweight='500')
        ax.set_ylabel("Average Score", fontsize=11, fontweight='600', color='#374151')
        ax.set_ylim(0, 5.5)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("ℹ️ No aggregated data available for selected filters.")
else:
    st.info("ℹ️ No data matches the selected filters.")

st.markdown("---")

# ==============================================================
# 📈 Model Score Trends
# ==============================================================
st.markdown("### 📈 Model Score Trends Over Time")

col1, col2 = st.columns([2, 1])
with col1:
    model_for_trend = st.selectbox("Select model for trend analysis", models, key="trend_model")

trend_df = df[df["model_name"] == model_for_trend].sort_values("created_at")

if not trend_df.empty and len(trend_df) > 1:
    trend_df["created_at"] = pd.to_datetime(trend_df["created_at"], errors="coerce")

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    fig2.patch.set_facecolor('white')
    ax2.set_facecolor('#f9fafb')
    
    ax2.plot(trend_df["created_at"], trend_df["faithfulness"], "-o", 
             label="Faithfulness", color="#667eea", linewidth=2.5, 
             markersize=7, markerfacecolor="#667eea", markeredgecolor="white", markeredgewidth=2)
    ax2.plot(trend_df["created_at"], trend_df["relevance"], "-o", 
             label="Relevance", color="#764ba2", linewidth=2.5, 
             markersize=7, markerfacecolor="#764ba2", markeredgecolor="white", markeredgewidth=2)
    
    ax2.set_xlabel("Date & Time", fontsize=11, fontweight='600', color='#374151')
    ax2.set_ylabel("Score (1–5)", fontsize=11, fontweight='600', color='#374151')
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.xaxis.set_major_formatter(DateFormatter("%b %d %H:%M"))
    
    plt.tight_layout()
    fig2.autofmt_xdate()
    st.pyplot(fig2)
else:
    st.info("ℹ️ Not enough data points for trend visualization. Need at least 2 data points.")

st.markdown("---")

# ==============================================================
# ⚡ Latency Trend
# ==============================================================
st.markdown("### ⚡ Latency Trend Analysis")

latency_df = df[df["model_name"] == model_for_trend].sort_values("created_at")
if not latency_df.empty and len(latency_df) > 1:
    latency_df["created_at"] = pd.to_datetime(latency_df["created_at"], errors="coerce")

    fig3, ax3 = plt.subplots(figsize=(14, 5))
    fig3.patch.set_facecolor('white')
    ax3.set_facecolor('#f9fafb')
    
    ax3.plot(
        latency_df["created_at"],
        latency_df["latency"],
        color="#ef4444",
        linewidth=3,
        marker="o",
        markersize=8,
        markerfacecolor="#ef4444",
        markeredgecolor="#ffffff",
        markeredgewidth=2,
        label="Response Time"
    )
    
    # Add average line
    avg_latency = latency_df["latency"].mean()
    ax3.axhline(y=avg_latency, color='#f59e0b', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Average: {avg_latency:.2f}s')

    ax3.set_xlabel("Date & Time", fontsize=11, fontweight='600', color='#374151')
    ax3.set_ylabel("Latency (seconds)", fontsize=11, fontweight='600', color='#374151')
    ax3.set_title(f"Response Time for {model_for_trend}", fontsize=13, 
                  fontweight='bold', color='#1f2937', pad=20)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.xaxis.set_major_formatter(DateFormatter("%b %d %H:%M"))
    
    plt.tight_layout()
    fig3.autofmt_xdate()
    st.pyplot(fig3)
else:
    st.info("ℹ️ Not enough latency data available for visualization.")

st.markdown("---")

# ==============================================================
# 📋 Detailed Evaluation Data
# ==============================================================
st.markdown("### 📋 Detailed Evaluation Data")

# Add search/filter options
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_term = st.text_input("🔍 Search in data", "", placeholder="Search by model name, sample ID...")
with col2:
    sort_column = st.selectbox("Sort by", ["created_at", "faithfulness", "relevance", "latency"])
with col3:
    sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

# Apply search filter
if search_term:
    display_df = filtered[
        filtered["model_name"].str.contains(search_term, case=False) |
        filtered["sample_id"].astype(str).str.contains(search_term, case=False)
    ]
else:
    display_df = filtered

# Apply sorting
ascending = sort_order == "Ascending"
display_df = display_df.sort_values(sort_column, ascending=ascending).reset_index(drop=True)

# Display dataframe with custom styling
st.dataframe(
    display_df,
    use_container_width=True,
    height=400
)

st.caption(f"Showing {len(display_df)} of {len(filtered)} evaluations")

# ==============================================================
# Footer
# ==============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p style='margin: 0;'>Built with Streamlit • Last updated: {}</p>
</div>
""".format(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)