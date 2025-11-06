# dashboard/app.py
import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

DB_PATH = "evaluation/eval_results.db"

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM evaluations", conn)
    conn.close()
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], unit="s", errors="coerce")
    return df


# ==============================================================
# ⚙️ Streamlit Layout and Data
# ==============================================================
st.set_page_config(layout="wide")
st.title("📊 Model Evaluation Dashboard")

df = load_data()
if df.empty:
    st.warning("No evaluation data found. Run evaluations first.")
    st.stop()

models = sorted(df["model_name"].unique())
model_sel = st.sidebar.multiselect("Models (choose one or more)", models, default=models)

sample_ids = sorted(df["sample_id"].unique())
sample_sel = st.sidebar.multiselect("Sample IDs", sample_ids, default=sample_ids)

filtered = df[df["model_name"].isin(model_sel) & df["sample_id"].isin(sample_sel)]

st.markdown(f"**Total evaluations shown:** {len(filtered)}")


# ==============================================================
# 📊 Average Scores by Model
# ==============================================================
st.header("Average Scores by Model")

if not filtered.empty:
    numeric_cols = ["faithfulness", "relevance", "latency"]
    agg = (
        filtered.groupby("model_name")[numeric_cols]
        .mean()
        .reset_index()
    )

    if not agg.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.35
        x = range(len(agg))
        ax.bar(x, agg["faithfulness"], width=width, label="Faithfulness")
        ax.bar([i + width for i in x], agg["relevance"], width=width, label="Relevance")
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(agg["model_name"], rotation=45, ha="right")
        ax.set_ylabel("Average score")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No aggregated data for selected filters.")
else:
    st.info("No data for selected filters.")


# ==============================================================
# 📈 Faithfulness & Relevance Trend (No resampling)
# ==============================================================
st.header("Model Score Trends (Faithfulness & Relevance)")

model_for_trend = st.selectbox("Select model for trend", models)
trend_df = df[df["model_name"] == model_for_trend].sort_values("created_at")

if not trend_df.empty:
    trend_df["created_at"] = pd.to_datetime(trend_df["created_at"], errors="coerce")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(trend_df["created_at"], trend_df["faithfulness"], "-o", label="Faithfulness", color="#007bff", linewidth=2)
    ax2.plot(trend_df["created_at"], trend_df["relevance"], "-o", label="Relevance", color="#ff7f0e", linewidth=2)
    ax2.set_xlabel("Date & Time")
    ax2.set_ylabel("Score (1–5)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.xaxis.set_major_formatter(DateFormatter("%b %d %H:%M"))
    fig2.autofmt_xdate()
    st.pyplot(fig2)
else:
    st.info("Not enough data for trend visualization.")


# ==============================================================
# ⚡ Latency Trend (Styled like your shared red chart)
# ==============================================================
st.header("Latency Trend (Response Time Over Time)")

latency_df = df[df["model_name"] == model_for_trend].sort_values("created_at")
if not latency_df.empty:
    latency_df["created_at"] = pd.to_datetime(latency_df["created_at"], errors="coerce")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(
        latency_df["created_at"],
        latency_df["latency"],
        color="#ff3b3b",
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor="#ff3b3b",
        markeredgecolor="#ffffff",
        markeredgewidth=1.2,
    )

    ax3.set_xlabel("Date & Time")
    ax3.set_ylabel("Latency (seconds)")
    ax3.set_title(f"Latency Trend for {model_for_trend}", fontsize=13, weight="bold")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.xaxis.set_major_formatter(DateFormatter("%b %d %H:%M"))
    fig3.autofmt_xdate()

    st.pyplot(fig3)
else:
    st.info("Not enough latency data available.")


# ==============================================================
# 📋 Raw Evaluation Rows
# ==============================================================
st.header("Raw Evaluation Rows")
st.dataframe(filtered.sort_values("created_at", ascending=False).reset_index(drop=True))
