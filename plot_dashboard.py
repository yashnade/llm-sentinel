# plot_dashboard.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "evaluation/eval_results.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM evaluations", conn)
conn.close()
if df.empty:
    print("No evaluation data.")
    exit(0)

df["created_at"] = pd.to_datetime(df["created_at"], unit="s")

# Avg per model
agg = df.groupby("model_name").agg({"faithfulness":"mean","relevance":"mean"}).reset_index()
fig, ax = plt.subplots(figsize=(8,4))
x = range(len(agg))
ax.bar(x, agg["faithfulness"], label="Faithfulness")
ax.bar(x, agg["relevance"], bottom=agg["faithfulness"], label="Relevance")
ax.set_xticks(x)
ax.set_xticklabels(agg["model_name"], rotation=45, ha='right')
ax.set_ylabel("Avg score")
ax.legend()
plt.tight_layout()
plt.savefig("avg_scores_by_model.png")
print("Saved avg_scores_by_model.png")

# Trend for the first model
model0 = agg["model_name"].iloc[0]
trend_df = df[df["model_name"] == model0].set_index("created_at").resample("D").mean().dropna()
if not trend_df.empty:
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(trend_df.index, trend_df["faithfulness"], label="Faithfulness")
    ax2.plot(trend_df.index, trend_df["relevance"], label="Relevance")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("score_trend.png")
    print("Saved score_trend.png for model:", model0)
else:
    print("Not enough data to produce trend plot for model:", model0)
