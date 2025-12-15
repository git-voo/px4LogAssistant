"""
Visualization Script for Evaluation Metrics
-------------------------------------------
Generates:
 - Box plot (Faithfulness)
 - Histogram (Contextual Precision)
 - Distribution (Contextual Recall)
 - Scatter (Answer Relevancy vs Faithfulness)
 - Radar chart (Mean scores across metrics)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Load results ===
EVAL_PATH = "rag_eval/data/eval_results.json"
SAVE_DIR = "rag_eval/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

details = data["details"]

# === Collect metric scores ===
metrics_data = {}
for test in details:
    for m in test["metrics"]:
        name = m["name"]
        score = m.get("score")
        if score is not None:
            metrics_data.setdefault(name, []).append(score)

# --- Convert to DataFrame for easier handling ---
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in metrics_data.items()]))
print("Available metrics:", list(df.columns))

# --------------------------------------------------
# 1️⃣ Box Plot — Faithfulness (low variance expected)
# --------------------------------------------------
if "Faithfulness" in df.columns:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df["Faithfulness"].dropna(), vert=True, patch_artist=True)
    plt.title("Box Plot of Faithfulness Scores")
    plt.ylabel("Faithfulness Score")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{SAVE_DIR}/fig4_2_box_faithfulness.png", dpi=300, bbox_inches="tight")
    plt.close()

# --------------------------------------------------
# 2️⃣ Histogram — Contextual Precision
# --------------------------------------------------
if "Contextual Precision" in df.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df["Contextual Precision"].dropna(), bins=10, color="skyblue", edgecolor="black")
    plt.title("Histogram of Contextual Precision Scores")
    plt.xlabel("Precision Score")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{SAVE_DIR}/fig4_3_hist_precision.png", dpi=300, bbox_inches="tight")
    plt.close()

# --------------------------------------------------
# 3️⃣ Distribution — Contextual Recall
# --------------------------------------------------
if "Contextual Recall" in df.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df["Contextual Recall"].dropna(), bins=10, color="lightcoral", edgecolor="black")
    plt.title("Distribution of Contextual Recall Scores")
    plt.xlabel("Recall Score")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"{SAVE_DIR}/fig4_4_dist_recall.png", dpi=300, bbox_inches="tight")
    plt.close()

# --------------------------------------------------
# 4️⃣ Scatter — Relevancy vs Faithfulness
# --------------------------------------------------
if {"Answer Relevancy", "Faithfulness"}.issubset(df.columns):
    plt.figure(figsize=(6, 4))
    plt.scatter(df["Faithfulness"], df["Answer Relevancy"], alpha=0.7, color="teal")
    plt.title("Scatter Plot of Relevancy vs Faithfulness")
    plt.xlabel("Faithfulness Score")
    plt.ylabel("Answer Relevancy Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"{SAVE_DIR}/fig4_5_scatter_relevancy_faithfulness.png", dpi=300, bbox_inches="tight")
    plt.close()

# --------------------------------------------------
# 5️⃣ Radar Chart — Mean Scores Across Metrics
# --------------------------------------------------
mean_scores = df.mean().dropna()
metrics = mean_scores.index.tolist()
scores = mean_scores.values

if len(metrics) > 2:
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    scores = np.concatenate((scores, [scores[0]]))
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, color="darkorange", linewidth=2)
    ax.fill(angles, scores, color="orange", alpha=0.25)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title("Radar Chart of Mean Scores Across Metrics", pad=20)
    plt.savefig(f"{SAVE_DIR}/fig4_6_radar_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

print(f"✅ All plots saved in: {SAVE_DIR}")
