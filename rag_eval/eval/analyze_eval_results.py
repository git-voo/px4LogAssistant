import json
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

EVAL_PATH = "rag_eval/data/gemini_eval/eval_results.json"
OUT_DIR = "rag_eval/data"

# === Load data ===
if not os.path.exists(EVAL_PATH):
    raise FileNotFoundError(f"{EVAL_PATH} not found.")

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

details = data.get("details", [])
if not details:
    raise ValueError("No evaluation details found in file.")

# === Aggregate metrics ===
metrics_summary = {}

for test in details:
    for metric in test["metrics"]:
        name = metric["name"]
        score = metric.get("score")
        passed = metric.get("passed")

        if name not in metrics_summary:
            metrics_summary[name] = {"scores": [], "passes": 0, "total": 0}

        if score is not None:
            metrics_summary[name]["scores"].append(score)
        if passed:
            metrics_summary[name]["passes"] += 1
        metrics_summary[name]["total"] += 1

# === Prepare data for plotting ===
metric_names = list(metrics_summary.keys())
avg_scores = [
    np.mean(metrics_summary[m]["scores"]) if metrics_summary[m]["scores"] else 0
    for m in metric_names
]
std_devs = [
    np.std(metrics_summary[m]["scores"]) if metrics_summary[m]["scores"] else 0
    for m in metric_names
]
pass_rates = [
    metrics_summary[m]["passes"] / metrics_summary[m]["total"] * 100
    for m in metric_names
]

# === Create output directory ===
os.makedirs(OUT_DIR, exist_ok=True)

# === Save CSV summary ===
csv_path = os.path.join(OUT_DIR, "metrics_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Metric", "Average Score", "Standard Deviation", "Pass Rate (%)", "Total Tests"])
    for i, m in enumerate(metric_names):
        writer.writerow([
            m,
            round(avg_scores[i], 3),
            round(std_devs[i], 3),
            round(pass_rates[i], 2),
            metrics_summary[m]["total"]
        ])

print(f"✅ Summary CSV saved to: {csv_path}")

# === Plot 1: Average score per metric ===
plt.figure(figsize=(8, 5))
plt.bar(metric_names, avg_scores, color="skyblue")
plt.title("Average Score per Metric")
plt.ylabel("Average Score")
plt.ylim(0, 1)
for i, v in enumerate(avg_scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "avg_scores.png"), dpi=300)
plt.show()

# === Plot 2: Pass rate per metric ===
plt.figure(figsize=(8, 5))
plt.bar(metric_names, pass_rates, color="lightgreen")
plt.title("Pass Rate per Metric (%)")
plt.ylabel("Pass Rate (%)")
plt.ylim(0, 100)
for i, v in enumerate(pass_rates):
    plt.text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pass_rates.png"), dpi=300)
plt.show()

# === Plot 3: Score trend across test cases ===
plt.figure(figsize=(10, 5))
for metric_name in metric_names:
    scores = [
        np.mean([
            m["score"] for m in test["metrics"]
            if m["name"] == metric_name and m.get("score") is not None
        ]) for test in details
    ]
    plt.plot(scores, label=metric_name, linewidth=2)
plt.title("Score Trend per Metric Across Test Cases")
plt.xlabel("Test Case #")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "score_trends.png"), dpi=300)
plt.show()

print("\n📊 Charts saved:")
print("  • avg_scores.png")
print("  • pass_rates.png")
print("  • score_trends.png")
