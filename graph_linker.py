# Re-import necessary modules after code execution state reset
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Reload graph-of-graphs data
with open("bulk_output/contented_thread_graphs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

graphs = data["graphs"]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Extract text representations
graph_representations = []

for idx, graph in enumerate(graphs):
    theme_texts = [node["label"] for node in graph["nodes"] if node["type"] == "Theme"]
    solution_texts = [node["label"] for node in graph["nodes"] if node["type"] == "Solution"]
    combined_text = " ".join(theme_texts + solution_texts) or graph["title"]

    graph_representations.append({
        "index": idx,
        "title": graph["title"],
        "text": combined_text
    })

# Step 2: Embed each graph's text
texts = [g["text"] for g in graph_representations]
embeddings = model.encode(texts, convert_to_tensor=True)

# Step 3: Compute similarity matrix
similarity_matrix = util.cos_sim(embeddings, embeddings)

# Step 4: Link graphs based on similarity
THRESHOLD = 0.75
for i in range(len(graphs)):
    for j in range(i + 1, len(graphs)):
        sim_score = similarity_matrix[i][j].item()
        if sim_score >= THRESHOLD:
            source_id = f"thread_{i}"
            target_id = f"thread_{j}"
            edge = {"source": source_id, "target": target_id, "label": "RELATED_TO"}
            graphs[i].setdefault("cross_edges", []).append(edge)
            graphs[j].setdefault("cross_edges", []).append(edge)

# Save updated graph
os.makedirs("output", exist_ok=True)
with open("bulk_output/thread_graphs_linked_75.json", "w", encoding="utf-8") as f:
    json.dump({"graphs": graphs}, f, indent=2)

# Display result for user inspection
# import pandas as pd
# # import ace_tools as tools
# tools.display_dataframe_to_user(name="Linked Graph Similarities", dataframe=pd.DataFrame([
#     {"Thread Index": g["index"], "Title": g["title"]} for g in graph_representations
# ]))
