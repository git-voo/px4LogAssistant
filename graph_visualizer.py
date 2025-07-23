import json
import networkx as nx
import matplotlib.pyplot as plt
import os

# Load extracted thread graphs
with open("output/thread_graphs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Choose which thread to visualize (0 = first one)
thread_index = 4
thread = data["graphs"][thread_index]
nodes = thread["nodes"]
edges = thread["edges"]

g = nx.DiGraph()

# Add nodes with labels
for node in nodes:
    g.add_node(node["id"], label=node["label"], type=node["type"])

# Add edges with relationship labels
for edge in edges:
    g.add_edge(edge["source"], edge["target"], label=edge["label"])

# Draw graph
pos = nx.spring_layout(g, seed=42)
labels = nx.get_node_attributes(g, "label")
types = nx.get_node_attributes(g, "type")
edge_labels = nx.get_edge_attributes(g, "label")

plt.figure(figsize=(12, 8))
nx.draw(g, pos, labels=labels, with_labels=True, node_size=3000,
        node_color="lightblue", font_size=8, font_weight="bold")
nx.draw_networkx_edge_labels(
    g, pos, edge_labels=edge_labels, font_color="red", font_size=7)
plt.title(f"Thread Graph: {thread['title'][:80]}..." if len(
    thread['title']) > 80 else f"Thread Graph: {thread['title']}")
plt.axis("off")
plt.tight_layout()
os.makedirs("viz", exist_ok=True)
plt.savefig("viz/thread_graph_visualization.png")
plt.show()

print("✅ Graph visualization saved to viz/thread_graph_visualization.png")
