import json
import networkx as nx
import matplotlib.pyplot as plt

# Load the linked graph data
with open("output/all_threads_graphs_linked.json", "r", encoding="utf-8") as f:
    data = json.load(f)

graphs = data["graphs"]

# Create a new NetworkX graph
G = nx.Graph()

# Add nodes and inter-graph edges
for idx, graph in enumerate(graphs):
    thread_id = f"thread_{idx}"
    G.add_node(thread_id, label=graph["title"])
    
    for edge in graph.get("cross_edges", []):
        src = edge["source"]
        tgt = edge["target"]
        if not G.has_edge(src, tgt):  # Avoid duplicates
            G.add_edge(src, tgt, label=edge["label"])

# Draw the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, font_size=8, font_weight="bold")
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
plt.title("Graph of Threads Linked by Similar Themes or Solutions")
plt.axis("off")
plt.show()
plt.savefig("viz/linked_thread_graph_visualization.png")