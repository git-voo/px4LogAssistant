import json
import networkx as nx
import matplotlib.pyplot as plt
import os

# Load the linked graphs
with open("output/thread_graphs_linked.json", "r", encoding="utf-8") as f:
    data = json.load(f)

graphs = data["graphs"]

G = nx.DiGraph()
positions = {}
labels = {}

x_offset = 0  # used to space out each thread's subgraph horizontally

# Step 1: Add all individual thread subgraphs
for i, graph in enumerate(graphs):
    node_ids = []
    for node in graph.get("nodes", []):
        node_id = f"{i}_{node['id']}"  # make node ID globally unique
        G.add_node(node_id, label=node["label"], type=node["type"])
        labels[node_id] = node["label"]
        node_ids.append(node_id)

    if not node_ids:
        print(f"⚠️ Skipping Thread {i+1} ({graph['title']}) — no nodes found.")
        continue

    for edge in graph.get("edges", []):
        source = f"{i}_{edge['source']}"
        target = f"{i}_{edge['target']}"
        G.add_edge(source, target, label=edge["label"])

    # Layout for this subgraph
    subgraph = G.subgraph(node_ids)
    local_pos = nx.spring_layout(subgraph, seed=42)

    for node, pos in local_pos.items():
        positions[node] = (pos[0] + x_offset, pos[1])
    x_offset += 3.5

    # Add a virtual hub node for this thread to connect cross-thread links
    thread_hub = f"thread_{i}"
    G.add_node(thread_hub, label=f"Thread {i+1}", type="Thread")
    labels[thread_hub] = f"Thread {i+1}"
    positions[thread_hub] = positions[node_ids[0]]

# Step 2: Add cross-thread edges
for i, graph in enumerate(graphs):
    for edge in graph.get("cross_edges", []):
        source = edge["source"]
        target = edge["target"]
        if G.has_node(source) and G.has_node(target):
            G.add_edge(source, target, label="RELATED_TO")

# Step 3: Draw the graph
plt.figure(figsize=(24, 12))
nx.draw(
    G,
    pos=positions,
    labels=labels,
    with_labels=True,
    node_size=2500,
    node_color="lightblue",
    font_size=8,
    font_weight="bold"
)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_color="red", font_size=7)

plt.title("Linked Graph of Thread Subgraphs", fontsize=14)
plt.axis("off")
plt.tight_layout()

os.makedirs("viz", exist_ok=True)
plt.savefig("viz/linked_thread_graphs.png", dpi=300)
plt.show()

print("✅ Linked graph visualization saved to viz/linked_thread_graphs.png")
