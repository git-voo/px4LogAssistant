import json
import networkx as nx
from pathlib import Path

# === CONFIG ===
INPUT_FILE = "output/merged_threads.json"
OUTPUT_JSON = "output/knowledge_graph.json"
OUTPUT_GRAPHML = "output/knowledge_graph.graphml"

# === STRUCTURES ===
nodes = {}
edges = []
G = nx.DiGraph()


def add_node(node_id, node_type, label):
    if node_id not in nodes:
        nodes[node_id] = {"id": node_id, "type": node_type, "label": label}
        G.add_node(node_id, type=node_type, label=label)


def add_edge(src, dst, rel_type):
    edges.append({"from": src, "to": dst, "type": rel_type})
    G.add_edge(src, dst, type=rel_type)


def trim(text, limit=80):
    return text.strip().replace("\n", " ")[:limit] + "..." if len(text) > limit else text.strip()

# === MAIN ===


def build_knowledge_graph():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        threads = json.load(f)

    for thread in threads:
        thread_id = f"T_{thread['thread_id']}"
        add_node(thread_id, "Thread", thread["title"])

        for tag in thread.get("tags", []):
            tag_id = f"Tag_{tag}"
            add_node(tag_id, "Tag", tag)
            add_edge(thread_id, tag_id, "HAS_TAG")

        # Firmware version (naive match)
        firmware_version = None
        for post in thread["posts"]:
            if "PX4 version" in post["raw"]:
                for line in post["raw"].split("\n"):
                    if "PX4 version" in line:
                        firmware_version = line.strip()
                        break
            if firmware_version:
                break
        if firmware_version:
            fw_id = f"FW_{firmware_version}"
            add_node(fw_id, "FirmwareVersion", firmware_version)
            add_edge(thread_id, fw_id, "RUNS_VERSION")

        # Posts and relationships
        for post in thread["posts"]:
            post_id = f"P_{post['post_id']}"
            label = trim(post["raw"])
            add_node(post_id, "Post", label)
            add_edge(thread_id, post_id, "HAS_POST")

            if post.get("is_solution"):
                sol_id = f"SOL_{post['post_id']}"
                add_node(sol_id, "Solution", f"Solution to: {label[:60]}")
                add_edge(post_id, sol_id, "IS_SOLUTION")

            for line in post["raw"].split("\n"):
                if line.strip().startswith(("make ", "./", "sudo ", "pip", "git")):
                    cmd = line.strip()
                    cmd_id = f"CMD_{hash(cmd)}"
                    add_node(cmd_id, "ActionCommand", cmd)
                    add_edge(post_id, cmd_id, "INVOKES_ACTION")

            if "ModuleNotFoundError" in post["raw"]:
                err_id = "ERR_ModuleNotFoundError"
                add_node(err_id, "ErrorCode", "ModuleNotFoundError")
                add_edge(post_id, err_id, "MENTIONS_ERROR_CODE")
            if "No such file or directory" in post["raw"]:
                err_id = "ERR_FileNotFoundError"
                add_node(err_id, "ErrorCode", "FileNotFoundError")
                add_edge(post_id, err_id, "MENTIONS_ERROR_CODE")

            if "WSL" in post["raw"] or "Ubuntu" in post["raw"]:
                env_id = "ENV_WSL_Ubuntu"
                add_node(env_id, "Environment", "WSL + Ubuntu")
                add_edge(post_id, env_id, "DESCRIBES_ENVIRONMENT")

            for word in ["kconfiglib", "future", "menuconfig", "ELRS", "Cubepilot"]:
                if word in post["raw"]:
                    if word.lower() in ["elrs", "cubepilot"]:
                        mod_id = f"HW_{word}"
                        add_node(mod_id, "HardwareModule", word)
                        add_edge(post_id, mod_id, "MENTIONS_MODULE")
                    else:
                        mod_id = f"SW_{word}"
                        add_node(mod_id, "SoftwareModule", word)
                        add_edge(post_id, mod_id, "MENTIONS_MODULE")

            for token in post["raw"].split():
                if token.isupper() and "_" in token:
                    param_id = f"PARAM_{token}"
                    add_node(param_id, "Parameter", token)
                    add_edge(post_id, param_id, "MENTIONS_PARAM")

        # Theme from title
        theme_label = thread["title"]
        theme_id = f"Theme_{hash(theme_label)}"
        add_node(theme_id, "Theme", theme_label)
        add_edge(thread_id, theme_id, "FOCUSES_ON")
        for post in thread["posts"]:
            post_id = f"P_{post['post_id']}"
            add_edge(theme_id, post_id, "DESCRIBED_BY")
            if post.get("is_solution"):
                add_edge(theme_id, post_id, "ADDRESSED_BY")

    # Output directory
    Path("output").mkdir(exist_ok=True)

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"nodes": list(nodes.values()), "edges": edges}, f, indent=2)
    print(f"✅ Saved knowledge graph to {OUTPUT_JSON}")

    # Save GraphML
    nx.write_graphml(G, OUTPUT_GRAPHML)
    print(f"✅ Saved GraphML for Neo4j/Gephi to {OUTPUT_GRAPHML}")


# Run the builder
if __name__ == "__main__":
    build_knowledge_graph()
