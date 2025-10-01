import json
import torch
from sentence_transformers import util
from .embedder import Embedder

class GraphRetriever:
    def __init__(self, graph_path="output/all_threads_graphs_linked.json"):
        self.embedder = Embedder()
        self.graphs = self._load_graphs(graph_path)
        self.representations = self._build_graph_texts()

    def _load_graphs(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["graphs"]

    def _build_graph_texts(self):
        reps = []
        for i, graph in enumerate(self.graphs):
            node_texts = [n["label"] for n in graph.get("nodes", [])]
            edge_types = [e["label"] for e in graph.get("edges", [])]

            # Optional: Include content if present
            content_texts = [
                n.get("content", "") for n in graph.get("nodes", [])
                if n.get("content")
            ]

            combined = node_texts + edge_types + content_texts
            fallback = graph.get("title", f"Graph {i}")
            text = " ".join(combined) or fallback

            reps.append({"index": i, "text": text})

        return reps

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedder.embed([query])[0]
        graph_embeddings = self.embedder.embed([r["text"] for r in self.representations])
        scores = util.cos_sim(query_embedding, graph_embeddings)[0]
        top_indices = torch.topk(scores, k=top_k).indices.tolist()
        return [self.graphs[i] for i in top_indices]
