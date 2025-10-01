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
            theme_texts = [n["label"] for n in graph["nodes"] if n["type"] == "Theme"]
            solution_texts = [n["label"] for n in graph["nodes"] if n["type"] == "Solution"]
            fallback = graph["title"]
            text = " ".join(theme_texts + solution_texts) or fallback
            reps.append({"index": i, "text": text})
        return reps

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedder.embed([query])[0]
        graph_embeddings = self.embedder.embed([r["text"] for r in self.representations])
        scores = util.cos_sim(query_embedding, graph_embeddings)[0]
        top_indices = torch.topk(scores, k=top_k).indices.tolist()
        return [self.graphs[i] for i in top_indices]
