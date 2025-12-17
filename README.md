## Quick Start (5 Minutes)

This section walks you through running ExpertLoop end-to-end with minimal setup.

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/UAVLab-SLU/flight_log_knowledge_graph.git
cd flight_log_knowledge_graph
```

### 2️⃣ Create and Activate a Virtual Environment

ExpertLoop uses a Python virtual environment to ensure dependency isolation.

macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell)
```bash
python3 -m venv venv
venv\Scripts\Activate
```

### 3️⃣ Install Dependencies

With the virtual environment activated:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Configure LLM Access

ExpertLoop relies on an external LLM for knowledge extraction and generation.

Set your API key as an environment variable:

macOS / Linux
```bash
export GEMINI_API_KEY="your_api_key_here"
```
Windows (PowerShell)
```bash
export GEMINI_API_KEY="your_api_key_here"
```
Or manually add API key to your .env file with the key ```GEMINI_API_KEY```

### 5️⃣ Prepare PX4 Forum Threads

Place PX4 forum thread data in the project root or `data/` directory.

The pipeline expects forum threads in JSON format. Example files already included:

```text
px4_threads.json
px4_posts.json
```

If you are adding your own data, ensure the structure follows the same schema used in ```px4_threads.json``` and ```px4_posts.json```.


### 6️⃣ Generate Knowledge Graphs from Forum Threads

Run the knowledge graph construction pipeline using the provided scripts.

Step 6.1 — Merge Threads and Posts
```bash
python merge_threads_and_posts.py
```
This produces merged thread representations used for knowledge extraction.

---

Step 6.2 — Run LLM-Assisted Knowledge Graph Extraction
```bash
python LLM-assisted-KG.py
```
This step:

- Extracts entities and relations using the LLM
- Builds thread-level knowledge graphs
- Outputs intermediate graph files

### Step 6.3 — Generate and Link Thread Graphs
```bash
python graph_linker.py
```
Outputs are written to:
```text
output/
├── thread_graphs.json
├── thread_graphs_linked.json
├── all_threads_graphs.json
├── all_threads_graphs_linked.json
├── merged_threads.json
├── knowledge_graph.graphml
```

These files represent:
- Individual thread subgraphs
- Linked graphs across related threads
- A merged global view (optional)

### 7️⃣ Run the RAG Engine (Query the System)

ExpertLoop’s RAG pipeline lives in the rag_engine/ directory.

Example Query Execution
```bash
python rag_engine/rag_pipeline.py \
  --query "Drone flips immediately on takeoff"
```
#### An example of the RAG pipeline invoccation is contained in ```/test_rag.py```

The pipeline will:

1. Retrieve relevant thread-level graphs
2. Build structured context using prompt_builder.py
3. Generate a grounded response via the LLM

Core RAG components:

- rag_engine/retriever.py
- rag_engine/prompt_builder.py
- rag_engine/embedded.py

### 8️⃣ (Optional) Run Evaluation

Evaluation scripts and datasets live in ```rag_eval/``` and ```eval/```.

To run RAG evaluation:
```bash
python eval/evaluate_rag.py
```
Evaluation artifacts (scores, plots, summaries) are saved in:
```bash
rag_eval/data/gemini_eval/
```

Including:
- metrics_summary.csv
- eval_results.json
- avg_scores.png
- pass_rates.png