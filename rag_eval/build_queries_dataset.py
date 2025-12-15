import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import json
import time
from pathlib import Path
from rag_engine.rag_pipeline import RAGPipeline

# --------------------------------------------------------------------
# 1️⃣ Load raw queries + references
# --------------------------------------------------------------------
input_path = Path("rag_eval/data/raw_queries_with_references.json")
output_path = Path("rag_eval/data/test_cases.json")

with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} query-reference pairs from {input_path}")

# --------------------------------------------------------------------
# 2️⃣ Initialize RAG Pipeline
# --------------------------------------------------------------------
rag = RAGPipeline()

def get_context_and_response(query: str):
    """
    Gets both retrieval context and model response from the RAG pipeline.
    Returns (context_text, response_text)
    """
    context, response = rag.answer(query)
    return context, response

# --------------------------------------------------------------------
# 3️⃣ Process each query through RAG
# --------------------------------------------------------------------
final_dataset = []

for i, item in enumerate(raw_data, start=1):
    query = item.get("query")
    reference = item.get("reference")

    print(f"\n[{i}/{len(raw_data)}] Processing query:")
    print(f"→ {query}")

    try:
        context, response = get_context_and_response(query)
        final_dataset.append({
            "query": query,
            "context": context,
            "response": response,
            "reference": reference
        })
    except Exception as e:
        print(f"⚠️ Error processing query #{i}: {e}")
        continue

    # Optional small delay (avoid API throttling if using remote LLM)
    time.sleep(2)

# --------------------------------------------------------------------
# 4️⃣ Save full DeepEval dataset
# --------------------------------------------------------------------
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, indent=2, ensure_ascii=False)

print(f"\n✅ Finished building DeepEval dataset with {len(final_dataset)} samples.")
print(f"Saved to {output_path}")











































# import json
# import re
# from pathlib import Path

# # --------------------------------------------------------------------
# # 1️⃣ Load all threads
# # --------------------------------------------------------------------
# input_path = Path("output/merged_threads.json")
# output_path = Path("rag_eval/data/queries.json")

# with open(input_path, "r", encoding="utf-8") as f:
#     threads = json.load(f)

# print(f"Total threads: {len(threads)}")

# # --------------------------------------------------------------------
# # 2️⃣ Skip the first 2500 (already used for graph)
# # --------------------------------------------------------------------
# test_candidates = threads[2500:]
# print(f"Remaining threads after skipping first 2500: {len(test_candidates)}")

# # --------------------------------------------------------------------
# # 3️⃣ Take the next 100 threads (or fewer if not enough)
# # --------------------------------------------------------------------
# selected_threads = test_candidates[:120]
# print(f"Selected {len(selected_threads)} threads for query extraction.")

# # --------------------------------------------------------------------
# # 4️⃣ Extract and clean first-post text
# # --------------------------------------------------------------------
# def clean_text(text: str) -> str:
#     """Simplify raw HTML or raw post text into a short, plain query."""
#     # remove HTML tags
#     text = re.sub(r"<[^>]+>", " ", text)
#     # collapse whitespace
#     text = re.sub(r"\s+", " ", text).strip()
#     # cut off after first sentence if long
#     if len(text.split(".")) > 1:
#         text = text.split(".")[0].strip() + "?"
#     # ensure ends with question mark if it's a question
#     if not text.endswith("?") and len(text.split()) < 15:
#         text += "?"
#     return text

# queries = []
# for i, thread in enumerate(selected_threads, 1):
#     posts = thread.get("posts", [])
#     if not posts:
#         continue
#     # prefer cooked text (HTML) or fall back to raw
#     content = posts[0].get("cooked") or posts[0].get("raw") or ""
#     cleaned_query = clean_text(content)
#     if cleaned_query:
#         queries.append({
#             "title": thread.get("title", ""),
#             "query": cleaned_query
#         })
#     else:
#         print(f"⚠️ Skipped thread {i} (no valid query text)")

# # --------------------------------------------------------------------
# # 5️⃣ Save queries to JSON
# # --------------------------------------------------------------------
# output_path.parent.mkdir(parents=True, exist_ok=True)
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(queries, f, indent=2, ensure_ascii=False)

# print(f"\n✅ Saved {len(queries)} cleaned queries to {output_path}")
