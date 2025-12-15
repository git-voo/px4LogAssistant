import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import json
import time
from rag_engine.rag_pipeline import RAGPipeline

# --------------------------------------------------------------------
# 1️⃣ Load all threads and filter to unseen 100 with an answer
# --------------------------------------------------------------------
with open("output/merged_threads.json", "r", encoding="utf-8") as f:
    threads = json.load(f)

print(f"Total threads: {len(threads)}")

# Skip the first 2500 (used in training / graph building)
test_candidates = threads
print(f"Remaining threads after skipping first 2500: {len(test_candidates)}")


# Keep only threads that have an 'answer' post
eligible_threads = [
    t for t in test_candidates
    if any(p.get("is_solution") for p in t.get("posts", []))
]

print(f"Eligible threads (have answer): {len(eligible_threads)}")

# Pick first 100
selected_threads = eligible_threads[:100]
print(f"Selected {len(selected_threads)} threads for evaluation set")

# --------------------------------------------------------------------
# 2️⃣ Prepare base dataset (query + reference only for now)
# --------------------------------------------------------------------
dataset = []
for t in selected_threads:
    first_post = t["posts"][0]["content"]
    answer_post = next(
        (p["content"] for p in t["posts"] if p.get("is_solution")), None
    )

    dataset.append({
        "query": first_post,
        "reference": answer_post
    })

print("✅ Built base query-reference pairs.")

# --------------------------------------------------------------------
# 3️⃣ Initialize RAG Pipeline
# --------------------------------------------------------------------
rag = RAGPipeline()

def get_context_and_response(query: str):
    """
    Gets both the retrieval context and the model response.
    Returns (context_text, response_text)
    """
    context, response = rag.answer(query)
    return context, response

# --------------------------------------------------------------------
# 4️⃣ Enrich each dataset entry with RAG context + response
# --------------------------------------------------------------------
final_dataset = []
for i, sample in enumerate(dataset, 1):
    query = sample["query"]
    reference = sample["reference"]
    print(f"\n[{i}/{len(dataset)}] Processing query: {query[:80]}...")

    try:
        context, response = get_context_and_response(query)
        final_dataset.append({
            "query": query,
            "context": context,
            "response": response,
            "reference": reference
        })
    except Exception as e:
        print(f"⚠️ Error processing sample {i}: {e}")
        continue

    # Optional small delay to avoid overloading the API
    time.sleep(2)

# --------------------------------------------------------------------
# 5️⃣ Save full dataset for DeepEval
# --------------------------------------------------------------------
output_path = "rag_eval/data/test_cases.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, indent=2, ensure_ascii=False)

print(f"\n✅ Finished building dataset with {len(final_dataset)} samples.")
print(f"Saved to {output_path}")
