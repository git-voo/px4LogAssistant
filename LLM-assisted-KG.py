
# """
# LLM-Assisted Knowledge Graph Extraction

# This script processes forum threads, sends them to Gemini (or another LLM),
# and receives structured knowledge graph data as JSON nodes and edges.

# Steps:
# 1. Load merged thread+post data
# 2. For each thread, construct a prompt based on the schema
# 3. Send to Gemini LLM using Google GenerativeAI API
# 4. Parse and store graph data
# 5. Save results in knowledge_graph.json and optionally export GraphML

# Author: Victor Onoja
# """

# import json
# import time
# from pathlib import Path
# import google.generativeai as genai

# # === Config ===
# API_KEY = os.getenv("GOOGLE_API_KEY")
# MERGED_FILE = "output/merged_threads.json"
# KG_JSON_PATH = "output/knowledge_graph.json"
# KG_GRAPHML_PATH = "output/knowledge_graph.graphml"

# # === Gemini Setup ===
# genai.configure(api_key=API_KEY)
# model = genai.GenerativeModel("gemini-2.5-pro")


# # === Prompt Template ===
# SYSTEM_PROMPT = """
# You are an intelligent knowledge graph builder. Given a forum thread with posts, your job is to extract entities and relationships based on this schema:

# Node types:
# - Thread
# - Post
# - Theme
# - Parameter
# - FirmwareVersion
# - SoftwareModule
# - HardwareModule
# - Environment
# - ErrorCode
# - ActionCommand
# - Solution

# Edge types:
# - HAS_POST (Thread → Post)
# - IS_SOLUTION (Post → Solution)
# - HAS_TAG (Thread → Tag)
# - RUNS_VERSION (Thread → FirmwareVersion)
# - FOCUSES_ON (Thread → Theme)
# - DESCRIBED_BY / ADDRESSED_BY (Theme → Post)
# - MENTIONS_PARAM / MENTIONS_MODULE (Post → Parameter/Module)
# - INVOKES_ACTION (Post → ActionCommand)
# - MENTIONS_ERROR_CODE (Post → ErrorCode)
# - DESCRIBES_ENVIRONMENT (Post → Environment)

# Return your output as:
# {
#   "nodes": [...],
#   "edges": [...]
# }
# """


import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re
from itertools import cycle

# === Load environment variables ===
load_dotenv()

# === Configure API key rotation ===
api_keys = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
    os.getenv("GOOGLE_API_KEY_6"),
]

key_cycle = cycle(api_keys)
current_key = next(key_cycle)

# === Gemini Model Config ===
def configure_model():
    genai.configure(api_key=current_key)
    return genai.GenerativeModel("gemini-2.5-pro")

model = configure_model()

# === Load threads ===
with open("output/merged_threads.json", "r", encoding="utf-8") as f:
    threads = json.load(f)

# === Load existing progress if any ===
output_path = "output/all_threads_graphs.json"
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        existing_graphs = json.load(f)["graphs"]
else:
    existing_graphs = []
    
print("🔄 Loaded existing graphs:", len(existing_graphs))
 
starting_index = len(existing_graphs)
print(f"🔄 Resuming from thread index {starting_index}...")

# === Schema instructions ===
schema_instructions = """
You are a knowledge graph extractor.
Extract structured graph data from each thread following this schema:

Each thread should be turned into its own subgraph with:
1. A "title".
2. A list of "nodes" with these fields:
    - id (string)
    - type (string) [Thread, Post, Theme, SoftwareModule, HardwareModule, Parameter, ActionCommand, ErrorCode, Environment, FirmwareVersion, UserRole, Solution]
    - label (string, natural language)
    - content (string) — main descriptive info
3. A list of "edges" with:
    - source (id)
    - target (id)
    - label (relationship type like HAS_POST, FOCUSES_ON, MENTIONS_PARAM, IS_SOLUTION, HAS_TAG, RUNS_VERSION, DESCRIBES_ENVIRONMENT, INVOKES_ACTION, MENTIONS_MODULE, MENTIONS_ERROR_CODE)

Avoid repetition. Prioritize meaningful concepts.
"""

output_graphs = existing_graphs.copy()

# === Function to rotate keys ===
def rotate_key():
    global current_key, model
    current_key = next(key_cycle)
    model = configure_model()
    print("🔁 Switched to new API key")

# === Main thread processing ===
for i, thread in enumerate(threads[starting_index:], start=starting_index):
    print(f"\n--- Processing Thread {i+1}/{len(threads)}: {thread['title']} ---")

    prompt = f"""
{schema_instructions}

Thread:
Title: {thread['title']}
Posts:
"""
    for post in thread["posts"]:
        prompt += f"\nPost {post['post_id']} by {post['author_username']} on {post['created_at']}\n"
        prompt += post["cooked"] + "\n"
        if post.get("is_solution"):
            prompt += "(This post is marked as the solution)\n"

    while True:
        try:
            response = model.generate_content(prompt, generation_config={"temperature": 0.4})
            structured = response.text
            match = re.search(r"{.*}", structured, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in LLM response.")

            json_text = match.group()
            parsed = json.loads(json_text)

            output_graphs.append({
                "title": thread["title"],
                "nodes": parsed.get("nodes", []),
                "edges": parsed.get("edges", [])
            })
            print(f"✅ Finished: {thread['title']}")
            break  # Success, break out of retry loop

        except Exception as e:
            error_str = str(e).lower()
            print(f"⚠️ Error: {e}")
            if "quota" in error_str or "429" in error_str:
                rotate_key()
                continue  # Retry same thread with next key
            else:
                print(f"❌ Skipping thread '{thread['title']}' due to error.")
                break

    # Save intermediate output after each thread
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"graphs": output_graphs}, f, indent=2)

print("\n🎉 All available threads processed. Graphs saved to output/all_thread_graphs.json")
