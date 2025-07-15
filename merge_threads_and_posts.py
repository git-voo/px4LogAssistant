import json
from collections import defaultdict
from pathlib import Path

# Load input files
with open("px4_threads.json", "r", encoding="utf-8") as f:
    threads = json.load(f)

with open("px4_posts.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

print(f"🔄 Loaded {len(threads)} threads and {len(posts)} posts...")

# Group posts by their topic_id (which links them to a thread)
posts_by_thread = defaultdict(list)
for post in posts:
    posts_by_thread[post["topic_id"]].append(post)

print("✅ Grouped posts by thread ID.")

# Merge posts into each thread
merged_threads = []
for i, thread in enumerate(threads, 1):
    thread_id = thread["id"]
    thread_posts = posts_by_thread.get(thread_id, [])
    print(f"[{i}/{len(threads)}] Merging Thread ID {thread_id} ({len(thread_posts)} posts)...")

    merged = {
        "thread_id": thread_id,
        "title": thread["title"],
        "created_at": thread["created_at"],
        "last_posted_at": thread["last_posted_at"],
        "category": thread["category"],
        "tags": thread.get("tags", []),
        "posts": thread_posts
    }
    merged_threads.append(merged)

# Save to output file
Path("output").mkdir(exist_ok=True)
output_path = "output/merged_threads.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_threads, f, indent=2)

print(f"\n✅ Merged {len(merged_threads)} threads into {output_path}")
