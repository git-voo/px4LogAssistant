# fetch_px4_data.py
import requests
import time
import json
import os

BASE_URL = "https://discuss.px4.io"

OUTPUT_THREADS = "px4_threads.json"
OUTPUT_POSTS   = "px4_posts.json"

# Pause between requests to avoid hitting rate limits
REQUEST_DELAY = 0.5

def fetch_json(path, params=None):
    """Helper to GET BASE_URL+path and return parsed JSON (or None on failure)."""
    url = BASE_URL + path
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Failed GET {url} [{resp.status_code}]")
        return None

def collect_all_categories():
    """
    Returns a list of category slugs, e.g. ['flight-review','development', …].
    """
    data = fetch_json("/categories.json")
    if not data or "category_list" not in data:
        return []
    cats = data["category_list"]["categories"]
    return [c["slug"] for c in cats]

def collect_topics_for_category(category_slug):
    """
    Page through /c/{slug}/l/latest.json to collect all topics in that category.
    Returns a list of thread‐level dicts (id, title, slug, created_at, tags, etc.).
    """
    all_topics = []
    page = 0
    while True:
        path = f"/c/{category_slug}/l/latest.json"
        params = {"page": page}
        data = fetch_json(path, params=params)
        if not data or "topic_list" not in data:
            break

        topics = data["topic_list"]["topics"]
        if not topics:
            break

        # Extract whichever fields you need; here we keep id, title, slug, created_at, last_posted_at, tags
        for t in topics:
            thread_info = {
                "id": t["id"],
                "title": t["title"],
                "slug": t["slug"],
                "created_at": t["created_at"],
                "last_posted_at": t["last_posted_at"],
                "category": category_slug,
                "tags": t.get("tags", []),
            }
            all_topics.append(thread_info)

        page += 1
        time.sleep(REQUEST_DELAY)
    return all_topics

def collect_posts_for_thread(thread_id):
    """
    Fetch /t/{thread_id}.json to get all posts for that topic.
    Returns a list of post‐level dicts (post id, author, cooked/html body, raw body, is_solution, etc.)
    """
    path = f"/t/{thread_id}.json"
    data = fetch_json(path)
    if not data or "post_stream" not in data:
        return []

    posts = []
    for p in data["post_stream"]["posts"]:
        post_info = {
            "post_id": p["id"],
            "topic_id": thread_id,
            "author_username": p["username"],
            "created_at": p["created_at"],
            "cooked": p.get("cooked", ""),                  # HTML‐formatted body
            "raw": p.get("raw", ""),                        # plain‐text fallback if missing
            "is_solution": p.get("post_number", None) == data.get("accepted_answer_id"),
            "reply_to_post_number": p.get("reply_to_post_number"),
        }
        posts.append(post_info)
    return posts


# def collect_posts_for_thread(thread_id):
#     """
#     Fetch /t/{thread_id}.json to get all posts for that topic.
#     Returns a list of post‐level dicts (post id, author, cooked/html body, raw body, is_solution, etc.)
#     """
#     path = f"/t/{thread_id}.json"
#     data = fetch_json(path)
#     if not data or "post_stream" not in data:
#         return []

#     posts = []
#     for p in data["post_stream"]["posts"]:
#         post_info = {
#             "post_id": p["id"],
#             "topic_id": thread_id,
#             "author_username": p["username"],
#             "created_at": p["created_at"],
#             "cooked": p["cooked"],    # HTML‐formatted body
#             "raw": p["raw"],          # plain text
#             "is_solution": p.get("post_number", None) == data.get("accepted_answer_id"),
#             "reply_to_post_number": p.get("reply_to_post_number"),
#         }
#         posts.append(post_info)
#     return posts

def main():
    # # Step 1: collect all categories
    # categories = collect_all_categories()
    # print(f"Found categories: {categories}")

    # # Step 2: for each category, collect all threads
    # all_threads = []
    # for cat in categories:
    #     print(f"Collecting threads for category '{cat}' …")
    #     threads = collect_topics_for_category(cat)
    #     print(f"  → {len(threads)} threads")
    #     all_threads.extend(threads)

    # # Deduplicate by thread ID (in case a thread appears in multiple pages)
    # unique_threads = {t["id"]: t for t in all_threads}
    # all_threads = list(unique_threads.values())
    # print(f"Total unique threads: {len(all_threads)}")

    # # Write threads to file
    # with open(OUTPUT_THREADS, "w", encoding="utf-8") as f:
    #     json.dump(all_threads, f, indent=2)
    # print(f"Wrote {len(all_threads)} threads to {OUTPUT_THREADS}")

    # Step 3: for each thread, fetch all posts

    # to collect posts for threads already in px4_threads.json:
    with open(OUTPUT_THREADS, "r", encoding="utf-8") as f:
        all_threads = json.load(f)

    all_posts = []
    for idx, thread in enumerate(all_threads, 1):
        tid = thread["id"]
        print(f"[{idx}/{len(all_threads)}] Fetching posts for thread {tid} …")
        posts = collect_posts_for_thread(tid)
        print(f"   → {len(posts)} posts")
        all_posts.extend(posts)
        time.sleep(REQUEST_DELAY)

    # Deduplicate posts by post_id
    unique_posts = {p["post_id"]: p for p in all_posts}
    all_posts = list(unique_posts.values())
    print(f"Total unique posts: {len(all_posts)}")

    # Write posts to file
    with open(OUTPUT_POSTS, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, indent=2)
    print(f"Wrote {len(all_posts)} posts to {OUTPUT_POSTS}")

if __name__ == "__main__":
    main()
