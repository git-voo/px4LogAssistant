import json

with open("./px4_threads.json", "r") as threads_file:
    threads = json.loads(threads_file.read())

with open("./px4_posts.json", "r") as posts_file:
    posts = json.loads(posts_file.read())

print(f"There are {len(threads)} threads and {len(posts)} posts")