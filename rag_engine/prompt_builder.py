def build_prompt(query, context_graphs):
    context_parts = []

    for graph in context_graphs:
        title = graph["title"]
        nodes = [f"[{n['type']}] {n['label']}" for n in graph["nodes"]]
        context_parts.append(f"Title: {title}\n" + "\n".join(nodes))

    context_text = "\n\n".join(context_parts)
    
    prompt = f"""You are a PX4 flight assistant. Answer the user's query using only the information below.
Context:
{context_text}

User query:
{query}

Answer:"""

    print("\n🤖 Context being sent to the LLM:\n")


    return prompt
