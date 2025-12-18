def build_prompt(query, context_graphs):
    context_parts = []

    for graph in context_graphs:
        title = graph["title"]
        nodes = [f"[{n['type']}] {n['label']}" for n in graph["nodes"]]

        # Add relationship (edge) info
        if "edges" in graph:
            edges = [f"({e['source']}) -[{e['label']}]→ ({e['target']})" for e in graph["edges"]]
            edge_info = "\nRelationships:\n" + "\n".join(edges)
        else:
            edge_info = ""

        context_parts.append(f"Title: {title}\n" + "\n".join(nodes) + f"\n{edge_info}")

    context_text = "\n\n".join(context_parts)

    prompt = f"""You are a PX4 flight assistant. Answer the user's query using the information below and your own knowledge.
Context:
{context_text}

User query:
{query}

Answer:"""

    print("\n🤖 Context being sent to the LLM:\n")
    # print(prompt)

    return prompt,context_text
