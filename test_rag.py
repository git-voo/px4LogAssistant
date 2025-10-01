from rag_engine.rag_pipeline import RAGPipeline

rag = RAGPipeline()
response = rag.answer("What causes motor speed imbalance during flight?")
print(response)


