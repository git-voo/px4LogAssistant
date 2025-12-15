# from .retriever import GraphRetriever
# from .prompt_builder import build_prompt
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv
# load_dotenv()

# class RAGPipeline:
#     def __init__(self, gemini_api_key=None):
#         print("🤖 Initializing RAG Pipeline...")
#         genai.configure(api_key=gemini_api_key or os.getenv("GOOGLE_API_KEY_1"))
#         self.model = genai.GenerativeModel("gemini-2.5-pro")
#         self.retriever = GraphRetriever()

#     def answer(self, query):
#         print("🤖 Generating answer for query:", query)
#         relevant_graphs = self.retriever.retrieve(query)
#         prompt, context_text = build_prompt(query, relevant_graphs)
#         response = self.model.generate_content(prompt)
#         return context_text, response.text



# The script below replaces the above with Ollama model integration in place of Gemini
from .retriever import GraphRetriever
from .prompt_builder import build_prompt
from rag_eval.eval.ollama_model import OllamaLLM

import os
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, model_name=None):
        """
        RAG pipeline using the local DeepSeek-R1 Ollama model.
        """
        print("🤖 Initializing RAG Pipeline (DeepSeek-R1 via Ollama)...")
        # default model name can be set in .env or argument
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "deepseek-r1")
        self.model = OllamaLLM(model_name=self.model_name)
        self.retriever = GraphRetriever()

    def answer(self, query: str):
        """
        Retrieves relevant graphs, builds prompt, and gets response.
        Returns (context_text, response_text).
        """
        print("🤖 Generating answer for query:", query)
        relevant_graphs = self.retriever.retrieve(query)
        prompt, context_text = build_prompt(query, relevant_graphs)

        try:
            # Use synchronous generation for simplicity
            response_text, _ = self.model.generate(prompt)
        except Exception as e:
            response_text = f"⚠️ DeepSeek generation error: {e}"

        return context_text, response_text
