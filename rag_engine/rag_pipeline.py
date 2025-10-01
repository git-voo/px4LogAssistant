from .retriever import GraphRetriever
from .prompt_builder import build_prompt
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

class RAGPipeline:
    def __init__(self, gemini_api_key=None):
        print("🤖 Initializing RAG Pipeline...")
        genai.configure(api_key=gemini_api_key or os.getenv("GOOGLE_API_KEY_1"))
        self.model = genai.GenerativeModel("gemini-2.5-pro")
        self.retriever = GraphRetriever()

    def answer(self, query):
        print("🤖 Generating answer for query:", query)
        relevant_graphs = self.retriever.retrieve(query)
        prompt = build_prompt(query, relevant_graphs)
        response = self.model.generate_content(prompt)
        return response.text
