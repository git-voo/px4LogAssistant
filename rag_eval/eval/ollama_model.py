"""
OllamaLLM: DeepEval-compatible local model wrapper
--------------------------------------------------

Allows DeepEval metrics (Faithfulness, ContextualPrecision, etc.)
to use locally running Ollama models via the OpenAI-compatible
API served at http://localhost:11434/v1/chat/completions.
"""

from typing import Any, Tuple, Union, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_community.callbacks import get_openai_callback
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import trimAndLoadJson



import hashlib, json, os, re, aiohttp, requests

CACHE_FILE = ".ollama_cache.json"
_cache = {}

# Load cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        _cache = json.load(f)

def cache_lookup(prompt: str):
    key = hashlib.sha256(prompt.encode()).hexdigest()
    return _cache.get(key)

def cache_store(prompt: str, response: str):
    key = hashlib.sha256(prompt.encode()).hexdigest()
    _cache[key] = response
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f)



# ---- Custom Chat wrapper ----------------------------------------------------
class CustomChatOpenAI(ChatOpenAI):
    """Extends LangChain's ChatOpenAI to optionally force JSON output."""
    format: str = None

    def __init__(self, format: str = None, **kwargs):
        super().__init__(**kwargs)
        self.format = format

    async def _acreate(
        self, messages: List[BaseMessage], **kwargs
    ) -> ChatResult:
        if self.format:
            kwargs["format"] = self.format
        return await super()._acreate(messages, **kwargs)


# ---- Main DeepEval-compatible LLM ------------------------------------------
class OllamaLLM(DeepEvalBaseLLM):
    """
    DeepEvalBaseLLM implementation for local Ollama models.
    Example:
        model = OllamaLLM("gpt-oss")
        text = model.generate("Summarize PX4 logs in one sentence.")
    """

    def __init__(
        self,
        model_name: str = "gpt-oss",
        base_url: str = "http://localhost:11434/v1/",
        json_mode: bool = True,
        temperature: float = 0.0,
        **kwargs,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.json_mode = json_mode
        self.temperature = temperature
        self.kwargs = kwargs
        super().__init__(model_name)

    # ------------------------------------------------------------------
    def load_model(self) -> CustomChatOpenAI:
        """Create a ChatOpenAI interface bound to the local Ollama API."""
        return CustomChatOpenAI(
            model_name=self.model_name,
            openai_api_key="ollama",           # dummy key; Ollama ignores it
            base_url=self.base_url,
            format="json" if self.json_mode else None,
            temperature=self.temperature,
            **self.kwargs,
        )

    # ------------------------------------------------------------------
    def generate(
        self, prompt: str, schema: Any = None, **kwargs
    ) -> Union[Any, Tuple[str, float]]:
        """Synchronous text generation."""
        
        cached = cache_lookup(prompt)
        if cached:
            return cached, 0.0

        chat_model = self.load_model()
        with get_openai_callback() as _:
            res = chat_model.invoke(prompt)
            content = res.content if hasattr(res, "content") else str(res)

            if schema is not None:
                try:
                    data = trimAndLoadJson(content, None)
                    return schema(**data)
                except Exception:
                    return trimAndLoadJson(content, None)
       
            cache_store(prompt, content)

            return content, 0.0

    # ------------------------------------------------------------------
    async def a_generate(
        self, prompt: str, schema: Any = None, **kwargs
    ) -> Union[Any, Tuple[str, float]]:
        """Asynchronous text generation."""
        chat_model = self.load_model()
        with get_openai_callback() as _:
            res = await chat_model.ainvoke(prompt)
            content = res.content if hasattr(res, "content") else str(res)

            if schema is not None:
                try:
                    data = trimAndLoadJson(content, None)
                    return schema(**data)
                except Exception:
                    return trimAndLoadJson(content, None)
            return content, 0.0
        
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name
    
    

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Model display name."""
        return self.model_name















































# import requests
# from deepeval.models import DeepEvalBaseLLM

# class OllamaModel(DeepEvalBaseLLM):
#     def __init__(self, model_name="gpt-oss"):
#         self.model_name = model_name
#         self.api_url = "http://localhost:11434/api/generate"

#     def load_model(self):
#         # Optional: used if model loading is required, but Ollama handles it internally
#         pass

#     def generate(self, prompt: str) -> str:
#         response = requests.post(
#             self.api_url,
#             json={"model": self.model_name, "prompt": prompt},
#             stream=True
#         )
#         output = ""
#         for line in response.iter_lines():
#             if line:
#                 data = line.decode("utf-8")
#                 if '"response":"' in data:
#                     chunk = data.split('"response":"')[1].split('"')[0]
#                     output += chunk
#         return output
    
#     async def a_generate(self, prompt: str) -> str:
#         """
#         Asynchronous text generation for DeepEval async metrics.
#         """
#         output = ""
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 self.api_url,
#                 json={"model": self.model_name, "prompt": prompt}
#             ) as resp:
#                 async for line in resp.content:
#                     data = line.decode("utf-8")
#                     if '"response":"' in data:
#                         chunk = data.split('"response":"')[1].split('"')[0]
#                         output += chunk
#         return output
    
#     @property
#     def name(self):
#         return self.model_name
    
#     def get_model_name(self):
#         return self.model_name
