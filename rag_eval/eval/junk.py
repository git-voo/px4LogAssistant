import json
import os
import re
import time
import google.generativeai as genai
from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from ollama_model import OllamaLLM

load_dotenv()

# ==============================
# 1️⃣ Gemini Model with Key Rotation
# ==============================
class GeminiModel(DeepEvalBaseLLM):
    """Gemini evaluator with automatic API key rotation and throttling."""

    _gemini_keys = [
        v for k, v in os.environ.items()
        if re.match(r"GOOGLE_API_KEY_\d+", k) and v.strip()
    ]
    _current_index = 0
    _usage_count = 0
    _MAX_CALLS_PER_KEY = 50   # limit per API key
    _RATE_LIMIT_DELAY = 30    # seconds between requests (~2/min)

    def __init__(self, model_name="gemini-2.5-pro"):
        if not self._gemini_keys:
            raise ValueError("❌ No valid Gemini API keys found in environment!")

        # Rotate key if limit reached
        if self._usage_count >= self._MAX_CALLS_PER_KEY:
            self._current_index = (self._current_index + 1) % len(self._gemini_keys)
            self._usage_count = 0
            print(f"🔄 Rotated to next Gemini API key: index {self._current_index + 1}/{len(self._gemini_keys)}")

        self.api_key = self._gemini_keys[self._current_index]
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        print(f"🔑 Using Gemini key: {self.api_key[:6]}... (index {self._current_index + 1}/{len(self._gemini_keys)})")

    def _respect_rate_limit(self):
        """Pause to respect Gemini's 2 requests/min limit."""
        print(f"⏳ Sleeping {self._RATE_LIMIT_DELAY}s to respect rate limits...")
        time.sleep(self._RATE_LIMIT_DELAY)

    def generate(self, prompt: str) -> str:
        """Synchronous call with key rotation & throttling."""
        self._usage_count += 1
        self._respect_rate_limit()
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}. Switching key...")
            # Force rotate key if error occurs (quota exhaustion)
            self._usage_count = self._MAX_CALLS_PER_KEY
            return self.generate(prompt)

    async def a_generate(self, prompt: str) -> str:
        """Async call (unused in this pipeline)."""
        return self.generate(prompt)

    def load_model(self):
        return self.model

    def get_model_name(self):
        return self.model_name


# ==============================
# 2️⃣ Helper for Cleaning Text
# ==============================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore").replace("â€™", "'").strip()


# ==============================
# 3️⃣ Load Dataset
# ==============================
with open("rag_eval/data/test_cases.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

test_cases = []
for s in samples:
    test_cases.append(
        LLMTestCase(
            input=clean_text(s["query"]),
            retrieval_context=[clean_text(s["context"])],
            actual_output=clean_text(s["response"]),
            expected_output=clean_text(s["reference"]),
        )
    )

print(f"✅ Prepared {len(test_cases)} test cases for evaluation.")


# ==============================
# 4️⃣ Define Evaluation Metrics
# ==============================
# You can switch between Ollama (local) or Gemini (API)
# ollamaModel = OllamaLLM("deepseek-r1")

metrics = [
    FaithfulnessMetric(model=GeminiModel(), async_mode=False),
    ContextualPrecisionMetric(model=GeminiModel(), async_mode=False),
    ContextualRecallMetric(model=GeminiModel(), async_mode=False),
    AnswerRelevancyMetric(model=GeminiModel(), async_mode=False),
]


# ==============================
# 5️⃣ Incremental Evaluation Runner
# ==============================
def run_incremental_evaluation(test_cases, metrics, save_path="rag_eval/data/eval_results.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Resume if file exists
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    else:
        results_data = {"summary": {}, "details": []}

    details = results_data.get("details", [])
    completed = {d["test_name"] for d in details}
    remaining_cases = [tc for tc in test_cases if tc.name not in completed]

    print(f"🔁 Resuming from {len(completed)} done, {len(remaining_cases)} remaining.")

    all_scores = [md["score"] for d in details for md in d["metrics"] if md.get("score") is not None]
    success_count = sum(d["success"] for d in details)

    for i, test_case in enumerate(remaining_cases, start=1):
        print(f"\n⚙️ Evaluating test case {i}/{len(remaining_cases)}...")

        try:
            result = evaluate(test_cases=[test_case], metrics=metrics)
            test_results = result.test_results
        except Exception as e:
            print(f"⚠️ Error on test case {i}: {e}")
            details.append({
                "test_name": f"test_case_{i}",
                "success": False,
                "error": str(e),
                "metrics": []
            })
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            continue

        for tr in test_results:
            success_count += int(tr.success)
            entry = {
                "test_name": tr.name,
                "success": tr.success,
                "metrics": [
                    {
                        "name": md.name,
                        "score": float(md.score) if md.score is not None else None,
                        "passed": md.success,
                        "reason": md.reason,
                    }
                    for md in tr.metrics_data
                ],
            }

            # Add scores and update summary
            test_scores = [float(md.score) for md in tr.metrics_data if md.score is not None]
            if test_scores:
                all_scores.extend(test_scores)

            details.append(entry)
            results_data["details"] = details
            results_data["summary"] = {
                "num_tests": len(details),
                "pass_rate": success_count / len(details),
                "avg_score": (sum(all_scores) / len(all_scores)) if all_scores else None,
            }

            # Save after each test
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Saved progress ({len(details)} total) → {save_path}")

    print(f"\n🎯 Evaluation complete — total {len(details)} test cases saved.")
    return results_data


# ==============================
# 6️⃣ Run It
# ==============================
if __name__ == "__main__":
    run_incremental_evaluation(test_cases, metrics)
