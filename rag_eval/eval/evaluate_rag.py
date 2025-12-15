import json
import os
import google.generativeai as genai
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from dotenv import load_dotenv
from ollama_model import OllamaLLM

load_dotenv()
ollamaModel = OllamaLLM("deepseek-r1")


# === Gemini wrapper (optional) ===
class GeminiModel(DeepEvalBaseLLM):
    def __init__(self, model_name="gemini-2.5-pro"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    async def a_generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    def load_model(self):
        return self.model

    def get_model_name(self):
        return self.model_name


# === Helper for cleaning text ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore").replace("â€™", "'").strip()


# === Load dataset ===
with open("rag_eval/data/test_cases.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

test_cases = []
for s in samples:
    query = clean_text(s["query"])
    context = clean_text(s["context"])
    output = clean_text(s["response"])
    reference = clean_text(s["reference"])

    test_cases.append(
        LLMTestCase(
            input=query,
            retrieval_context=[context],
            actual_output=output,
            expected_output=reference,
        )
    )

print(f"✅ Prepared {len(test_cases)} test cases for evaluation.")


# === Define metrics ===
metrics = [
    FaithfulnessMetric(model=GeminiModel()),
    ContextualPrecisionMetric(model=GeminiModel()),
    ContextualRecallMetric(model=GeminiModel()),
    AnswerRelevancyMetric(model=GeminiModel()),
    # FaithfulnessMetric(model=ollamaModel, async_mode=False),
    # ContextualPrecisionMetric(model=ollamaModel, async_mode=False),
    # ContextualRecallMetric(model=ollamaModel, async_mode=False),
    # AnswerRelevancyMetric(model=ollamaModel, async_mode=False),
    # AnswerCorrectnessMetric(model=ollamaModel, async_mode=False),
]


# === Incremental evaluation function ===
def run_incremental_evaluation(test_cases, metrics, save_path="rag_eval/data/gemini_eval/eval_results.json"):
    """
    Runs DeepEval metrics per test case, saving progress after each iteration.
    Safely skips malformed outputs or JSON errors from DeepSeek.
    """
    import time

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load existing results if resuming
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
            print(f"⚠️ Skipping test {i} due to error: {e}")
            # Log failure entry so it’s not retried endlessly
            details.append({
                "test_name": f"test_case_{i}",
                "success": False,
                "error": str(e),
                "metrics": []
            })
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            continue

        # --- Save successful test results ---
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

            # Collect scores
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

            # Save progress safely
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Saved progress ({len(details)} total) → {save_path}")

        # Give Ollama some breathing room between tests
        time.sleep(3)

    print(f"\n🎯 Evaluation complete — total {len(details)} test cases saved.")
    return results_data

# === Run the incremental evaluation ===
if __name__ == "__main__":
    run_incremental_evaluation(test_cases, metrics)

# from deepeval.metrics import BaseMetric

# class ExpertAlignmentMetric(BaseMetric):
#     def __init__(self, model):
#         self.model = model
#         self.metric_name = "ExpertAlignmentMetric"

#     async def a_measure(self, test_case):
#         prompt = f"""
#         Community context: {test_case.retrieval_context}
#         Model answer: {test_case.actual_output}
#         Reference (expert consensus): {test_case.expected_output}

#         Rate from 0-1 how well the model's reasoning aligns with community expert reasoning.
#         """
#         response = await self.model.a_generate(prompt)
#         score = float(response.strip() or 0.0)
#         self.score = min(max(score, 0), 1)
#         self.reason = "LLM-based evaluation of expert reasoning alignment"



# # Script to check for available metrics

# import deepeval, pkgutil
# print("DeepEval version:", deepeval.__version__)
# print("\nAvailable metric modules:")
# for m in pkgutil.iter_modules(deepeval.metrics.__path__):
#     print(" -", m.name)

