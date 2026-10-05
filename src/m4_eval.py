from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, GEMINI_MODEL, TEST_SET_PATH


def _finite_score(value) -> float:
    try:
        score = float(value)
        return score if math.isfinite(score) else 0.0
    except (TypeError, ValueError):
        return 0.0


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    try:
        if not GEMINI_API_KEY:
            raise ValueError("Set GEMINI_API_KEY in .env to enable RAGAS evaluation")
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        from ragas.llms import LangchainLLMWrapper
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from src.llm import gemini_chat_model

        answer_relevancy.strictness = 1

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=LangchainLLMWrapper(gemini_chat_model(temperature=0)),
            embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"}),
            run_config=RunConfig(max_workers=1, max_retries=10, timeout=120),
        )
        rows = result.to_pandas().to_dict(orient="records")
        per_question = [EvalResult(
            question=row["question"], answer=row["answer"], contexts=row["contexts"],
            ground_truth=row["ground_truth"],
            **{name: _finite_score(row.get(name, 0.0)) for name in metric_names},
        ) for row in rows]
        aggregate = {name: _finite_score(result[name]) for name in metric_names}
        return {**aggregate, "evaluation_status": "completed",
                "evaluation_model": GEMINI_MODEL, "embedding_model": GEMINI_EMBEDDING_MODEL,
                "per_question": per_question}
    except Exception as exc:
        print(f"  ⚠️  RAGAS evaluation failed: {exc}")
        per_question = [EvalResult(q, a, c, gt, 0.0, 0.0, 0.0, 0.0)
                        for q, a, c, gt in zip(questions, answers, contexts, ground_truths)]
        return {**{name: 0.0 for name in metric_names}, "evaluation_status": "unavailable",
                "per_question": per_question, "evaluation_error": str(exc)}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("Answer may contain claims unsupported by retrieved context.",
                         "Tighten the context only prompt and lower generation temperature."),
        "context_recall": ("Retrieved context is missing facts needed for the answer.",
                           "Improve chunk boundaries and add lexical BM25 coverage."),
        "context_precision": ("Retrieved context contains irrelevant chunks.",
                              "Rerank candidates and filter by document metadata."),
        "answer_relevancy": ("Answer does not directly address the question.",
                             "Improve the answer prompt and preserve the user question."),
    }
    metrics = tuple(diagnostic_tree)
    ranked = sorted(eval_results, key=lambda row: sum(getattr(row, m) for m in metrics) / len(metrics))
    failures = []
    for row in ranked[:max(0, bottom_n)]:
        worst = min(metrics, key=lambda name: getattr(row, name))
        diagnosis, suggested_fix = diagnostic_tree[worst]
        failures.append({
            "question": row.question, "answer": row.answer,
            "contexts": row.contexts, "ground_truth": row.ground_truth,
            "worst_metric": worst, "score": float(getattr(row, worst)),
            "average_score": sum(getattr(row, m) for m in metrics) / len(metrics),
            "diagnosis": diagnosis, "suggested_fix": suggested_fix,
            "error_tree": f"Answer incorrect/unsupported → inspect retrieved context → "
                          f"{'context incomplete' if worst == 'context_recall' else 'inspect query and ranking'} → {suggested_fix}",
        })
    return failures


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: float(v) if isinstance(v, (int, float)) else v
                      for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
