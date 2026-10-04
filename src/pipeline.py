from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import json, os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import GEMINI_API_KEY, RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)
    latency = {}

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_lookup = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_lookup.update({(doc["metadata"].get("source", ""), p.metadata["parent_id"]): p.text
                              for p in parents})
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    latency["chunking_seconds"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({latency['chunking_seconds']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    latency["enrichment_seconds"] = time.time() - t0
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    latency["indexing_seconds"] = time.time() - t0
    print(f"  ✓ Indexed ({latency['indexing_seconds']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    latency["reranker_setup_seconds"] = time.time() - t0
    print(f"  ✓ Reranker ready ({latency['reranker_setup_seconds']:.1f}s)", flush=True)

    search.parent_lookup = parent_lookup
    search.latency = latency
    search.query_timings = []

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    timings = {}
    t0 = time.perf_counter()
    results = search.search(query)
    timings["retrieval_seconds"] = time.perf_counter() - t0
    docs, seen_parents = [], set()
    parent_lookup = getattr(search, "parent_lookup", {})
    for result in results:
        metadata = result.metadata
        parent_key = (metadata.get("source", ""), metadata.get("parent_id"))
        text = parent_lookup.get(parent_key, result.text)
        if parent_key[1] and parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)
        docs.append({"text": text, "score": result.score, "metadata": metadata})
    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    timings["reranking_seconds"] = time.perf_counter() - t0
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    t0 = time.perf_counter()
    if GEMINI_API_KEY and contexts:
        try:
            context_str = "\n\n".join(contexts)
            from src.llm import gemini_chat_model
            answer = gemini_chat_model(temperature=0).invoke([
                ("system", "Bạn là trợ lý chính sách nhân sự. Trả lời ngắn gọn bằng tiếng Việt, chỉ dựa trên context. Khi có nhiều phiên bản, ưu tiên phiên bản mới nhất còn hiệu lực. Giữ nguyên các phủ định, số liệu và điều kiện. Nếu context không đủ, nêu rõ phần còn thiếu; không suy đoán."),
                ("human", f"Context:\n{context_str}\n\nCâu hỏi: {query}"),
            ]).content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    timings["answer_generation_seconds"] = time.perf_counter() - t0
    if hasattr(search, "query_timings"):
        search.query_timings.append(timings)
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    query_start = time.time()
    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)
    query_seconds = time.time() - query_start

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    eval_seconds = time.time() - t0
    print(f"  ✓ RAGAS done ({eval_seconds:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if results.get("evaluation_error"):
            print(f"  — {m}: unavailable")
        else:
            s = results.get(m, 0)
            print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    if results.get("evaluation_error"):
        failures = []
        print("  ⚠️  Metrics unavailable; see evaluation_error in the report.", flush=True)
    else:
        failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures)
    query_timings = getattr(search, "query_timings", [])
    latency = {**getattr(search, "latency", {}),
               "query_total_seconds": query_seconds,
               "query_avg_seconds": query_seconds / max(len(test_set), 1),
               "reragas_evaluation_seconds": eval_seconds,
               "questions": len(test_set),
               "dense_enabled": search.dense._indexed,
               "reranker_mode": "cross_encoder" if reranker._model not in (None, False) else "lexical_fallback",
               "generation_mode": "gemini" if GEMINI_API_KEY else "local_fallback",
               "evaluation_status": results.get("evaluation_status", "unknown")}
    for key in ("retrieval_seconds", "reranking_seconds", "answer_generation_seconds"):
        latency[f"{key.removesuffix('_seconds')}_total_seconds"] = sum(row.get(key, 0) for row in query_timings)
        latency[f"{key.removesuffix('_seconds')}_avg_seconds"] = (
            latency[f"{key.removesuffix('_seconds')}_total_seconds"] / max(len(query_timings), 1))
    latency["pipeline_total_seconds"] = sum(latency.get(stage, 0) for stage in (
        "chunking_seconds", "enrichment_seconds", "indexing_seconds",
        "reranker_setup_seconds", "query_total_seconds", "reragas_evaluation_seconds"))
    os.makedirs("reports", exist_ok=True)
    with open("reports/latency_report.json", "w", encoding="utf-8") as f:
        json.dump(latency, f, ensure_ascii=False, indent=2)
    stages = ["chunking_seconds", "enrichment_seconds", "indexing_seconds",
              "retrieval_total_seconds", "reranking_total_seconds",
              "answer_generation_total_seconds", "reragas_evaluation_seconds"]
    with open("reports/latency_report.md", "w", encoding="utf-8") as f:
        f.write("# Pipeline Latency Breakdown\n\n")
        f.write(f"Total: **{latency['pipeline_total_seconds']:.2f}s** across {len(test_set)} questions.\n\n")
        f.write(f"Modes: dense={'on' if latency['dense_enabled'] else 'off'}, "
                f"reranker={latency['reranker_mode']}, generation={latency['generation_mode']}, "
                f"RAGAS={latency['evaluation_status']}.\n\n")
        f.write("| Stage | Duration (s) |\n|---|---:|\n")
        for stage in stages:
            f.write(f"| {stage.replace('_seconds', '').replace('_', ' ').title()} | {latency.get(stage, 0):.2f} |\n")
        f.write(f"| Average query | {latency['query_avg_seconds']:.2f} |\n")
    print("Latency breakdown saved to reports/latency_report.json and .md", flush=True)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
