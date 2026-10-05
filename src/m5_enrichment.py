from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
import json
import re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY
from src.llm import gemini_chat_model


def _response_text(response) -> str:
    content = response.content
    return content if isinstance(content, str) else "".join(
        part.get("text", "") for part in content if isinstance(part, dict)
    )


def _fallback_enrichment(text: str, source: str) -> dict:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    questions = [f"{s.rstrip('.')}?" for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10][:3]
    lower = text.lower()
    category = ("it" if any(word in lower for word in ("mật khẩu", "vpn", "cntt")) else
                "finance" if any(word in lower for word in ("lương", "chi phí", "vnđ")) else "hr")
    return {
        "summary": " ".join(sentences[:2]) or text,
        "questions": questions,
        "context": f"Trích từ {source}." if source else "",
        "metadata": {"topic": text.splitlines()[0][:100] if text else "general",
                     "entities": [], "category": category, "language": "vi"},
    }


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if GEMINI_API_KEY:
        try:
            return _response_text(gemini_chat_model().invoke([
                ("system", "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."),
                ("human", text),
            ])).strip()
        except Exception as exc:
            print(f"  ⚠️  Gemini summarize failed: {exc}")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return " ".join(sentences[:2]) or text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if GEMINI_API_KEY:
        try:
            response = gemini_chat_model().invoke([
                ("system", f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."),
                ("human", text),
            ])
            return [q.strip().lstrip("0123456789.-) ") for q in _response_text(response).splitlines()
                    if q.strip()][:n_questions]
        except Exception as exc:
            print(f"  ⚠️  Gemini HyQA failed: {exc}")
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{sentence.rstrip('.')}?" for sentence in sentences[:max(0, n_questions)]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if GEMINI_API_KEY:
        try:
            response = gemini_chat_model().invoke([
                ("system", "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."),
                ("human", f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"),
            ])
            return f"{_response_text(response).strip()}\n\n{text}"
        except Exception as exc:
            print(f"  ⚠️  Gemini contextual failed: {exc}")
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if GEMINI_API_KEY:
        try:
            response = gemini_chat_model().invoke([
                ("system", 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'),
                ("human", text),
            ])
            return json.loads(_response_text(response))
        except Exception as exc:
            print(f"  ⚠️  Gemini metadata failed: {exc}")
    lower = text.lower()
    category = ("it" if any(word in lower for word in ("mật khẩu", "vpn", "cntt")) else
                "finance" if any(word in lower for word in ("lương", "chi phí", "vnđ")) else "hr")
    return {"topic": text.splitlines()[0][:100] if text else "general",
            "entities": [], "category": category, "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "enrichment_cache.json")
_ENRICHMENT_CACHE: dict[str, dict] = {}


def _load_enrichment_cache():
    global _ENRICHMENT_CACHE
    if not _ENRICHMENT_CACHE and os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                _ENRICHMENT_CACHE = json.load(f)
        except Exception:
            pass


def _save_enrichment_cache():
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_ENRICHMENT_CACHE, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    _load_enrichment_cache()
    cache_key = f"{source}:::{text}"
    if cache_key in _ENRICHMENT_CACHE:
        return _ENRICHMENT_CACHE[cache_key]

    result = None
    if GEMINI_API_KEY:
        try:
            response = gemini_chat_model().invoke([
                ("system", 'Phân tích đoạn văn, chỉ trả về JSON hợp lệ với khóa summary, questions, context, metadata (topic, entities, category, language).'),
                ("human", f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"),
            ])
            content = _response_text(response).strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            result = json.loads(content)
        except Exception as exc:
            print(f"  ⚠️  Enrichment API failed: {exc}")
    if not result:
        result = _fallback_enrichment(text, source)

    _ENRICHMENT_CACHE[cache_key] = result
    _save_enrichment_cache()
    return result


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "") or ""
            questions = result.get("questions", []) or []
            if isinstance(questions, str):
                questions = [questions]
            questions = [str(question) for question in questions]
            context_line = result.get("context", "") or ""
            enriched_text = "\n\n".join(part for part in (
                context_line,
                f"Tóm tắt: {summary}" if summary else "",
                "Câu hỏi liên quan: " + " | ".join(questions) if questions else "",
                text,
            ) if part)
            auto_meta = result.get("metadata", {})
            if not isinstance(auto_meta, dict):
                auto_meta = {}
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
