# Failure Analysis — Lab 18: Production RAG

**Họ và tên:** Trần Nam Anh  
**Khóa:** K4 - Track 3B

## Evaluation status

The pipeline processed 20 questions and wrote both reports, but RAGAS did not run: `No module named 'datasets'`. `OPENAI_API_KEY` is also not configured. The report records `evaluation_status: unavailable`; its zero placeholders are **not scores**. The baseline dense index was also unavailable because `qdrant-client` and `sentence_transformers` are not installed. Therefore there is no valid baseline/production comparison or RAGAS bottom-five for this run.

| Metric | Naive baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | N/A | N/A | N/A |
| Answer Relevancy | N/A | N/A | N/A |
| Context Precision | N/A | N/A | N/A |
| Context Recall | N/A | N/A | N/A |

## Five qualitative retrieval failures

These are manual examples from the local BM25 plus lexical fallback run, not a RAGAS-ranked bottom-five.

### 1. Leave for employee marriage

- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Expected:** 3 paid working days, not deducted from annual leave.
- **Observed:** The first reranked result came from `hoan_chi_dao_tao.md`; a relevant `nghi_phep_dac_biet.md` result was also retrieved.
- **Error Tree:** Answer incomplete → relevant source exists in candidates → fallback reranker scores word overlap and elevates unrelated “ngày” matches → load the cross-encoder and rerun.
- **Root cause:** The local fallback reranker is a token-overlap heuristic, not a semantic cross-encoder.
- **Suggested fix:** Install/load `BAAI/bge-reranker-v2-m3`; add a regression check that ranks the special-leave policy first.

### 2. Current annual-leave entitlement

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 days under v2024; v2023's 12 days is superseded.
- **Observed:** The top BM25 source was `nghi_phep_dac_biet.md`, which does not answer annual entitlement.
- **Error Tree:** Wrong policy retrieved → query is broad and overlaps other leave policies → no category/version filter → retrieve from annual-leave policy and prefer current versions.
- **Root cause:** BM25 scores shared terms such as “nghỉ phép” without understanding policy scope or version status.
- **Suggested fix:** Add category and effective-version metadata filters before reranking.

### 3. Seniority leave calculation

- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** v2024 adds one day per three years; v2023 required five years.
- **Observed:** The first result came from `nghi_phep_nam_v2023.md`.
- **Error Tree:** Version-sensitive answer may be stale → old and current policy both match → retrieval does not filter superseded documents → prefer v2024 and include the version in the answer context.
- **Root cause:** Old and current policies are indexed with no active/superseded filter.
- **Suggested fix:** Extract document version/status and filter to the latest effective policy for current-policy questions.

### 4. Password change interval

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Every 120 days under v2.0; v1.0's 90 days is superseded.
- **Observed:** `mat_khau_v1.md` ranked first and `mat_khau_v2.md` second.
- **Error Tree:** Likely stale answer → both policy chunks are retrieved → lexical ranking favors repeated wording in v1 → apply active-version metadata filter before final ranking.
- **Root cause:** The retrieval stage does not distinguish a superseded policy from the active one.
- **Suggested fix:** Use version metadata and rerank with the complete question plus version constraints.

### 5. Laptop purchase approval (multi-hop)

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director approval, IT configuration confirmation, and at least three quotes.
- **Observed:** The first result came from `hoan_chi_dao_tao.md`; purchase-specific evidence was not ranked first.
- **Error Tree:** Multi-part answer at risk → top result is from the wrong topic → one BM25 query handles approval, IT configuration, and quotes together → retrieve purchase and IT policy evidence separately, then combine it.
- **Root cause:** A single lexical ranking can miss one or more hops in a compound question.
- **Suggested fix:** Split the question into approval, IT confirmation, and quote requirements; retrieve each subquestion and merge evidence before answering.

## Case study

**Question:** Bao lâu phải đổi mật khẩu một lần?

1. **Output đúng?** Cannot score with RAGAS here; the first context points to the obsolete 90-day policy.
2. **Context đúng?** Both v1 and v2 were among the retrieved contexts, so relevant evidence exists but is not ordered safely.
3. **Query rewrite OK?** The query contains no explicit “current policy” constraint, and the index has no active-version metadata filter.
4. **Fix:** Filter to active v2.0 policy, then use the cross-encoder to rank its 120-day chunk first.

If another hour were available, install the declared embedding, reranking, and evaluation dependencies, configure the API key, rerun baseline and production RAGAS, and replace this qualitative review with the measured bottom-five.
