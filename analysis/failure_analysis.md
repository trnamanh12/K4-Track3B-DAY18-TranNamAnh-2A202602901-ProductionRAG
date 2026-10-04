# Failure Analysis — Lab 18: Production RAG

**Họ và tên:** Trần Nam Anh  
**Khóa:** K4 - Track 3B

## Evaluation status

The pipeline processed all 20 questions. RAGAS did not run because `GEMINI_API_KEY` is not configured; the report records `evaluation_status: unavailable`. The zero placeholders are **not scores**, so there is no valid baseline/production comparison or RAGAS-ranked bottom-five. The local run used BM25 and lexical reranking because bge-m3 and the cross-encoder weights were not cached.

| Metric | Naive baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | N/A | N/A | N/A |
| Answer Relevancy | N/A | N/A | N/A |
| Context Precision | N/A | N/A | N/A |
| Context Recall | N/A | N/A | N/A |

## Five qualitative retrieval risks

These are manual examples from the local BM25 plus lexical fallback run, not measured RAGAS failures.

### 1. Senior leave and salary (multi-hop)

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 annual-leave days under v2024, plus a Senior salary range of 20–35 million VND/month.
- **Observed:** The first context is the correct 2024 annual-leave policy and contains the 18-day calculation. The top three contexts do not include the salary policy.
- **Error Tree:** Partial answer → annual-leave context is correct → salary evidence is missing → retrieve leave and salary subquestions separately and merge evidence.
- **Root cause:** One query ranks by overall lexical overlap; it does not guarantee coverage of both hops.
- **Suggested fix:** Decompose the query into leave entitlement and Senior salary retrieval, then answer only when both contexts are present.

### 2. Unpaid leave approval

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** CEO approval for 16–30 days.
- **Observed:** The first context is the annual-leave policy; the second context is the correct unpaid-leave policy with the 16–30 day approval rule.
- **Error Tree:** First answer context is off-topic → correct policy is in the candidate set → lexical ranking overweights shared “nghỉ phép” terms → rerank with the actual cross-encoder.
- **Root cause:** The cross-encoder model was not cached, so the fallback preserves retrieval scores and cannot reliably disambiguate policy scope.
- **Suggested fix:** Load `BAAI/bge-reranker-v2-m3` and verify the unpaid-leave source ranks first.

### 3. Laptop purchase (multi-hop)

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director approval, an IT configuration confirmation, and at least three quotes.
- **Observed:** The first context is the correct purchase policy and contains all three requirements. The other returned contexts are unrelated, which may lower context precision.
- **Error Tree:** Required evidence present → first context answers the question → extra contexts add noise → use the cross-encoder to remove unrelated parents.
- **Root cause:** The local fallback cannot judge semantic relevance as well as the intended cross-encoder.
- **Suggested fix:** Rerank candidates with the model and keep the smallest context set that covers approval, IT confirmation, and quotes.

### 4. Annual-leave policy versions

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 days under v2024; v2023's 12 days is superseded.
- **Observed:** The first context is v2024 with 15 days; v2023 also appears among the top three contexts.
- **Error Tree:** Current answer evidence ranks first → older policy remains nearby → answer may mix versions → use source status and version-aware reranking.
- **Root cause:** Version boosting prefers the newest source but does not remove older versions from the candidate list.
- **Suggested fix:** Keep the current policy first and include the old version only when the question requests historical comparison.

### 5. Password policy versions

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Every 120 days under v2.0; v1.0's 90 days is superseded.
- **Observed:** BM25 ranks `mat_khau_v2.md` first after version boosting, but both versions can remain in candidates.
- **Error Tree:** Current source ranks first → stale policy can still enter the context set → answer risks mixing intervals → retrieve active version first and inspect the final context set.
- **Root cause:** Filename version ranking is a heuristic and does not encode active/superseded status as a filter.
- **Suggested fix:** Extract policy status metadata and add a current-version filter; retain v1 only for historical questions.

## Case study

**Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?

1. **Output đúng?** The local fallback answer is not a generated answer; RAGAS cannot score this run.
2. **Context đúng?** The annual-leave context is correct and gives 18 days, but the salary range is absent from the top three.
3. **Query rewrite OK?** A single compound query did not retrieve both policy sources reliably.
4. **Fix:** Retrieve the annual-leave and salary subquestions separately, check that both sources are present, then synthesize one answer.

After configuring `GEMINI_API_KEY` and pre-downloading the embedding/reranker models, rerun baseline and production RAGAS and replace this qualitative review with the measured bottom-five.
