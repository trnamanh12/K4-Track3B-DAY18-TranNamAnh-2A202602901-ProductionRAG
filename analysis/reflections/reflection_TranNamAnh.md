# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Trần Nam Anh  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Với `all-MiniLM-L6-v2` và threshold 0.85, `compare_strategies()` tạo 208 chunks từ các tài liệu nối lại, so với 51 basic chunks. Đây là nhiều hơn vì biên similarity tách nhiều nhóm câu; hierarchical strategy phù hợp hơn cho pipeline này. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | So sánh trên toàn corpus cho 87 child/11 parent; pipeline xử lý từng tài liệu nên tạo 100 child chunks từ 26 tài liệu. Child ≤256 ký tự giữ `parent_id`; sau retrieval, pipeline đổi child thành parent để trả đủ ngữ cảnh. |
| BM25 + Dense fusion | M2 | `BM25Search.search()`, `DenseSearch.search()`, `reciprocal_rank_fusion()` | RRF cộng `1 / (k + rank + 1)` theo thứ hạng. Với underthesea và ưu tiên version mới nhất, BM25 xếp `mat_khau_v2.md` trên v1 cho câu hỏi chu kỳ đổi mật khẩu. Dense branch chưa index được vì model bge-m3 chưa có trong cache và Hugging Face không tải được trong run này. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Pipeline giữ tối đa 3 context. Model `BAAI/bge-reranker-v2-m3` chưa có trong cache; local run dùng lexical fallback và giữ thứ tự retrieval score, nên chưa đo precision hoặc latency của cross-encoder thật. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Code truyền Gemini chat model và Gemini embeddings vào bốn metric. Chưa có `GEMINI_API_KEY`, nên report ghi `evaluation_status: unavailable`; placeholder 0 không phải điểm. |
| Contextual enrichment | M5 | `_enrich_single_call()`, `contextual_prepend()` | Combined mode dùng một call/chunk để lấy summary, câu hỏi giả thuyết, context và metadata; enriched text ghép các trường này với nội dung gốc để cải thiện lexical retrieval. Run này không có API key nên dùng fallback cục bộ. |

## Phần 2: Khó khăn & cách giải quyết

- **Lỗi kỹ thuật gặp phải (exact error):** `bash: python: command not found`.
  - **Cách debug:** Kiểm tra lệnh có sẵn và chạy bằng `python3`; các lệnh kiểm tra và `main.py` đều chạy được.
- **Lỗi môi trường (exact messages):** `bash: python: command not found`, `ModuleNotFoundError: No module named 'pypdf'`, và `Set GEMINI_API_KEY in .env to enable RAGAS evaluation`.
  - **Cách debug:** Dùng Python 3.12 trong uv environment, cài dependency từ `requirements.txt`, và cài pypdf để load PDF có text. Xác minh các provider import thành công; RAGAS không gọi được API vì chưa cấu hình key.
- **Lỗi tải model:** `We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.`
  - **Cách debug:** Bật offline mode để xác nhận các fallback và tests chạy ổn. Dense embedding và cross-encoder chưa tải được nên report ghi rõ dense/reranker đang tắt hoặc dùng fallback.
- **Nguyên nhân gốc rễ & kiến thức còn thiếu:** Runtime ban đầu thiếu dependencies; sau khi cài, vẫn thiếu Gemini API key và model weights. BM25 có thể nhầm scope/version khi các policy dùng chung từ khóa.
- **Bổ sung kiến thức:** Thêm `GEMINI_API_KEY` vào file `.env`, pre-download bge-m3 và cross-encoder theo README, chạy lại baseline và production trên cùng test set. So sánh bốn metric và thay đổi retrieval dựa trên bottom-five thực tế.

## Phần 3: Action Plan cho project cá nhân

### Project: HR Policy RAG (corpus chính sách nhân sự của lab)

#### Hiện trạng

- **Pipeline hiện tại:** Markdown/PDF text → hierarchical chunks → contextual enrichment → BM25 + (tùy môi trường) dense retrieval → reranking → câu trả lời theo context → RAGAS.
- **Known issues:** Gemini key và bge model weights chưa được cấu hình/tải trong lần chạy; một số câu hỏi policy version-sensitive và multi-hop cần nhiều evidence.

#### Kế hoạch áp dụng

1. **Chunking strategy:** Dùng hierarchical chunking cho policy dài; giữ section header và bổ sung category/version vào metadata.
2. **Search retrieval:** Dùng BM25 + bge-m3 + RRF để bắt cả từ khóa chính xác lẫn cách diễn đạt tương đương; lọc policy superseded khi câu hỏi yêu cầu quy định hiện hành.
3. **Reranking:** Dùng `BAAI/bge-reranker-v2-m3` để chọn tối đa ba context liên quan; so sánh với BM25 trước khi triển khai.
4. **Evaluation:** Chạy bốn RAGAS metrics trên cùng 20 câu hỏi sau khi cấu hình Gemini API; theo dõi thêm retrieval hit rate cho câu version-sensitive và multi-hop.
5. **Enrichment:** Bật combined enrichment một call/chunk cho nguồn đã kiểm duyệt; dùng contextual prefix và metadata, kiểm tra chi phí và chất lượng trước khi index toàn bộ.

#### Timeline

- **Tuần 1:** Cài Qdrant và model dependencies; tạo baseline dense; ghi lại lỗi và thời gian từng bước.
- **Tuần 2:** Thêm metadata phiên bản/trạng thái, reranker; chạy lại test set và phân loại lỗi retrieval theo Error Tree.
- **Tuần 3:** So sánh bốn RAGAS metrics, chọn thay đổi có cải thiện đo được, rồi cập nhật failure analysis.
