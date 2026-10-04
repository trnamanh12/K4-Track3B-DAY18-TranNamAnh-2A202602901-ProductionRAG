# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Trần Nam Anh  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Khi có model `all-MiniLM-L6-v2`, cosine similarity giữa hai câu liền kề quyết định ranh giới chunk. Môi trường này thiếu `sentence_transformers`, nên pipeline dùng fallback theo đoạn văn; chưa thể đo hiệu quả semantic hoặc so sánh threshold 0.85. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Pipeline tạo 100 child chunks từ 26 tài liệu; baseline paragraph chunking tạo 57 chunks. Child giữ `parent_id`, giúp lấy đoạn nhỏ để tìm kiếm và giữ liên kết về ngữ cảnh lớn hơn. |
| BM25 + Dense fusion | M2 | `BM25Search.search()`, `DenseSearch.search()`, `reciprocal_rank_fusion()` | RRF cộng `1 / (k + rank + 1)` theo thứ hạng, nên có thể gộp lexical và dense mà không cần so sánh hai thang điểm. Lần chạy này chỉ có BM25 vì thiếu Qdrant và sentence-transformers; nhánh dense chưa được đo. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Thiết kế rerank top candidates xuống top 3. Model không tải được (`No module named 'sentence_transformers'`), nên local run dùng token-overlap fallback; ví dụ câu hỏi nghỉ kết hôn bị chunk đào tạo xếp trên chunk nghỉ đặc biệt. Chưa có số đo latency/precision của cross-encoder thật. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Có đủ bốn metric và Diagnostic Tree trong code. Runtime không có `datasets` và không có API key, nên report ghi `evaluation_status: unavailable`; các số 0 không đại diện chất lượng. |
| Contextual enrichment | M5 | `_enrich_single_call()`, `contextual_prepend()` | Combined mode dùng một lệnh gọi cho summary, câu hỏi giả thuyết, context và metadata. Không có API key nên fallback thêm tên tài liệu làm context prefix và giữ nguyên nội dung chunk; đã quan sát `enriched_text` khác `original_text`. |

## Phần 2: Khó khăn & cách giải quyết

- **Lỗi kỹ thuật gặp phải (exact error):** `bash: python: command not found`.
  - **Cách debug:** Kiểm tra lệnh có sẵn và chạy bằng `python3`; các lệnh kiểm tra và `main.py` đều chạy được.
- **Lỗi phụ thuộc (exact errors):** `ModuleNotFoundError: No module named 'pypdf'`, `No module named 'sentence_transformers'`, và `No module named 'datasets'`.
  - **Cách debug:** Cài `pypdf` từ `requirements.txt` để đọc PDF có text. Hai dependency còn lại và `OPENAI_API_KEY` chưa có trong runtime; code giữ fallback cho pipeline, đồng thời report đánh dấu RAGAS là unavailable thay vì giả định các số 0 là kết quả đánh giá.
- **Nguyên nhân gốc rễ & kiến thức còn thiếu:** Chưa thiết lập đầy đủ môi trường production RAG (Qdrant, embedding/reranker models, RAGAS và OpenAI API). BM25 cũng dễ xếp nhầm tài liệu cũ khi nhiều policy dùng cùng từ khóa.
- **Bổ sung kiến thức:** Cài dependency theo `requirements.txt`, khởi động Qdrant, cấu hình API key, rồi chạy lại baseline và production trên cùng test set. Thêm metadata trạng thái/phiên bản chính sách và đo từng metric trước khi chọn thay đổi.

## Phần 3: Action Plan cho project cá nhân

### Project: HR Policy RAG (corpus chính sách nhân sự của lab)

#### Hiện trạng

- **Pipeline hiện tại:** Markdown/PDF text → hierarchical chunks → contextual enrichment → BM25 + (tùy môi trường) dense retrieval → reranking → câu trả lời theo context → RAGAS.
- **Known issues:** môi trường này chưa có dense embeddings, cross-encoder, RAGAS runtime/API key; tài liệu cũ và mới có thể cùng được retrieve; câu hỏi nhiều bước cần evidence từ nhiều policy.

#### Kế hoạch áp dụng

1. **Chunking strategy:** Dùng hierarchical chunking cho policy dài; giữ section header và bổ sung category/version vào metadata.
2. **Search retrieval:** Dùng BM25 + bge-m3 + RRF để bắt cả từ khóa chính xác lẫn cách diễn đạt tương đương; lọc policy superseded khi câu hỏi yêu cầu quy định hiện hành.
3. **Reranking:** Dùng `BAAI/bge-reranker-v2-m3` để chọn tối đa ba context liên quan; so sánh với BM25 trước khi triển khai.
4. **Evaluation:** Chạy bốn RAGAS metrics trên cùng 20 câu hỏi sau khi cấu hình OpenAI API; theo dõi thêm retrieval hit rate cho câu version-sensitive và multi-hop.
5. **Enrichment:** Bật combined enrichment một call/chunk cho nguồn đã kiểm duyệt; dùng contextual prefix và metadata, kiểm tra chi phí và chất lượng trước khi index toàn bộ.

#### Timeline

- **Tuần 1:** Cài Qdrant và model dependencies; tạo baseline dense; ghi lại lỗi và thời gian từng bước.
- **Tuần 2:** Thêm metadata phiên bản/trạng thái, reranker; chạy lại test set và phân loại lỗi retrieval theo Error Tree.
- **Tuần 3:** So sánh bốn RAGAS metrics, chọn thay đổi có cải thiện đo được, rồi cập nhật failure analysis.
