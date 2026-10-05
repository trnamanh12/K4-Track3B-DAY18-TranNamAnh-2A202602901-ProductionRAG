# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Trần Nam Anh  
**MSSV:** 2A202602901  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 05/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|---|:---:|---|---|
| **Semantic chunking** | M1 | `chunk_semantic()` | Dùng `all-MiniLM-L6-v2` tính cosine similarity giữa các câu với ngưỡng 0.85. Khi chạy thử, số lượng chunk tăng đáng kể so với basic chunking vì nhiều câu chuyển ý bị ngắt riêng. Cách này giữ trọn ý câu nhưng độ dài chunk không đều; với văn bản quy định dài, chiến lược Hierarchical ổn định hơn. |
| **Hierarchical chunking** | M1 | `chunk_hierarchical()` | Cắt Parent 2048 ký tự, Child 256 ký tự có gắn `parent_id`. 26 tài liệu tạo ra đúng 100 child chunk. Khi search, vector matching trên child cho độ khớp cao, nhưng sau đó đổi sang parent để gửi cho LLM. Nhờ vậy Context Recall đạt **0.9500** (tăng 12.5% so với baseline). |
| **BM25 + Dense fusion** | M2 | `reciprocal_rank_fusion()`, `BM25Search`, `DenseSearch` | Tiền xử lý từ tiếng Việt bằng `underthesea`, thay dấu `_` thành khoảng trắng để BM25 không bị lệch token. Dense search dùng `BAAI/bge-m3` qua Qdrant. Công thức RRF ($k=60$) cân bằng tốt giữa từ khóa chính xác (mã quy định, ngày tháng) và câu hỏi diễn đạt tự nhiên. |
| **Cross-encoder reranking** | M3 | `CrossEncoderReranker.rerank()` | Dùng `BAAI/bge-reranker-v2-m3` chấm điểm tương tác sâu giữa query và top 20 candidate, lọc lấy top 3. Latency rerank tốn khoảng 5.6s/lượt nhưng cải thiện rõ Answer Relevancy (lên **0.8238**) do loại bỏ các chunk chỉ vô tình trùng lặp từ khóa. |
| **RAGAS 4 metrics** | M4 | `evaluate_ragas()` | Đánh giá 4 chỉ số trên 20 câu test: Faithfulness (**0.9250**), Relevancy (**0.8238**), Precision (**0.7750**), Recall (**0.9500**). Tự động phân tích Bottom-5 để tìm lỗi theo cây chẩn đoán (Diagnostic Tree). |
| **Contextual embeddings / Enrichment** | M5 | `_enrich_single_call()`, `enrich_chunks()` | Dùng 1 prompt trích xuất cùng lúc 4 thông tin (summary, câu hỏi HyQA, context prefix, metadata), tiết kiệm 75% số lượt gọi API. Thêm cache đĩa `enrichment_cache.json` để không phải gọi lại API khi re-run pipeline. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Trong quá trình thực hiện bài lab, tôi gặp 4 lỗi kỹ thuật thực tế sau:

1. **Lỗi `evaluate() got an unexpected keyword argument 'max_workers'` trong Ragas:**
   - **Exact error:** `TypeError: evaluate() got an unexpected keyword argument 'max_workers'`
   - **Nguyên nhân & Debug:** Ragas bản `0.1.22` không nhận trực tiếp tham số `max_workers` ở hàm `evaluate()`. Cần đóng gói cấu hình thông qua `RunConfig`.
   - **Cách xử lý:** Đổi thành `run_config=RunConfig(max_workers=1, max_retries=10, timeout=120)`.

2. **Lỗi không hỗ trợ multiple candidates trên endpoint Gemini:**
   - **Exact error:** `BadRequestError: Error code: 400 - Multiple candidates is not enabled for this model`
   - **Nguyên nhân & Debug:** Metric `answer_relevancy` mặc định yêu cầu LLM sinh $n=3$ câu hỏi giả định (`n=self.strictness`), trong khi endpoint OpenAI-compatible của Gemini chỉ cho phép $n=1$.
   - **Cách xử lý:** Gán `answer_relevancy.strictness = 1` trước khi chạy eval. Metric vẫn tính toán chuẩn xác và không bị crash.

3. **Lỗi tràn bộ nhớ GPU khi nạp nhiều model PyTorch (CUDA OOM):**
   - **Exact error:** `CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 3.68 GiB of which 3.94 MiB is free.`
   - **Nguyên nhân & Debug:** VRAM card đồ họa máy chỉ có 3.68 GB. Việc nạp cùng lúc cả `bge-m3` trong DenseSearch và `HuggingFaceEmbeddings` trong Ragas khiến bộ nhớ GPU bị cạn kiệt.
   - **Cách xử lý:** Cấu hình tham số `device="cpu"` trong `SentenceTransformer` và `model_kwargs={"device": "cpu"}` trong `HuggingFaceEmbeddings`. Máy có 12 nhân CPU và 13 GB RAM nên chạy hoàn toàn mượt mà, không còn lo tràn VRAM.

4. **Lỗi chạm ngưỡng Rate Limit của tài khoản Free-Tier:**
   - **Exact error:** `429 RESOURCE_EXHAUSTED: Quota exceeded for metric ... limit: 15/minute, 500/day`
   - **Nguyên nhân & Debug:** Khi chạy liên tục 20 câu hỏi và 100 chunk enrichment, số request gửi lên vượt quá giới hạn 15 RPM của model cụ thể.
   - **Cách xử lý:**
     - Đặt độ trễ nhỏ `time.sleep(1.5)` giữa các query trong pipeline.
     - Dùng cơ chế `with_fallbacks` của LangChain để tự động chuyển model dự phòng (`gemini-3.1-flash-lite`, `gemini-3.5-flash`, `gemini-flash-latest`) khi model chính hết quota ngày.
     - Lưu kết quả enrichment vào file JSON cục bộ để tái sử dụng ngay lập tức ở các lần chạy sau.

---

## Phần 3: Action Plan cho Project Cá Nhân

### Project: Trợ lý Tra cứu Quy chế & Quy trình Vận hành Nội bộ (Internal Policy Bot)

#### 1. Hiện trạng
- **Kiến trúc cũ:** Naive RAG chia đoạn theo độ dài cố định 500 ký tự; tìm kiếm Dense-only bằng cosine similarity; chưa có bước rerank.
- **Vấn đề gặp phải:**
  - Tìm kiếm kém với các từ viết tắt chuyên ngành (PVI, VPN, MFA, BCTC).
  - Thiếu khả năng nhận biết văn bản mới thay thế văn bản cũ, thỉnh thoảng trích dẫn nhầm điều khoản đã hết hiệu lực.
  - Câu hỏi gồm nhiều vế thường chỉ trả lời được một phần do ngữ cảnh bị chiếm bởi một chủ đề.

#### 2. Kế hoạch cải tiến
1. **Chunking Strategy:** Áp dụng Hierarchical chunking (Parent 2048 - Child 256). Riêng các bảng biểu (bảng lương, phụ cấp) sẽ dùng Structure-aware chunking để tránh ngắt giữa dòng bảng.
2. **Search Retrieval:** Kết hợp BM25 (có phân đoạn từ bằng `underthesea`) và Dense embedding (`bge-m3`) thông qua thuật toán RRF. Thêm bộ lọc metadata `status: active` để mặc định bỏ qua các tài liệu đã hết hiệu lực.
3. **Reranking:** Dùng `bge-reranker-v2-m3` lọc từ top 20 candidate xuống top 3 gửi cho LLM. Thử nghiệm thêm `Flashrank` trên môi trường CPU nếu cần tối ưu độ trễ xuống dưới 100ms.
4. **Enrichment:** Chạy combined enrichment lúc ingest dữ liệu vào hệ thống (tạo tóm tắt ngắn, câu hỏi HyQA và ngữ cảnh tài liệu) để tăng khả năng bắt trúng văn bản của BM25.
5. **Evaluation:** Đưa bộ 4 metrics RAGAS vào pipeline kiểm thử tự động, đặt ngưỡng Faithfulness ≥ 0.90 và Answer Relevancy ≥ 0.80 trước khi release phiên bản kho tri thức mới.

#### 3. Timeline triển khai (3 tuần)
- **Tuần 1:** Chuẩn hóa dữ liệu văn bản, triển khai Hierarchical Chunking và dựng Qdrant collection.
- **Tuần 2:** Tích hợp Hybrid Search (BM25 + Dense + RRF) và Cross-Encoder Reranker; đo đạc độ trễ từng bước.
- **Tuần 3:** Xây dựng tập test 30 câu hỏi thực tế, chạy benchmark tự động bằng RAGAS và tinh chỉnh prompt system dựa trên failure analysis.
