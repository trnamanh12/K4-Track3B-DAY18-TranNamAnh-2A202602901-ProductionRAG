# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Trần Nam Anh  
**MSSV:** 2A202602901  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|:-------------:|:----------:|:---:|
| Faithfulness | 0.8704 | 0.9250 | +0.0546 |
| Answer Relevancy | 0.7211 | 0.8238 | +0.1027 |
| Context Precision | 0.8500 | 0.7750 | -0.0750 |
| Context Recall | 0.8250 | 0.9500 | +0.1250 |

**Nhận xét nhanh:**
- **Context Recall tăng mạnh (+12.5%):** Nhờ Hybrid Search (BM25 + Dense) kết hợp RRF, khắc phục tình trạng Dense-only bỏ sót keyword viết tắt hoặc thuật ngữ riêng.
- **Faithfulness (0.9250) & Relevancy (0.8238) tăng rõ rệt:** Reranker gạn lọc đúng top 3 chunk trọng tâm, LLM ít bị loãng ngữ cảnh nên trả lời sát và ít bịa hơn.
- **Context Precision giảm nhẹ (0.85 → 0.775):** Do dùng Hierarchical chunking trả về Parent chunk (~2048 ký tự) để lấy đủ context cho LLM, vô tình kéo thêm một số câu không liên quan trực tiếp vào cửa sổ ngữ cảnh.

---

## Bottom-5 Failures

### #1: Câu hỏi đa ý định (Multi-hop Query)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 18 ngày phép (15 ngày + 3 ngày thâm niên). Lương Senior (P3-P4): 20–35 triệu VNĐ/tháng.
- **Got:** Trả lời đúng 18 ngày phép theo bản 2024, nhưng báo ngữ cảnh không có thông tin về thang lương.
- **Worst metric:** Context Precision (0.0000) / Context Recall
- **Error Tree:** Output thiếu ý lương → Context thiếu file bảng lương (chỉ có 3 chunk về nghỉ phép) → Query ghép 2 chủ đề khác nhau (phép + lương) → Fix ở tầng Query Planning.
- **Root cause:** Truy vấn chứa 2 ý định độc lập ("phép năm" và "mức lương"). Ngữ nghĩa về nghỉ phép chiếm ưu thế nên cả 3 kết quả trả về đều rơi vào chính sách nghỉ phép, file `bang_luong_2024.md` bị trôi khỏi top-k.
- **Suggested fix:** Áp dụng Query Decomposition: tách thành 2 câu hỏi con ("phép năm theo thâm niên" và "khung lương Senior"), truy vấn song song rồi gộp context.

### #2: Nhầm lẫn giữa các phiên bản chính sách
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Chính sách v2024: từ 3 năm trở lên, mỗi 3 năm cộng 1 ngày. Bản cũ v2023 yêu cầu 5 năm.
- **Got:** Trả lời đúng quy định v2024 (từ 3 năm), nhưng không nhắc đến bản cũ v2023.
- **Worst metric:** Context Precision (0.0000)
- **Error Tree:** Output chỉ nêu bản mới → Context bị kéo cả file v2024 lẫn v2023 → RAGAS phạt vì chunk v2023 không phục vụ câu trả lời → Fix ở bước lọc Metadata.
- **Root cause:** Hai file `nghi_phep_nam_v2024.md` và `nghi_phep_nam_v2023.md` trùng nhiều keyword. Reranker đưa v2024 lên đầu nhưng v2023 vẫn nằm ở vị trí số 2 trong top 3.
- **Suggested fix:** Đưa trường `is_active: true` vào metadata khi ingest; mặc định chỉ retrieve các bản đang hiệu lực trừ khi query có từ khóa "trước đây", "lịch sử".

### #3: Nhiễu từ khóa "Lương" kéo sai tài liệu
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Thuộc dữ liệu Bí mật (cấp 3), phải mã hóa và phân quyền theo need-to-know.
- **Got:** Thông tin lương thuộc cấp độ Bí mật.
- **Worst metric:** Context Precision (0.0000)
- **Error Tree:** Output đúng nhãn → Context chứa cả file `bang_luong_2024.md` (chỉ ghi số tiền, không ghi cấp độ bảo mật) → Tồn tại chunk thừa trong top 3 → Fix ở bước Reranking.
- **Root cause:** BM25 bắt từ khóa "lương" quá mạnh khiến bảng số liệu lương lọt vào candidate, dù nội dung không liên quan đến an toàn dữ liệu.
- **Suggested fix:** Đặt threshold điểm số tối thiểu cho Cross-Encoder để loại bỏ các chunk có điểm thấp hơn rõ rệt so với top 1.

### #4: Tài liệu cũ có độ khớp từ khóa cao
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Chính sách v2.0 hiện hành: mỗi 120 ngày. Bản cũ là 90 ngày (đã thay thế).
- **Got:** Mật khẩu phải được thay đổi mỗi 120 ngày.
- **Worst metric:** Context Precision (0.0000)
- **Error Tree:** Output đúng 120 ngày → Context vẫn lọt file `mat_khau_v1.md` và quy trình sự cố → Tỷ lệ chunk hữu ích thấp → Fix ở bước tiền lọc.
- **Root cause:** Bản cũ `mat_khau_v1.md` dùng câu chữ rất khớp câu hỏi ("Chu kỳ thay đổi mỗi 90 ngày"), BM25 chấm điểm rất cao.
- **Suggested fix:** Khai thác metadata `status: ĐÃ THAY THẾ` từ bước M5 Enrichment để trừ điểm hoặc loại hẳn tài liệu cũ khỏi retrieval.

### #5: LLM suy luận số học bị RAGAS chấm lệch
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nghỉ trước hạn cam kết 1 năm nên hoàn trả 100% chi phí, tương đương 25.000.000 VNĐ.
- **Got:** Hoàn trả 100% chi phí (tương đương 25.000.000 VNĐ) do nghỉ trước hạn cam kết 1 năm.
- **Worst metric:** Faithfulness (0.3333)
- **Error Tree:** Output hoàn toàn đúng thực tế → Context chỉ ghi "hoàn trả 100% chi phí", không có con số "25 triệu" (số này nằm ở câu hỏi) → Evaluator coi con số 25 triệu là hallucination ngoài context → Fix ở Prompt Generation.
- **Root cause:** RAGAS bóc câu trả lời thành từng mệnh đề độc lập để đối chiếu với context. Mệnh đề có con số cụ thể bị gán cờ vì context gốc không ghi con số đó.
- **Suggested fix:** Chỉnh system prompt để LLM nói rõ nguồn: "Theo chính sách, phải hoàn trả 100%. Áp dụng vào mức 25 triệu nêu trong câu hỏi, số tiền cần trả là 25 triệu VNĐ".

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
> *"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

**Error Tree walkthrough:**
1. **Output đúng?** → Sai một nửa: tính đúng 18 ngày phép, nhưng thiếu hoàn toàn thông tin mức lương Senior.
2. **Context đúng?** → Sai: Top 3 context chỉ toàn văn bản về nghỉ phép (`v2024`, `v2023`, `không lương`), không có văn bản bảng lương.
3. **Query rewrite OK?** → Chưa OK: Câu hỏi gốc là câu ghép 2 ý định, đưa nguyên văn vào vector search khiến nhánh "nghỉ phép" áp đảo hoàn toàn nhánh "lương".
4. **Fix ở bước:** Tầng Query Pre-processing (Query Decomposition).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Viết 1 router nhỏ bằng LLM trước khi search: nếu câu hỏi chứa nhiều mệnh đề độc lập, tách thành 2 truy vấn con rồi merge context lại.
- Thêm metadata filter theo trạng thái hiệu lực (`status == 'active'`) để loại bỏ hoàn toàn các văn bản v2022/v2023 khi không được yêu cầu.
