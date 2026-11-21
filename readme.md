# Phát Hiện Tin Giả Tiếng Việt Với PhoBERT Và Giải Thích Mô Hình (XAI)

Đồ án môn **Học máy và ứng dụng** - Khoa Công nghệ Thông tin, Trường Đại học Văn Lang.

- **Nhóm thực hiện:** Nhóm 12
- **Giảng viên hướng dẫn:** ThS. Nguyễn Thái Anh
- **Thành viên:**
  - Châu Gia Bảo (MSSV: 2474802010021)
  - Nguyễn Trần Khôi Vĩ (MSSV: 2474802010433)
  - Nguyễn Lê Trọng (MSSV: 2474802010406)

## 1. Giới thiệu bài toán
Hệ thống phát hiện tin giả tiếng Việt trên tập dữ liệu 5.097 bài đăng mạng xã hội có độ lệch lớp cao (4.166 Tin thật chiếm 81,73% và 931 Tin giả chiếm 18,27%). Dự án triển khai quy trình 4 bước:
1. Tiền xử lý văn bản (chuẩn hóa NFC, tách từ tiếng Việt bằng PyVi) và trích xuất 4 đặc trưng hành vi từ văn bản gốc (độ dài, số dấu chấm than, số dấu hỏi, tỷ lệ chữ in hoa).
2. Huấn luyện và đối sánh 8 cấu hình học máy cơ sở dựa trên TF-IDF (kết hợp `MaxAbsScaler` và `class_weight='balanced'`).
3. Trích xuất vector ngữ nghĩa 768 chiều bằng `vinai/phobert-base-v2` (Mean-Pooling theo Attention Mask) và đánh giá 4 cấu hình phân loại.
4. Tích hợp thuật toán giải thích từ khóa (XAI) theo cơ chế che từ (Occlusion) trên ứng dụng Web Demo tại cổng 8080.

## 2. Kết quả thực nghiệm (tập kiểm tra 1.275 mẫu)
Dữ liệu được chia phân tầng 75% huấn luyện (3.822 mẫu) và 25% kiểm tra (1.275 mẫu) với `stratify=y`, `random_state=17`.

| Cấu hình mô hình | Accuracy | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| Baseline tốt nhất: Linear SVM (TF-IDF + 4 Hành vi) | 0,8824 | 0,6615 | 0,7296 | 0,6939 |
| PhoBERT + Logistic | 0,8533 | 0,5728 | 0,7768 | 0,6594 |
| PhoBERT + TF-IDF (Lai ghép) | 0,8706 | 0,6223 | 0,7425 | 0,6771 |
| PhoBERT + Mạng MLP | 0,9129 | 0,7798 | 0,7296 | 0,7539 |
| **PhoBERT + SVM nhân RBF (Đề xuất)** | **0,9106** | **0,7490** | **0,7682** | **0,7585** |

## 3. Hướng dẫn cài đặt và chạy Web Demo
```bash
git clone [https://github.com/BaoLilo/vietnamese-fake-news-phobert-xai.git](https://github.com/BaoLilo/vietnamese-fake-news-phobert-xai.git)
cd vietnamese-fake-news-phobert-xai
pip install -r requirements.txt
python3 <ten_file_chay_web>.py