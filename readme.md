# Phát Hiện Tin Giả Tiếng Việt Với PhoBERT Và Giải Thích Mô Hình (XAI)

Đồ án môn **Học máy và ứng dụng** - Khoa Công nghệ Thông tin, Trường Đại học Văn Lang.

* **Nhóm thực hiện:** Nhóm 13
* **Giảng viên hướng dẫn:** ThS. Nguyễn Thái Anh
* **Thành viên:**
  * Châu Gia Bảo (MSSV: 2474802010021)
  * Nguyễn Trần Khôi Vĩ (MSSV: 2474802010433)
  * Nguyễn Lê Trọng (MSSV: 2474802010406)

## 1. Giới thiệu bài toán
Hệ thống phân loại tin giả tiếng Việt trên tập dữ liệu mất cân bằng (4.835 mẫu sạch sau khi loại 262 bài trùng lặp: 83,12% Tin thật và 16,88% Tin giả), đối sánh 8 mô hình cơ sở (TF-IDF kết hợp 4 đặc trưng hành vi) với 4 mô hình dựa trên vector ngữ nghĩa PhoBERT bằng kiểm định chéo 5-fold StratifiedKFold, đồng thời tích hợp giải thích cục bộ theo cơ chế che từ `<mask>` (Occlusion XAI) trên giao diện Web tại cổng 8080.

## 2. Kết quả chính trên tập kiểm tra độc lập (1.209 mẫu)
* **Mô hình cơ sở tốt nhất (5-fold CV):** `SVM tuyến tính (TF-IDF + Hành vi)` đạt CV F1 `0,6321 ± 0,0109`; trên tập kiểm tra đạt Accuracy `87,34%`, Precision `61,23%`, Recall `68,14%`, F1-score `0,6450`.
* **Mô hình đề xuất tốt nhất (5-fold CV):** `PhoBERT + SVM nhân RBF` đạt CV F1 `0,6871 ± 0,0487`; trên tập kiểm tra đạt Accuracy `91,15%`, Precision `76,50%`, Recall `68,63%`, F1-score `0,7235` (tăng +7,85 điểm phần trăm F1 và giảm hơn một nửa số ca báo nhầm Tin thật từ 88 xuống 43).

## 3. Cấu trúc thư mục
* `b1_tien_xu_ly.py`: Làm sạch văn bản, tách từ PyVi, loại mẫu trùng, trích xuất 4 đặc trưng hành vi và lưu thống kê vào `data/`.
* `b2_mo_hinh_co_so.py`: Huấn luyện, kiểm định chéo 5-fold và đánh giá 8 mô hình cơ sở.
* `b3_phobert.py`: Trích xuất vector 768 chiều từ `vinai/phobert-base-v2` (Mean-Pooling), đánh giá 4 mô hình SOTA và thống kê phân tích lỗi `FP`/`FN`.
* `b4_giai_thich_demo.py`: Máy chủ Web Demo XAI tại cổng 8080 dùng trực tiếp mô hình thắng ở Bước 3 và che từ `<mask>`.
* `data/`: Lưu dữ liệu sạch, các bảng CSV kết quả (`thong_ke_tien_xu_ly.csv`, `thong_ke_hanh_vi.csv`, `ket_qua_co_so.csv`, `ket_qua_sota.csv`, `bang_so_sanh_tong_hop.csv`, `phan_tich_loi_sota.csv`, `ket_qua_xai_mau.csv`) và ảnh ma trận nhầm lẫn.

## 4. Hướng dẫn chạy
```bash
pip install -r requirements.txt
python3 b1_tien_xu_ly.py
python3 b2_mo_hinh_co_so.py
python3 b3_phobert.py
python3 b4_giai_thich_demo.py