# Phát hiện tin giả tiếng Việt bằng PhoBERT và SVM

Đồ án môn Học máy và ứng dụng - Khoa CNTT, Trường ĐH Văn Lang.

- Nhóm: 13
- GVHD: ThS. Nguyễn Thái Anh
- Thành viên:
  - Châu Gia Bảo 
  - Nguyễn Trần Khôi Vĩ 
  - Nguyễn Lê Trọng 

## Tổng quan

Repo chứa code thực nghiệm phân loại bài đăng mạng xã hội tiếng Việt (Tin thật / Tin giả) trên 4.835 mẫu sạch sau khi loại 262 mẫu trùng (83,12% Tin thật, 16,88% Tin giả).

Nhóm đối sánh 8 mô hình cơ sở (TF-IDF + 4 đặc trưng hành vi) với 4 mô hình dùng vector 768 chiều từ `vinai/phobert-base-v2`, chọn mô hình qua 5-fold StratifiedKFold. Phần demo chạy web cục bộ tại cổng 8080 có tô màu từ khóa ảnh hưởng bằng cách che `<mask>`.

## Kết quả trên tập test (1.209 mẫu)

| Mô hình | 5-fold CV F1 | Accuracy | Precision | Recall | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Linear SVM (TF-IDF + Hành vi) | 0,6321 ± 0,0109 | 87,34% | 61,23% | 68,14% | 0,6450 |
| PhoBERT + SVM RBF | 0,6871 ± 0,0487 | 91,15% | 76,50% | 68,63% | 0,7235 |

Mô hình `PhoBERT + SVM RBF` tăng F1 thêm 7,85 điểm phần trăm so với baseline và giảm số ca báo nhầm tin thật (FP) từ 88 xuống 43.

## Các file chính

 - `b1_tien_xu_ly.py`: làm sạch văn bản, tách từ bằng PyVi, lọc trùng và tính 4 đặc trưng hành vi.
 - `b2_mo_hinh_co_so.py`: chạy 5-fold CV và đánh giá 8 mô hình cơ sở.
 - `b3_phobert.py`: trích xuất vector PhoBERT (Mean-Pooling), đánh giá 4 mô hình và thống kê lỗi FP/FN.
 - `b4_giai_thich_demo.py`: chạy web demo XAI ở cổng 8080 dùng trực tiếp mô hình thắng ở bước 3.
 - `data/`: lưu dữ liệu sạch, các file CSV kết quả và ảnh ma trận nhầm lẫn.

## Cách chạy

```bash
pip install -r requirements.txt
python3 b1_tien_xu_ly.py
python3 b2_mo_hinh_co_so.py
python3 b3_phobert.py
python3 b4_giai_thich_demo.py
```

Sau khi chạy bước 4, mở trình duyệt tại `http://localhost:8080`.
