# Adult Census Income Classification

## Thông tin môn học

- Môn học: Học máy
- Mã môn học: <điền>
- Học kỳ: I
- Năm học: 2026–2027
- Giảng viên hướng dẫn: <điền>
- Nhóm: <điền>

## Thành viên

| Họ tên | MSSV | Công việc |
|---|---|---|
| Lê Minh Trí | 2353229 | Xây dựng, huấn luyện, tối ưu và đánh giá mô hình Deep Learning MLP |
| Trần Nguyễn Nhất Thông | 2353149 | Tiền xử lý, Feature Engineering, xây dựng và đánh giá các mô hình ML truyền thống |
| Nguyễn Minh Khôi | 2352618 | Thu thập dữ liệu, EDA, trực quan hóa, phân tích kết quả và tổng hợp báo cáo |


## Mục tiêu

Xây dựng pipeline học máy truyền thống để phân loại thu nhập
<=50K hoặc >50K trên dữ liệu UCI Adult.
So sánh các cấu hình tiền xử lý và mô hình; triển khai thêm
pipeline học sâu để đối chiếu kết quả.

## Dữ liệu

- Nguồn công khai: https://archive.ics.uci.edu/dataset/2/adult
- Train: 32.561 mẫu; test: 16.281 mẫu; 14 thuộc tính đầu vào.
- Nhãn: <=50K → 0; >50K → 1.
- Notebook tự tải adult.data và adult.test.
- Không cần mount Google Drive hoặc tải dữ liệu thủ công.

## Phương pháp

### Phân tích khám phá dữ liệu (EDA)

Thực hiện EDA trên tập huấn luyện: thống kê phân phối nhãn thu nhập, kiểu dữ liệu, giá trị thiếu, bản ghi trùng lặp và phân phối các thuộc tính số, thuộc tính phân loại. Kết quả được sử dụng để mô tả đặc điểm dữ liệu và tình trạng mất cân bằng giữa hai lớp thu nhập.

### Tiền xử lý và biểu diễn đặc trưng

- **Imputation:** điền giá trị thiếu của thuộc tính số bằng trung vị; thử nghiệm điền thuộc tính phân loại bằng giá trị xuất hiện nhiều nhất hoặc nhóm `Unknown`.
- **Encoding:** sử dụng One-Hot Encoding cho thuộc tính phân loại, hỗ trợ xử lý các giá trị chưa xuất hiện khi huấn luyện. Nhãn `<=50K` được mã hóa thành `0`, nhãn `>50K` thành `1`.
- **Scaling:** so sánh StandardScaler với MinMaxScaler có `feature_range=(0, 1)` và `feature_range=(-1, 1)`.
- **PCA:** thử nghiệm giữ lại 90% và 95% phương sai trong pipeline Logistic Regression để đánh giá ảnh hưởng của giảm chiều.

Các bước tiền xử lý được fit trên phần huấn luyện của từng fold nhằm tránh rò rỉ dữ liệu.

### Mô hình học máy truyền thống

Sử dụng DummyClassifier làm mốc so sánh; huấn luyện và đánh giá Logistic Regression, Logistic Regression với trọng số lớp cân bằng, LinearSVC và Random Forest. Random Forest được tinh chỉnh bằng GridSearchCV với 12 cấu hình kết hợp phương pháp điền dữ liệu phân loại, độ sâu cây và số mẫu tối thiểu tại nút lá.

### Phần mở rộng học sâu

Thử nghiệm 16 cấu hình mạng MLP và mạng residual, kết hợp các cách biểu diễn thuộc tính số: chuẩn hóa, biến đổi log, biến đổi quantile và mã hóa tuyến tính từng đoạn. Các cấu hình có sử dụng dropout, weight decay và early stopping. Hai cấu hình mạng đơn tốt nhất theo CV được thử thêm ensemble ba mạng với các seed khởi tạo khác nhau, lấy trung bình xác suất dự đoán.

### Lựa chọn và đánh giá mô hình

Giữ nguyên cách chia train/test chính thức của UCI Adult. Lựa chọn cấu hình bằng macro-F1 trung bình trên Stratified 3-fold Cross-Validation với seed 42. Sau khi chốt cấu hình, đánh giá Random Forest và mô hình học sâu được chọn trên tập test bằng accuracy, precision, recall, F1-score, ROC-AUC, average precision và ma trận nhầm lẫn. Kết quả test không được dùng để điều chỉnh cấu hình mô hình.

## Hướng dẫn chạy

1. Mở notebook bằng liên kết Colab bên dưới.
2. Sử dụng runtime CPU.
3. Chọn Runtime → Run all.
4. Notebook tự cài thư viện, tải source và dữ liệu.
5. Sau khi chạy xong, tải notebook có output và ZIP kết quả
   từ bảng Files của Colab.

Yêu cầu kết nối Internet để tải thư viện, module và dữ liệu.
Notebook ghim source tại một commit và kiểm tra SHA-256.
Phiên bản thư viện thực tế được ghi trong environment.json.

## Cấu trúc thư mục

```text
ML-Assignment-Adult-Census/
├── README.md
├── requirements.txt
├── notebooks/
│   └── main_pipeline.ipynb
├── modules/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessor.py
│   ├── traditional_models.py
│   └── deep_learning_models.py
├── reports/
│   └── Report_AdultCensus.pdf
└── features/
    ├── X_train_cfg1.npy
    ├── X_test_cfg1.npy
    ├── y_train.npy
    ├── y_test.npy
    └── metadata_cfg1.json
```

## Kết quả

Kết quả trên 16.281 mẫu thuộc tập test chính thức của UCI Adult, lấy từ output đã lưu trong [notebook](notebooks/main_pipeline.ipynb), khớp với Bảng 11 trong [báo cáo](reports/Report_AdultCensus.pdf):

| Mô hình | Accuracy | Macro-F1 | F1 lớp >50K | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|
| Random Forest tuned | 0.8628 | 0.7949 | 0.6768 | 0.9149 | 0.7965 |
| MLP piecewise weighted 256-128 + ensemble 3 seeds | 0.8607 | 0.8109 | 0.7139 | 0.9230 | 0.8131 |

Tổ hợp MLP đạt macro-F1 cao hơn Random Forest 0.0160, trong khi Random Forest đạt accuracy cao hơn 0.0021. Các mô hình được chọn bằng kết quả CV trên tập huấn luyện trước khi đánh giá trên tập test.

## File đặc trưng

features/ chứa đặc trưng đã biến đổi bởi bộ tiền xử lý
của Random Forest và các nhãn tương ứng.
metadata_cfg1.json mô tả tên đặc trưng, ánh xạ nhãn
và phiên bản thư viện.
Mô hình học sâu sử dụng bộ tiền xử lý riêng.

## Liên kết

- [Colab notebook](https://colab.research.google.com/github/Tri8505/ML-Assignment-Adult-Census/blob/main/notebooks/main_pipeline.ipynb)
- [Báo cáo PDF](reports/Report_AdultCensus.pdf)
- [Mã nguồn](https://github.com/Tri8505/ML-Assignment-Adult-Census)
