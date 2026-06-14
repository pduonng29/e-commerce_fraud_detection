# 💳 Phát hiện Gian lận Giao dịch Thương mại Điện tử bằng Machine Learning

> **Bài tập lớn môn Học máy — Bài toán Phân loại nhị phân**
>
> Xây dựng và đánh giá các mô hình **Machine Learning** nhằm dự đoán một giao dịch thương mại điện tử có phải là giao dịch gian lận hay không dựa trên thông tin người dùng, số tiền giao dịch, vị trí, khoảng cách vận chuyển và các cơ chế xác thực bảo mật.
>
> Dự án thực hiện đầy đủ quy trình từ kiểm tra dữ liệu, tiền xử lý, khai phá dữ liệu, Feature Engineering, xử lý mất cân bằng, huấn luyện mô hình, tối ưu siêu tham số bằng Optuna, tuning threshold đến lựa chọn mô hình tốt nhất.

---

## 📋 Mục lục

* [Giới thiệu](#-giới-thiệu)
* [Mục tiêu bài toán](#-mục-tiêu-bài-toán)
* [Dataset](#-dataset)
* [Quy trình thực hiện](#-quy-trình-thực-hiện)
* [Tiền xử lý dữ liệu](#-tiền-xử-lý-dữ-liệu)
* [Feature Engineering](#-feature-engineering)
* [Khai phá dữ liệu](#-khai-phá-dữ-liệu)
* [Xử lý mất cân bằng](#-xử-lý-mất-cân-bằng)
* [Các mô hình sử dụng](#-các-mô-hình-sử-dụng)
* [Tối ưu mô hình](#-tối-ưu-mô-hình)
* [Đánh giá mô hình](#-đánh-giá-mô-hình)
* [Kết quả](#-kết-quả)
* [Cấu trúc dự án](#-cấu-trúc-dự-án)
* [Yêu cầu môi trường](#-yêu-cầu-môi-trường)
* [Cài đặt và sử dụng](#-cài-đặt-và-sử-dụng)
* [Hạn chế và hướng phát triển](#-hạn-chế-và-hướng-phát-triển)
* [Tác giả](#-tác-giả)

---

## 📖 Giới thiệu

Gian lận giao dịch thương mại điện tử có thể gây thiệt hại tài chính cho khách hàng, doanh nghiệp và các tổ chức thanh toán.

Dự án xây dựng mô hình phân loại một giao dịch thành hai nhóm:

| Nhãn | Ý nghĩa            |
| ---: | ------------------ |
|  `0` | Giao dịch hợp lệ   |
|  `1` | Giao dịch gian lận |

Đây là bài toán **Tabular Binary Classification**.

Dữ liệu có hiện tượng mất cân bằng mạnh, trong đó giao dịch gian lận chỉ chiếm khoảng **2,21%** tổng số giao dịch. Vì vậy, Accuracy không được sử dụng làm chỉ số chính. Dự án tập trung vào:

* **Recall:** khả năng phát hiện các giao dịch gian lận thật;
* **F2-score:** ưu tiên Recall cao hơn Precision;
* **PR-AUC:** phù hợp với dữ liệu mất cân bằng;
* **MCC:** đánh giá cân bằng dựa trên toàn bộ Confusion Matrix;
* **Cost:** phạt trường hợp bỏ sót gian lận nặng hơn cảnh báo nhầm.

---

## 🎯 Mục tiêu bài toán

Dự án hướng tới các mục tiêu:

* Kiểm tra và làm sạch dữ liệu giao dịch;
* Phân tích sự khác biệt giữa fraud và non-fraud;
* Xử lý missing value, duplicate và dữ liệu bất hợp lý;
* Chuẩn hóa và mã hóa các biến phân loại;
* Tạo thêm các feature có ý nghĩa nghiệp vụ;
* Rút gọn các feature bị trùng lặp hoặc ít đóng góp;
* Xử lý hiện tượng mất cân bằng lớp;
* So sánh nhiều thuật toán phân loại;
* Tối ưu LightGBM và XGBoost bằng Optuna;
* Tìm threshold phù hợp thay vì cố định ở `0.5`;
* Lựa chọn Champion Model trên tập Validation;
* Đánh giá cuối cùng trên tập Test;
* Phân tích mức độ quan trọng của các feature.

---

## 📦 Dataset

Dự án sử dụng file:

```text
transactions.csv
```

### Thông tin tổng quan

| Thuộc tính     |               Giá trị |
| -------------- | --------------------: |
| Số giao dịch   |               299.695 |
| Số cột ban đầu |                    17 |
| Biến mục tiêu  |            `is_fraud` |
| Tỷ lệ fraud    |          Khoảng 2,21% |
| Loại bài toán  | Binary Classification |

### Danh sách các nhóm biến

#### Thông tin định danh

| Cột              | Ý nghĩa       |
| ---------------- | ------------- |
| `transaction_id` | Mã giao dịch  |
| `user_id`        | Mã người dùng |

#### Thông tin người dùng

| Cột                       | Ý nghĩa                                     |
| ------------------------- | ------------------------------------------- |
| `account_age_days`        | Số ngày tài khoản đã hoạt động              |
| `total_transactions_user` | Tổng số giao dịch của người dùng            |
| `avg_amount_user`         | Giá trị giao dịch trung bình của người dùng |

#### Thông tin giao dịch

| Cột                    | Ý nghĩa                       |
| ---------------------- | ----------------------------- |
| `amount`               | Số tiền giao dịch             |
| `channel`              | Kênh thực hiện giao dịch      |
| `merchant_category`    | Nhóm ngành của người bán      |
| `promo_used`           | Trạng thái sử dụng khuyến mãi |
| `shipping_distance_km` | Khoảng cách vận chuyển        |

#### Thông tin vị trí

| Cột           | Ý nghĩa                      |
| ------------- | ---------------------------- |
| `country`     | Quốc gia phát sinh giao dịch |
| `bin_country` | Quốc gia phát hành thẻ       |

#### Thông tin bảo mật

| Cột             | Ý nghĩa                       |
| --------------- | ----------------------------- |
| `avs_match`     | Kết quả xác thực địa chỉ      |
| `cvv_result`    | Kết quả xác thực CVV          |
| `three_ds_flag` | Trạng thái xác thực 3D Secure |

#### Biến mục tiêu

| Giá trị | Ý nghĩa            |
| ------: | ------------------ |
|     `0` | Giao dịch hợp lệ   |
|     `1` | Giao dịch gian lận |

---

## 🔄 Quy trình thực hiện

```text
Đọc dữ liệu
    ↓
Kiểm tra cấu trúc và chất lượng dữ liệu
    ↓
Loại bỏ các cột không sử dụng
    ↓
Kiểm tra missing, duplicate và dữ liệu bất hợp lý
    ↓
Chuẩn hóa biến phân loại
    ↓
Gộp nhóm hiếm
    ↓
Feature Engineering rút gọn
    ↓
Điền missing value
    ↓
One-Hot Encoding và Scaling
    ↓
Khai phá dữ liệu EDA
    ↓
Chia Train / Validation / Test
    ↓
Xử lý mất cân bằng
    ↓
Huấn luyện các mô hình
    ↓
Tối ưu LightGBM và XGBoost bằng Optuna
    ↓
Threshold Tuning
    ↓
Lựa chọn Champion Model
    ↓
Đánh giá cuối cùng trên Test
```

---

## 🧹 Tiền xử lý dữ liệu

### Loại bỏ các cột không sử dụng

Ba cột được loại khỏi dữ liệu mô hình:

```text
transaction_id
user_id
transaction_time
```

Lý do:

* `transaction_id` và `user_id` chỉ là mã định danh;
* Các mã định danh có thể khiến mô hình ghi nhớ từng đối tượng;
* `transaction_time` được loại để bài toán không sử dụng yếu tố chuỗi thời gian.

Sau bước này, đề tài được xác định là:

> **Tabular Binary Classification**, không phải Time Series Forecasting.

### Kiểm tra missing value

Dự án kiểm tra cả:

* Missing thực sự: `NaN`, `None`, `pd.NA`;
* Missing dạng chuỗi: `"NA"`, `"N/A"`, `"null"`, `"Unknown"`, `"?"`, `"-"`.

Cách xử lý:

| Loại biến      | Phương pháp         |
| -------------- | ------------------- |
| Biến số        | Điền bằng median    |
| Biến phân loại | Điền bằng `Unknown` |

Median được lựa chọn vì ít bị ảnh hưởng bởi outlier hơn mean.

### Kiểm tra duplicate

Duplicate được kiểm tra theo nhiều mức:

* Trùng toàn bộ dòng;
* Trùng sau khi chuẩn hóa chuỗi;
* Trùng toàn bộ feature nhưng không xét target;
* Feature giống nhau nhưng target khác nhau;
* Số lần xuất hiện của từng nhóm duplicate.

Chỉ các dòng trùng hoàn toàn mới được tự động loại bỏ.

### Kiểm tra dữ liệu bất hợp lý

Các biến sau không nên nhận giá trị âm:

```text
amount
avg_amount_user
shipping_distance_km
account_age_days
total_transactions_user
```

Giá trị âm được chuyển thành `NaN` và xử lý bằng median ở bước tiếp theo.

### Chuẩn hóa biến phân loại

Các biến categorical được:

* Xóa khoảng trắng thừa;
* Chuyển về chữ thường;
* Chuẩn hóa các chuỗi biểu diễn missing;
* Gộp nhóm xuất hiện dưới `0.5%` thành `other`.

Ví dụ:

```text
"US"       → "us"
" us "     → "us"
"MOBILE"   → "mobile"
```

### One-Hot Encoding

Các biến phân loại được chuyển thành các cột nhị phân.

Ví dụ:

```text
channel = mobile, web, pos

→ channel_mobile
→ channel_web
→ channel_pos
```

One-Hot Encoding tránh tạo ra quan hệ thứ tự giả giữa các nhóm.

### Loại cột hằng số và gần hằng số

Các cột bị loại gồm:

* Cột chỉ có một giá trị duy nhất;
* Cột có một giá trị chiếm trên `99.9%` số quan sát.

Những cột này gần như không giúp phân biệt fraud và non-fraud.

### Scaling

Dự án sử dụng `RobustScaler`:

```text
x_scaled = (x - Median) / (Q3 - Q1)
```

`RobustScaler` ít bị ảnh hưởng bởi outlier hơn `StandardScaler`.

* Logistic Regression sử dụng dữ liệu đã scaling;
* Decision Tree, Random Forest, LightGBM và XGBoost sử dụng dữ liệu One-Hot chưa scaling.

---

## 🛠 Feature Engineering

Feature Engineering được rút gọn nhằm giảm số cột trùng lặp, giảm thời gian huấn luyện và hạn chế overfitting.

### Các feature mới

| Feature                    | Ý nghĩa                                        |
| -------------------------- | ---------------------------------------------- |
| `log_amount`               | Số tiền giao dịch sau log-transform            |
| `log_shipping_distance_km` | Khoảng cách vận chuyển sau log-transform       |
| `country_mismatch`         | Quốc gia giao dịch khác quốc gia phát hành thẻ |
| `security_score`           | Tổng số cơ chế bảo mật thành công              |
| `security_fail_count`      | Số cơ chế bảo mật thất bại                     |
| `security_strong`          | Cả AVS, CVV và 3DS đều thành công              |
| `amount_5x_avg`            | Giao dịch lớn hơn ít nhất 5 lần mức trung bình |
| `is_new_account_90d`       | Tài khoản có tuổi không quá 90 ngày            |
| `shipping_gt_500km`        | Khoảng cách vận chuyển từ 500 km trở lên       |

### Log-transform

Phép biến đổi được sử dụng:

```text
x_log = ln(1 + x)
```

Log-transform giúp:

* Giảm độ lệch phải;
* Nén các giá trị quá lớn;
* Giảm ảnh hưởng của outlier;
* Giúp Logistic Regression học ổn định hơn.

### Lợi ích của Feature Engineering rút gọn

* Giảm số chiều dữ liệu;
* Giảm thời gian train;
* Giảm thời gian chạy Optuna;
* Hạn chế feature trùng thông tin;
* Giảm nguy cơ overfitting;
* Giúp Feature Importance dễ giải thích hơn.

---

## 🔍 Khai phá dữ liệu

### Phân phối biến mục tiêu

Biểu đồ cột và donut chart được sử dụng để phân tích số lượng và tỷ lệ của hai lớp.

Kết quả cho thấy lớp fraud chiếm tỷ lệ nhỏ, xác nhận dữ liệu bị mất cân bằng mạnh.

### Phân tích outlier

Outlier được xác định bằng phương pháp IQR:

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Outlier không được tự động xóa vì giao dịch bất thường có thể chính là dấu hiệu gian lận.

### Boxplot theo target

Boxplot được sử dụng để so sánh phân phối của các biến giữa:

* Giao dịch hợp lệ;
* Giao dịch gian lận.

### Phân phối trước và sau log-transform

Histogram và KDE được sử dụng để quan sát tác dụng của log-transform đối với:

* `amount`;
* `shipping_distance_km`.

### Fraud Rate

Fraud rate của mỗi nhóm được tính bằng:

```text
Fraud Rate = Fraud Count / Total Transactions
```

### Fraud Lift

Fraud Lift được tính bằng:

```text
Lift = Fraud Rate của nhóm / Fraud Rate trung bình
```

* `Lift = 1`: bằng mức rủi ro trung bình;
* `Lift > 1`: rủi ro cao hơn trung bình;
* `Lift < 1`: rủi ro thấp hơn trung bình.

### Heatmap tương tác

Heatmap được sử dụng để phân tích fraud rate khi hai điều kiện cùng xuất hiện.

Một số tổ hợp được khảo sát:

* Bảo mật yếu và giao dịch lớn bất thường;
* Quốc gia không khớp và không có 3D Secure;
* Tài khoản mới và giao dịch lớn;
* Khoảng cách vận chuyển xa và trạng thái bảo mật.

Các giá trị được hiển thị trực tiếp trong từng ô bằng:

```python
annot=True
fmt=".2f"
```

### Correlation Heatmap

Correlation Heatmap thể hiện hệ số tương quan Pearson:

```text
-1 ≤ r ≤ 1
```

* `r` gần `1`: tương quan thuận mạnh;
* `r` gần `-1`: tương quan nghịch mạnh;
* `r` gần `0`: quan hệ tuyến tính yếu.

Các hệ số được hiển thị trực tiếp trong từng ô để dễ phát hiện những feature có thông tin trùng lặp.

---

## ⚖️ Xử lý mất cân bằng

Dữ liệu fraud có tỷ lệ lớp dương rất thấp.

Hệ số trọng số được tính bằng:

```text
scale_pos_weight = số non-fraud / số fraud
```

Các phương pháp được sử dụng:

| Mô hình             | Phương pháp                         |
| ------------------- | ----------------------------------- |
| Logistic Regression | `class_weight="balanced"`           |
| Decision Tree       | `class_weight="balanced"`           |
| Random Forest       | `class_weight="balanced_subsample"` |
| LightGBM            | `scale_pos_weight`                  |
| XGBoost             | `scale_pos_weight`                  |

Ngoài ra, threshold được tối ưu trên tập Validation để tăng khả năng phát hiện fraud.

---

## 🤖 Các mô hình sử dụng

### Baseline Models

| Mô hình             | Vai trò                           |
| ------------------- | --------------------------------- |
| Dummy Classifier    | Mốc đánh giá cơ sở                |
| Logistic Regression | Mô hình tuyến tính                |
| Decision Tree       | Mô hình cây đơn                   |
| Random Forest       | Ensemble theo phương pháp bagging |

### Advanced Models

| Mô hình       | Vai trò                                      |
| ------------- | -------------------------------------------- |
| LightGBM Base | Gradient Boosting hiệu quả trên dữ liệu bảng |
| XGBoost Base  | Gradient Boosting có regularization mạnh     |

### Tuned Models

| Mô hình                | Vai trò                           |
| ---------------------- | --------------------------------- |
| LightGBM Optuna Tuned  | LightGBM được tối ưu siêu tham số |
| XGBoost Optuna Tuned ⭐ | Mô hình tốt nhất của dự án        |

---

## 🔧 Tối ưu mô hình

Optuna được sử dụng để tối ưu LightGBM và XGBoost.

Một số siêu tham số được khảo sát:

* `n_estimators`;
* `learning_rate`;
* `max_depth`;
* `num_leaves`;
* `min_child_samples`;
* `min_child_weight`;
* `subsample`;
* `colsample_bytree`;
* `reg_alpha`;
* `reg_lambda`;
* `scale_pos_weight`.

Hàm mục tiêu ưu tiên F2-score và đồng thời xem xét MCC cùng chi phí:

```text
Score = F2 + 0.05 × max(MCC, 0) - 10⁻⁶ × Cost
```

---

## 📈 Đánh giá mô hình

### Precision

```text
Precision = TP / (TP + FP)
```

Cho biết trong số các giao dịch bị cảnh báo fraud, có bao nhiêu giao dịch thực sự là fraud.

### Recall

```text
Recall = TP / (TP + FN)
```

Cho biết mô hình phát hiện được bao nhiêu giao dịch fraud thật.

### F2-score

```text
F2 = 5 × Precision × Recall / (4 × Precision + Recall)
```

F2-score ưu tiên Recall hơn Precision.

### PR-AUC

PR-AUC đánh giá chất lượng mô hình trên đường Precision–Recall và phù hợp với dữ liệu mất cân bằng.

### ROC-AUC

ROC-AUC đánh giá khả năng phân biệt hai lớp trên nhiều threshold.

### MCC

MCC sử dụng cả TP, TN, FP và FN nên cung cấp đánh giá cân bằng hơn khi dữ liệu lệch lớp.

### Cost

Quy tắc chi phí:

```text
Cost = 10 × FN + 1 × FP
```

Bỏ sót một giao dịch gian lận được xem là nghiêm trọng hơn cảnh báo nhầm một giao dịch hợp lệ.

### Threshold Tuning

Threshold được khảo sát trong khoảng:

```text
0.001 → 0.999
```

Threshold tốt nhất được lựa chọn trên Validation thay vì sử dụng mặc định `0.5`.

---

## 🏆 Kết quả

Kết quả đánh giá các mô hình trên tập Test:

| Model                      | Threshold |  Precision |     Recall |         F1 |         F2 |        MCC |    ROC-AUC |     PR-AUC |      FP |      FN |      TP |         TN |      Cost |
| -------------------------- | --------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ------: | ------: | ------: | ---------: | --------: |
| **XGBoost Optuna Tuned ⭐** | **0.633** | **0.7954** | **0.8306** | **0.8126** | **0.8233** | **0.8085** | **0.9791** | **0.8733** | **212** | **168** | **824** | **43.750** | **1.892** |
| Random Forest              |     0.329 |     0.7854 |     0.8266 |     0.8055 |     0.8180 |     0.8013 |     0.9745 |     0.8594 |     224 |     172 |     820 |     43.738 |     1.944 |
| LightGBM Optuna Tuned      |     0.171 |     0.7647 |     0.8286 |     0.7954 |     0.8150 |     0.7912 |     0.9787 |     0.8714 |     253 |     170 |     822 |     43.709 |     1.953 |
| LightGBM Base              |     0.787 |     0.7544 |     0.8296 |     0.7902 |     0.8134 |     0.7862 |     0.9783 |     0.8645 |     268 |     169 |     823 |     43.694 |     1.958 |
| XGBoost Base               |     0.835 |     0.7471 |     0.8306 |     0.7866 |     0.8125 |     0.7827 |     0.9784 |     0.8699 |     279 |     168 |     824 |     43.683 |     1.959 |
| Decision Tree              |     0.947 |     0.7884 |     0.7964 |     0.7924 |     0.7948 |     0.7877 |     0.9610 |     0.8293 |     212 |     202 |     790 |     43.750 |     2.232 |
| Logistic Regression        |     0.685 |     0.7005 |     0.7923 |     0.7436 |     0.7721 |     0.7389 |     0.9760 |     0.8297 |     336 |     206 |     786 |     43.626 |     2.396 |
| Dummy Classifier           |     0.001 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.0000 |     0.5000 |     0.0221 |       0 |     992 |       0 |     43.962 |     9.920 |

### 🥇 Champion Model

> **XGBoost Optuna Tuned** được lựa chọn là mô hình tốt nhất với threshold bằng **0.633**.

Mô hình đạt:

* **Precision:** 79,54%;
* **Recall:** 83,06%;
* **F1-score:** 81,26%;
* **F2-score:** 82,33%;
* **MCC:** 80,85%;
* **ROC-AUC:** 97,91%;
* **PR-AUC:** 87,33%;
* **Cost:** 1.892.

Mô hình phát hiện đúng:

```text
824 / 992 giao dịch gian lận
```

Kết quả Confusion Matrix:

| Thành phần     | Số lượng |
| -------------- | -------: |
| True Positive  |      824 |
| True Negative  |   43.750 |
| False Positive |      212 |
| False Negative |      168 |

### Nhận xét

XGBoost Optuna Tuned đạt:

* F2-score cao nhất;
* MCC cao nhất;
* ROC-AUC cao nhất;
* PR-AUC cao nhất;
* Cost thấp nhất.

Kết quả cho thấy quá trình tối ưu bằng Optuna đã giúp XGBoost cải thiện rõ rệt so với XGBoost Base, đặc biệt về Precision, F2-score và tổng chi phí.

---

## 📁 Cấu trúc dự án

```text
e-commerce-fraud-detection/
│
├── BTL_phan_loai_nhi_phan_markdown_toi_uu.ipynb
├── README.md
├── .gitignore
│
├── data/
│   └── transactions.csv
│
└── models/
    └── fraud_detection_model.joblib
```

Nếu `transactions.csv` hoặc file mô hình có dung lượng lớn, có thể không tải trực tiếp lên GitHub mà cung cấp đường dẫn tải riêng.

---

## ⚙️ Yêu cầu môi trường

| Thành phần   | Phiên bản khuyến nghị |
| ------------ | --------------------- |
| Python       | 3.9 – 3.12            |
| pandas       | ≥ 1.5                 |
| NumPy        | ≥ 1.23                |
| scikit-learn | ≥ 1.2                 |
| Matplotlib   | ≥ 3.6                 |
| Seaborn      | ≥ 0.12                |
| LightGBM     | ≥ 4.0                 |
| XGBoost      | ≥ 2.0                 |
| Optuna       | ≥ 3.0                 |
| Joblib       | ≥ 1.2                 |

---

## 🚀 Cài đặt và sử dụng

### 1. Clone repository

```bash
git clone <repository-url>
cd e-commerce-fraud-detection
```

### 2. Tạo virtual environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Linux hoặc macOS

```bash
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost optuna joblib
```

### 4. Chuẩn bị dữ liệu

Đặt file:

```text
transactions.csv
```

vào thư mục:

```text
data/
```

Hoặc thay đổi đường dẫn trong notebook:

```python
DATA_PATH = "/đường/dẫn/đến/transactions.csv"
```

### 5. Chạy notebook

Mở file:

```text
BTL_phan_loai_nhi_phan_markdown_toi_uu.ipynb
```

bằng Google Colab hoặc Jupyter Notebook.

Trên Google Colab chọn:

```text
Runtime → Run all
```

Notebook sẽ hiển thị:

* Kết quả kiểm tra chất lượng dữ liệu;
* Các biểu đồ EDA;
* Ma trận tương quan;
* Kết quả các mô hình;
* Threshold tốt nhất;
* Bảng so sánh Validation và Test;
* Champion Model;
* Feature Importance;
* Dự đoán các mẫu giao dịch từ Test.

---

## ⚠️ Hạn chế và hướng phát triển

### Hạn chế

* `transaction_time` đã được loại bỏ nên không phân tích được concept drift;
* Kết quả phụ thuộc vào threshold đã lựa chọn;
* Một số feature tổng hợp có thể tương quan với feature gốc;
* Chất lượng mô hình phụ thuộc vào độ chính xác của nhãn fraud;
* Tiền xử lý được thực hiện trước khi chia tập theo cấu trúc của bài.

### Hướng phát triển

* Đóng gói tiền xử lý bằng `Pipeline` và `ColumnTransformer`;
* Fit imputer, encoder và scaler chỉ trên Train;
* Sử dụng SHAP để giải thích từng giao dịch;
* Kiểm tra Probability Calibration;
* Thử nghiệm Stacking hoặc Blending;
* Tối ưu threshold theo chi phí nghiệp vụ thực tế;
* Xây dựng API bằng FastAPI;
* Xây dựng giao diện dự đoán;
* Theo dõi dữ liệu mới và cập nhật mô hình định kỳ.

---

## 👤 Tác giả

| Thông tin            | Nội dung                                        |
| -------------------- | ----------------------------------------------- |
| Họ và tên            | Phạm Thùy Dương                                 |
| Mã sinh viên         | A50757                                          |
| Lớp                  | TI3702                                          |
| Môn học              | Học máy                                         |
| Đề tài               | Phát hiện gian lận giao dịch thương mại điện tử |
| Giảng viên hướng dẫn | Ngô Mạnh Cường                                  |

---

## ✅ Kết luận

Dự án đã xây dựng một quy trình Machine Learning hoàn chỉnh cho bài toán phát hiện gian lận giao dịch thương mại điện tử.

Quy trình bao gồm:

* Kiểm tra và làm sạch dữ liệu;
* Tiền xử lý biến số và biến phân loại;
* Feature Engineering rút gọn;
* Khai phá dữ liệu;
* Xử lý mất cân bằng;
* Huấn luyện nhiều mô hình;
* Tối ưu LightGBM và XGBoost bằng Optuna;
* Tuning threshold;
* Lựa chọn Champion Model;
* Đánh giá cuối cùng trên tập Test.

Mô hình **XGBoost Optuna Tuned** được lựa chọn là Champion Model nhờ đạt F2-score, MCC, ROC-AUC và PR-AUC cao nhất, đồng thời có tổng chi phí dự đoán sai thấp nhất trong các mô hình được thử nghiệm.

