PHÂN CỤM KHÁCH HÀNG SIÊU THỊ BẰNG THUẬT TOÁN K-MEANS

1. Giới thiệu đề tài

1.1. Bài toán

Trong lĩnh vực kinh doanh bán lẻ, việc hiểu rõ hành vi khách hàng đóng vai trò quan trọng trong việc xây dựng chiến lược marketing và nâng cao hiệu quả kinh doanh. Tuy nhiên, dữ liệu khách hàng thường không có nhãn phân loại sẵn, gây khó khăn cho việc áp dụng các phương pháp học có giám sát.

Bài toán đặt ra là phân nhóm khách hàng dựa trên các đặc trưng hành vi và nhân khẩu học, từ đó phát hiện các nhóm khách hàng có đặc điểm tương đồng mà không cần biết trước nhãn.

1.2. Mục tiêu

Mục tiêu của đề tài là:

Áp dụng thuật toán K-Means để phân cụm khách hàng
Phân tích hành vi mua sắm dựa trên thu nhập và mức chi tiêu
Xác định các nhóm khách hàng điển hình phục vụ cho marketing mục tiêu, chăm sóc khách hàng và đề xuất chiến lược kinh doanh

2. Dataset

2.1. Nguồn dữ liệu

Dataset được sử dụng trong đề tài là Mall Customers Dataset, một bộ dữ liệu phổ biến trong lĩnh vực khai phá dữ liệu và học máy.

Nguồn: Kaggle
Tên dataset: Mall Customer Data
Link tải:
https://www.kaggle.com/datasets/kahkashanmanzoor/mall-customer-dataset

2.2. Mô tả dữ liệu

Tập dữ liệu gồm 200 khách hàng, mỗi khách hàng được mô tả bởi 5 thuộc tính:

CustomerID: Mã định danh khách hàng
Gender: Giới tính khách hàng (Male / Female)
Age: Tuổi của khách hàng
Annual Income (k$): Thu nhập hàng năm (nghìn USD)
Spending Score (1–100): Điểm chi tiêu của khách hàng

2.3. Đặc điểm dữ liệu

Số lượng mẫu: 200

Giới tính:
Nữ: 56%
Nam: 44%

Tuổi: từ 18 đến 70
Thu nhập: từ 15 đến 137 (nghìn USD)
Điểm chi tiêu: từ 1 đến 99

Dữ liệu có sự phân bố đa dạng về thu nhập và hành vi chi tiêu, phù hợp cho việc áp dụng các thuật toán phân cụm không giám sát.

2.4. Thuộc tính sử dụng trong phân cụm

Trong quá trình phân tích, đề tài tập trung sử dụng hai thuộc tính chính:

Annual Income (k$)
Spending Score (1–100)

Hai thuộc tính này phản ánh trực tiếp khả năng chi trả và hành vi tiêu dùng của khách hàng, giúp mô hình phân cụm đạt hiệu quả trực quan và dễ diễn giải.

3. Pipeline thực hiện

3.1. Tiền xử lý dữ liệu

Loại bỏ các cột không phục vụ phân cụm như CustomerID
Mã hóa thuộc tính Gender nếu cần (không sử dụng trong phân cụm chính)
Lựa chọn hai đặc trưng chính: Annual Income (k$), Spending Score (1–100)
Chuẩn hóa dữ liệu bằng StandardScaler nhằm đưa các đặc trưng về cùng thang đo

3.2. Huấn luyện mô hình (Training)

Áp dụng thuật toán K-Means
Thử nghiệm nhiều giá trị số cụm K khác nhau (từ 4 đến 7)
Huấn luyện mô hình trên tập dữ liệu đã chuẩn hóa để tìm ra các tâm cụm tối ưu

3.3. Đánh giá mô hình (Evaluation)

Đánh giá bằng các chỉ số nội tại: Silhouette Score và Davies–Bouldin Index
Các chỉ số này phản ánh độ chặt chẽ trong cụm và độ tách biệt giữa các cụm

3.4. Suy luận (Inference)

Gán nhãn cụm cho từng khách hàng
Phân tích đặc điểm từng cụm để phục vụ phân khúc khách hàng và đề xuất chiến lược marketing

4. Mô hình sử dụng

4.1. Mô hình K-Means

K-Means là thuật toán phân cụm không giám sát phổ biến, hoạt động dựa trên việc tối thiểu hóa tổng khoảng cách từ các điểm dữ liệu đến tâm cụm.

Lý do lựa chọn K-Means:

Phù hợp với bài toán phân khúc khách hàng
Dễ triển khai và trực quan hóa kết quả
Hiệu quả với dữ liệu số liên tục
Thường được sử dụng trong các bài toán marketing thực tế

5. Kết quả và đánh giá

5.1. Kết quả đánh giá theo số cụm K

Số cụm (K) | Silhouette Score | Davies–Bouldin Index
4 | 0.4943 | 0.6975
5 | 0.5547 | 0.5722
6 | 0.5138 | 0.6239
7 | 0.5020 | 0.6925

5.2. Phân tích kết quả

Silhouette Score cao nhất tại K = 5
Davies–Bouldin Index thấp nhất tại K = 5

Do đó, K = 5 được lựa chọn là số cụm tối ưu cho bài toán phân khúc khách hàng.

6. Hướng dẫn chạy chương trình

6.1. Cài đặt môi trường

Hệ điều hành: Windows / Linux / macOS
Python: >= 3.9
Công cụ: Git, Jupyter Notebook hoặc VS Code

Cài đặt thư viện bằng lệnh:

pip install -r requirements.txt

6.2. Chạy huấn luyện mô hình

Mở Jupyter Notebook
Mở file: app/BTL_ML.ipynb
Chọn Kernel → Restart & Run All

6.3. Chạy demo / suy luận

Chương trình sẽ gán nhãn cụm cho từng khách hàng và trực quan hóa kết quả phân cụm bằng biểu đồ scatter.

6.4. Cấu trúc thư mục dự án

BTL_MachineLearning/
app/
└── BTL_ML.ipynb
data/
└── Mall_Customers.csv
demo/
└── demo.html
slides/
└── PHÂN CỤM KHÁCH HÀNG SIÊU...
.gitignore
README.md
requirements.txt

THÔNG TIN TÁC GIẢ

Triệu Quang Thịnh – MSV: 12423034
Đào Đức Trọng – MSV: 10123329
Lớp: 124231
