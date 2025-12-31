1 Giới thiệu đề tài

Giới thiệu

Dự án Mall Customer Segmentation nhằm phân đoạn khách hàng của một trung tâm mua sắm (mall) dựa trên các đặc điểm hành vi và dữ liệu giao dịch. Mục tiêu là giúp các nhà quản lý và tiếp thị hiểu rõ hơn về nhóm khách hàng của họ, từ đó tối ưu hóa chiến lược marketing và cải thiện trải nghiệm khách hàng.

Bằng cách sử dụng các thuật toán phân tích dữ liệu như K-means clustering, dự án này phân loại khách hàng thành các nhóm dựa trên các đặc điểm như tuổi, giới tính, tần suất mua sắm, chi tiêu và nhiều yếu tố khác.

Mục tiêu

Phân đoạn khách hàng: Tạo ra các nhóm khách hàng có đặc điểm tương đồng.

Phân tích hành vi khách hàng: Hiểu rõ hơn về thói quen và sở thích của từng nhóm.

Tối ưu hóa chiến lược marketing: Cung cấp thông tin chi tiết để cải thiện các chiến lược quảng cáo và khuyến mãi.

Yêu cầu

Python 3.x

Các thư viện cần thiết:

- pandas

- numpy

- matplotlib

- seaborn

- sklearn

2 Dataset

Mall Customer Dataset trên Kaggle. Đây là bộ dữ liệu rất phổ biến, thường được sử dụng trong các dự án phân tích dữ liệu và phân đoạn khách hàng. Bộ dữ liệu này chứa thông tin về khách hàng của một trung tâm mua sắm, và bạn có thể sử dụng nó để phân tích hành vi khách hàng, xây dựng mô hình phân nhóm (clustering), hoặc làm cơ sở cho các dự án học máy khác.

Để bắt đầu với bộ dữ liệu này, bạn có thể thực hiện các bước sau:

Tải Dữ Liệu:

Truy cập liên kết Kaggle của bạn: Mall Customer Dataset

Bạn cần có tài khoản Kaggle để tải bộ dữ liệu này về máy tính của mình.

Giải Nén Dữ Liệu:

Sau khi tải về, bạn sẽ có một tệp CSV, ví dụ: Mall_Customers.csv. Giải nén và mở tệp này trong môi trường lập trình như Jupyter Notebook, Google Colab, hoặc môi trường Python.

Khám Phá Dữ Liệu:

Bộ dữ liệu này thường có các cột như:

CustomerID: ID của khách hàng

Gender: Giới tính của khách hàng (Male/Female)

Age: Tuổi của khách hàng

Annual Income (k$): Thu nhập hàng năm của khách hàng

Spending Score (1-100): Điểm chi tiêu của khách hàng (dựa trên hành vi mua sắm)

Tiền Xử Lý Dữ Liệu:

Trước khi áp dụng bất kỳ thuật toán phân nhóm nào, bạn cần làm sạch và chuẩn hóa dữ liệu:

Kiểm tra các giá trị bị thiếu và xử lý chúng.

Mã hóa cột Gender (nếu cần).

Chuẩn hóa hoặc tiêu chuẩn hóa dữ liệu nếu bạn sử dụng các thuật toán như K-means.

Phân Nhóm (Clustering) với K-means:

Dưới đây là ví dụ về cách bạn có thể sử dụng K-means clustering với bộ dữ liệu này:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Mall_Customers.csv")

print(data.head())

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1, 11):

kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)

kmeans.fit(X_scaled)

wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,6))

plt.plot(range(1, 11), wcss)

plt.title('Elbow Method For Optimal K')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)

y_kmeans = kmeans.fit_predict(X_scaled)

data['Cluster'] = y_kmeans

plt.figure(figsize=(10, 6))

sns.scatterplot(data=data, x='Age', y='Annual Income (k$)', hue='Cluster', palette='Set1', s=100, alpha=0.7)

plt.title('Mall Customer Segmentation')

plt.xlabel('Age')

plt.ylabel('Annual Income (k$)')

plt.show()

3 Pipeline tổng thể

Để xây dựng một pipeline tổng thể cho dự án Mall Customer Segmentation, bạn sẽ cần trải qua các bước chính sau:

Thu thập dữ liệu: Tải và làm sạch dữ liệu.

Tiền xử lý dữ liệu: Bao gồm xử lý giá trị thiếu, chuẩn hóa và mã hóa dữ liệu.

Khám phá dữ liệu: Khám phá các đặc điểm của dữ liệu.

Xây dựng mô hình phân nhóm: Áp dụng thuật toán K-means (hoặc thuật toán phân nhóm khác).

Đánh giá mô hình và phân tích kết quả: Kiểm tra chất lượng mô hình và phân tích các nhóm khách hàng.

Trực quan hóa kết quả: Vẽ biểu đồ để dễ dàng hiểu và chia sẻ kết quả.

Dưới đây là một pipeline tổng thể cho dự án phân đoạn khách hàng trung tâm mua sắm (Mall Customer Segmentation), sử dụng K-means clustering và các bước chuẩn bị cần thiết.

1. Cài Đặt Các Thư Viện Cần Thiết

Trước khi bắt đầu, hãy chắc chắn rằng bạn đã cài đặt tất cả các thư viện cần thiết:

pip install pandas numpy matplotlib seaborn scikit-learn

2. Pipeline Tổng Thể

#Import các thư viện cần thiết

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score

#Bước 1: Tải dữ liệu

data = pd.read_csv("Mall_Customers.csv")

#Bước 2: Khám phá dữ liệu

print("Thông tin dữ liệu: ")

print(data.info()) # Kiểm tra kiểu dữ liệu và các giá trị thiếu

print("\nMô tả dữ liệu: ")

print(data.describe()) # Thống kê cơ bản về dữ liệu

#Kiểm tra dữ liệu thiếu

print("\nKiểm tra giá trị thiếu: ")

print(data.isnull().sum())

#Bước 3: Tiền xử lý dữ liệu

#Mã hóa giới tính: Male=0, Female=1

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

#Chọn các đặc trưng (features) cho phân nhóm: Age, Annual Income, Spending Score

features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

#Chuẩn hóa dữ liệu

scaler = StandardScaler()

features_scaled = scaler.fit_transform(features)

#Bước 4: Xác định số nhóm tối ưu (Sử dụng Elbow Method)

wcss = []

for i in range(1, 11): # Từ 1 đến 10 nhóm

kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)

kmeans.fit(features_scaled)

wcss.append(kmeans.inertia_)

#Vẽ đồ thị Elbow

plt.figure(figsize=(8,6))

plt.plot(range(1, 11), wcss)

plt.title('Elbow Method For Optimal K')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()

#Bước 5: Áp dụng K-means với số nhóm tối ưu (ví dụ K=5)

optimal_k = 5

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)

data['Cluster'] = kmeans.fit_predict(features_scaled)

#Bước 6: Đánh giá mô hình - Silhouette Score

sil_score = silhouette_score(features_scaled, data['Cluster'])

print(f"Silhouette Score: {sil_score:.4f}")

#Bước 7: Phân tích kết quả phân nhóm

#Trực quan hóa kết quả phân nhóm

plt.figure(figsize=(10, 6))

sns.scatterplot(x=data['Age'], y=data['Annual Income (k$)'], hue=data['Cluster'], palette='Set1', s=100, alpha=0.7)

plt.title(f'Mall Customer Segmentation (K={optimal_k})')

plt.xlabel('Age')

plt.ylabel('Annual Income (k$)')

plt.show()

#Trực quan hóa các nhóm theo Spending Score và Age

plt.figure(figsize=(10, 6))

sns.scatterplot(x=data['Age'], y=data['Spending Score (1-100)'], hue=data['Cluster'], palette='Set1', s=100, alpha=0.7)

plt.title(f'Mall Customer Segmentation - Spending Score vs Age (K={optimal_k})')

plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.show()

#Trực quan hóa các nhóm theo Spending Score và Annual Income

plt.figure(figsize=(10, 6))

sns.scatterplot(x=data['Annual Income (k$)'], y=data['Spending Score (1-100)'], hue=data['Cluster'], palette='Set1', s=100, alpha=0.7)

plt.title(f'Mall Customer Segmentation - Spending Score vs Annual Income (K={optimal_k})')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.show()

#Bước 8: Phân tích các nhóm

for i in range(optimal_k):

print(f"\nNhóm {i}:")

print(data[data['Cluster'] == i].describe())

3. Mô Tả Các Bước

Bước 1: Tải Dữ Liệu

Tải dữ liệu từ tệp CSV và hiển thị thông tin về các cột.

Bước 2: Khám Phá Dữ Liệu

Kiểm tra thông tin về dữ liệu, kiểu dữ liệu và các giá trị thiếu.

Thống kê mô tả dữ liệu để hiểu rõ hơn về đặc điểm khách hàng.

Bước 3: Tiền Xử Lý Dữ Liệu

Mã hóa giới tính (Male = 0, Female = 1).

Lựa chọn các đặc trưng cần thiết cho phân nhóm (tuổi, thu nhập hàng năm và điểm chi tiêu).

Chuẩn hóa dữ liệu để đảm bảo các đặc trưng có cùng tỷ lệ (sử dụng StandardScaler).

Bước 4: Xác Định Số Nhóm Tối Ưu (Elbow Method)

Dùng phương pháp Elbow để xác định số lượng nhóm tối ưu cho K-means.

Bước 5: Áp Dụng K-means

Sử dụng thuật toán K-means để phân nhóm khách hàng. Sau đó, gán nhãn nhóm vào cột mới trong dữ liệu (Cluster).

Bước 6: Đánh Giá Mô Hình

Sử dụng Silhouette Score để đánh giá chất lượng phân nhóm. Silhouette Score càng gần 1 thì mô hình phân nhóm càng tốt.

Bước 7: Trực Quan Hóa Kết Quả

Vẽ các biểu đồ scatter để trực quan hóa sự phân bố của khách hàng trong các nhóm, so với các đặc trưng như tuổi, thu nhập và điểm chi tiêu.

Bước 8: Phân Tích Các Nhóm

Phân tích mô tả cho mỗi nhóm khách hàng, ví dụ như độ tuổi trung bình, thu nhập, và điểm chi tiêu.

4 Mô hình sử dụng

Giới thiệu

Dự án Mall Customer Segmentation sử dụng thuật toán K-means clustering để phân nhóm khách hàng của một trung tâm mua sắm (mall) dựa trên các đặc điểm hành vi mua sắm của họ. Mục tiêu của dự án này là giúp các nhà quản lý và tiếp thị hiểu rõ hơn về nhóm khách hàng của họ, từ đó tối ưu hóa chiến lược marketing và cải thiện trải nghiệm khách hàng.

Các đặc điểm phân đoạn chính:

Tuổi (Age)

Thu nhập hàng năm (Annual Income)

Điểm chi tiêu (Spending Score)

Kết quả phân nhóm sẽ giúp các nhà tiếp thị và quản lý hiểu được các nhóm khách hàng với đặc điểm hành vi và nhu cầu khác nhau, từ đó tạo ra các chiến lược quảng cáo và khuyến mãi hiệu quả hơn.

Mục Tiêu

Phân đoạn khách hàng: Tạo ra các nhóm khách hàng có đặc điểm hành vi mua sắm tương tự.

Phân tích hành vi khách hàng: Hiểu rõ hơn về thói quen mua sắm và chi tiêu của từng nhóm khách hàng.

Tối ưu hóa chiến lược marketing: Cung cấp thông tin chi tiết để điều chỉnh các chiến lược quảng cáo và khuyến mãi phù hợp.

Yêu Cầu

Để chạy dự án này, bạn cần cài đặt Python và các thư viện sau:

Python 3.x

pandas

numpy

matplotlib

seaborn

scikit-learn

Cài Đặt

Để cài đặt các thư viện cần thiết, bạn có thể sử dụng pip:

pip install pandas numpy matplotlib seaborn scikit-learn

Dữ Liệu

Bộ dữ liệu Mall Customer Dataset có sẵn trên Kaggle và chứa các thông tin sau:

CustomerID: ID của khách hàng.

Gender: Giới tính của khách hàng (Male/Female).

Age: Tuổi của khách hàng.

Annual Income (k$): Thu nhập hàng năm của khách hàng.

Spending Score (1-100): Điểm chi tiêu của khách hàng (dựa trên hành vi mua sắm).

Link Tải Dữ Liệu: Mall Customer Dataset on Kaggle

Quy Trình Dự Án

1. Tiền Xử Lý Dữ Liệu

Mã hóa giới tính: Chuyển các giá trị Male và Female thành các giá trị số (0 và 1).

Chuẩn hóa dữ liệu: Chuẩn hóa các đặc trưng để các giá trị có tỷ lệ đồng nhất, giúp thuật toán K-means hoạt động hiệu quả hơn.

2. Phân Nhóm Khách Hàng

Sử dụng thuật toán K-means clustering để phân nhóm khách hàng thành k nhóm. Thuật toán này sẽ chia khách hàng thành các nhóm sao cho các điểm dữ liệu trong mỗi nhóm có sự tương đồng cao nhất.

3. Đánh Giá Mô Hình

Đánh giá chất lượng phân nhóm bằng chỉ số Silhouette Score, chỉ số này giúp đánh giá mức độ phân tách giữa các nhóm và sự đồng nhất trong mỗi nhóm.

4. Trực Quan Hóa Kết Quả

Sử dụng các biểu đồ để trực quan hóa kết quả phân nhóm và giúp hiểu rõ hơn về các nhóm khách hàng.

Cách Sử Dụng

Bước 1: Tải và làm sạch dữ liệu

Tải dữ liệu từ Kaggle và làm sạch dữ liệu bằng cách mã hóa giới tính và xử lý các giá trị thiếu (nếu có).

Bước 2: Phân nhóm khách hàng

Chạy thuật toán K-means để phân nhóm khách hàng dựa trên các đặc trưng như tuổi, thu nhập hàng năm, và điểm chi tiêu.

import pandas as pd

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

#Tải dữ liệu

data = pd.read_csv("Mall_Customers.csv")

#Mã hóa giới tính

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

#Chọn các cột để phân nhóm

features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

#Chuẩn hóa dữ liệu

scaler = StandardScaler()

features_scaled = scaler.fit_transform(features)

#Xác định số nhóm tối ưu với phương pháp Elbow

wcss = []

for i in range(1, 11):

kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)

kmeans.fit(features_scaled)

wcss.append(kmeans.inertia_)

#Vẽ biểu đồ Elbow

plt.plot(range(1, 11), wcss)

plt.title('Elbow Method For Optimal K')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()

#Chạy K-means với số nhóm tối ưu (giả sử K = 5)

optimal_k = 5

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)

data['Cluster'] = kmeans.fit_predict(features_scaled)

#Hiển thị kết quả phân nhóm

sns.scatterplot(data=data, x='Age', y='Annual Income (k$)', hue='Cluster', palette='Set1')

plt.title('Mall Customer Segmentation')

plt.show()

Bước 3: Đánh Giá Mô Hình

Đánh giá chất lượng phân nhóm bằng Silhouette Score:

from sklearn.metrics import silhouette_score

sil_score = silhouette_score(features_scaled, data['Cluster'])

print(f"Silhouette Score: {sil_score:.4f}")

Bước 4: Phân Tích và Trực Quan Hóa

Phân tích các nhóm khách hàng dựa trên các đặc điểm như tuổi, thu nhập và điểm chi tiêu.

Vẽ các biểu đồ phân nhóm để hiển thị các nhóm khách hàng.

Kết Quả và Ứng Dụng

Sau khi phân nhóm, bạn sẽ có các nhóm khách hàng với các đặc điểm tương tự nhau. Ví dụ:

Nhóm 1: Khách hàng trẻ với thu nhập thấp và chi tiêu cao.

Nhóm 2: Khách hàng trung niên với thu nhập cao và chi tiêu ít.

Nhóm 3: Khách hàng cao tuổi với thu nhập thấp và chi tiêu thấp.

5 Đánh giá kết quả

Kết quả phân nhóm này có thể được sử dụng để tạo ra các chiến lược marketing nhắm mục tiêu cho từng nhóm, chẳng hạn như:

Chiến dịch quảng cáo cho nhóm khách hàng trẻ tuổi.

Giảm giá hoặc khuyến mãi cho nhóm khách hàng có thu nhập thấp.

Tương Lai và Cải Tiến

Áp dụng các thuật toán phân nhóm khác: Bạn có thể thử nghiệm với các thuật toán phân nhóm khác như DBSCAN hoặc Agglomerative Clustering.

Phân tích cảm xúc khách hàng: Sử dụng phân tích cảm xúc từ đánh giá hoặc phản hồi khách hàng để phân nhóm dựa trên cảm xúc.

6️ Demo / Inference (Web Application)

Đây là một ứng dụng web đơn giản (Single Page Application) minh họa cách áp dụng thuật toán K-Means Clustering để phân loại khách hàng dựa trên thu nhập và hành vi chi tiêu.

Giới thiệu

Ứng dụng sử dụng các trọng tâm (centroids) đã được huấn luyện sẵn từ mô hình Machine Learning để dự đoán nhóm khách hàng ngay lập tức trên trình duyệt. Người dùng chỉ cần nhập dữ liệu, thuật toán sẽ tính toán Khoảng cách Euclid để tìm cụm gần nhất.

Tính năng chính

Phân loại thời gian thực: Xác định nhóm khách hàng ngay sau khi bấm nút.

Giao diện trực quan: Kết quả hiển thị kèm màu sắc định danh và chiến lược kinh doanh cụ thể.

Kiểm soát dữ liệu: Có các bước validate đầu vào (thu nhập > 0, điểm chi tiêu 1-100).

Thiết kế Responsive: Giao diện hiện đại, thân thiện với người dùng.

Công nghệ sử dụng

HTML5 & CSS3: Cấu trúc và giao diện (sử dụng Linear Gradient, Flexbox/Box-shadow).

JavaScript (Vanilla JS): Xử lý logic tính toán khoảng cách và điều khiển DOM.

K-Means Logic: Áp dụng công thức tính khoảng cách Euclid:

d
=
(
x
1
−
x
2
)
2
+
(
y
1
−
y
2
)
2

Cấu trúc các nhóm khách hàng

Dựa trên dữ liệu huấn luyện, khách hàng được chia thành 5 nhóm chính:

Nhóm khách Đặc điểm Chiến lược kinh doanh hàng

VIP Thu nhập cao -- Chi Chăm sóc đặc biệt, cá nhân hóa dịch tiêu cao vụ

Tiềm năng Thu nhập thấp -- Chi Ưu đãi, giảm giá để giữ chân tiêu cao

Cần kích cầu Thu nhập cao -- Chi Gửi các chương trình khuyến mãi, tiêu thấp gợi ý sản phẩm

Trung bình Thu nhập & Chi tiêu Duy trì tương tác, chăm sóc định kỳ mức vừa

Tiết kiệm Thu nhập thấp -- Chi Không ưu tiên nguồn lực marketing tiêu thấp
Hướng dẫn sử dụng

Tải tệp demo.html về máy tính.

Mở tệp bằng trình duyệt web bất kỳ (Chrome, Edge, Firefox).

Nhập Thu nhập hàng năm (k$) (ví dụ: 70).

Nhập Điểm chi tiêu (1-100) (ví dụ: 80).

Nhấn nút "Xác định nhóm khách hàng" để xem kết quả.

8️ Cấu trúc thư mục

mall_customer_segmentation/

├── data/

│ └── Mall_Customers.csv

├── models/

│ ├── kmeans_model.pkl

│ └── scaler.pkl

├── demo/

│ └── demo.html

├── src/

│ ├── train.py

│ ├── inference.py

│ └── utils.py

├── requirements.txt

└── README.md

9️ Kết luận

Dự án thể hiện rõ cách áp dụng **K-Means Clustering** cho bài toán thực tế.

Phần demo web giúp kết nối lý thuyết, code Python và ứng dụng trực quan, rất phù hợp cho đồ án học máy và trình diễn.


Triệu Quang Thịnh-12423034

Đào Đức Trọng - 10123329
