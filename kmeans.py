import matplotlib.pyplot as plt #Thư viện matplotlib.pyplot để vẽ biểu đồ
from sklearn.cluster import KMeans #Thư viện sklearn.cluster.KMeans để thực hiện phân cụm KMeans
import numpy as np #Thư viện numpy để xử lý mảng


 

#Khởi tạo dữ liệu đầu vào dưới dạng mảng numpy
# Dữ liệu: [Thu nhập, Chi tiêu]
data = np.array([
    [15000, 1000], [16000, 1200], [80000, 6000],
    [82000, 6200], [78000, 6100], [40000, 2500],
    [42000, 2600], [10000, 900], [43000, 2400],
    [12000, 1100]
])

# Áp dụng K-Means với K=3 (phân thành 3 cụm)
kmeans = KMeans(n_clusters=3, random_state=0) # radom_state để tái tạo kết quả, giúp kết quả cố định
kmeans.fit(data) #Huấn luyện mô hình KMeans với dữ liệu đầu vào
labels = kmeans.labels_ # Nhãn cụm cho mỗi điểm dữ liệu
centroids = kmeans.cluster_centers_ # Tâm cụm của KMeans

# Vẽ biểu đồ phân cụm
colors = ['red', 'green', 'blue']
for i in range(3): # Duyệt qua từng cụm
    # Lọc dữ liệu theo nhãn cụm
    cluster = data[labels == i] # Lấy dữ liệu thuộc cụm i ( gán nhãn cho dữ liệu)
    # cluster là mảng 2 chiều, mỗi hàng là một điểm dữ liệu thuộc cụm i
    # Vẽ dữ liệu của cụm i
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cụm {i+1}')

# Vẽ tâm cụm
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Tâm cụm')
'''centroids[:, 0] Lấy tất cả giá trị tọa độ X (ví dụ: thu nhập) của các tâm cụm.
centroids[:, 1] Lấy tất cả giá trị tọa độ Y (ví dụ: chi tiêu) của các tâm cụm.
c='black' Màu vẽ tâm cụm là màu đen.
marker='X' Hình dạng điểm vẽ là dấu "X" lớn (dễ nhận biết).
s=200 Kích thước điểm vẽ lớn (200 điểm), để nổi bật so với các điểm dữ liệu khác.
label='Tâm cụm' Gắn nhãn cho chú thích (legend) hiển thị là “Tâm cụm”.
'''
plt.xlabel('Thu nhập ($)')
plt.ylabel('Chi tiêu hàng tháng ($)')
plt.title('Phân nhóm khách hàng bằng K-Means') # tiêu đề biểu đồ
plt.legend() # hiển thị chú thích
plt.grid(True) # kẻ ôô
plt.show()