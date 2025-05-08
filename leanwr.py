# 🔰 Nhập thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 📦 Dữ liệu diện tích nhà (m²) và giá tương ứng (triệu đồng)
dien_tich = np.array([30, 40, 50, 60, 70, 80, 90, 100]).reshape(-1, 1)
gia = np.array([300, 400, 500, 600, 700, 800, 900, 1000])

# 🔀 Tách dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(dien_tich, gia, test_size=0.2, random_state=0)

# 🧠 Huấn luyện mô hình Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 🔍 Dự đoán trên dữ liệu kiểm tra
y_pred = model.predict(X_test)

# 🎯 In thông tin đánh giá
print("Hệ số góc (w):", model.coef_[0])
print("Độ lệch (b):", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 📈 Vẽ biểu đồ
plt.scatter(X_test, y_test, color='blue', label='Giá thực tế')
plt.plot(X_test, y_pred, color='red', label='Giá dự đoán')
plt.xlabel("Diện tích (m²)")
plt.ylabel("Giá nhà (triệu đồng)")
plt.title("Hồi quy tuyến tính: Diện tích vs Giá nhà")
plt.legend()
plt.grid(True)
plt.show()
