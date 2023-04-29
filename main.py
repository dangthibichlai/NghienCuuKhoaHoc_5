import numbers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


dataset = pd.read_csv('cleveland.csv',sep=',')
dataset = dataset.replace('?', np.nan)

print(dataset.info())

# thiếu 6 cái dữ liệu missing

data = dataset.dropna(subset=['ca','thal'])

print(data.info())

# đã drop dòng dữ liệu missing
# Export du lieu ra file csv

# 0 là không bị bệnh tim, 1234 là bị bệnh tim
data['num'] = data['num'].apply(lambda x: 0 if x == 0 else 1)
# xem dữ liệu biểu đồ
# set cột y chỉ chứa giá trị nguyên
# suwr dungj plt để vẽ biểu đồ cho chol, thalach,oldpeak
#Sử dụng thư viện matplotlib để vẽ biểu đồ cho 3 thuộc tính:  chol, thalach,oldpeak

#
# plt.figure(figsize=(10, 10))
# plt.subplot(3, 1, 1)
# plt.hist(data['chol'], bins=20)
# plt.title('chol')
# plt.subplot(3, 1, 2)
# plt.hist(data['thalach'], bins=20)
# plt.title('thalach')
# plt.subplot(3, 1, 3)
# plt.hist(data['oldpeak'], bins=20)
# plt.title('oldpeak')
# plt.show()

# thuật toán KNN
# chia dữ liệu thành 2 phần train và test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop(['num'], axis=1), data['num'], test_size=0.25, random_state=0)
# StandardScaler
import sklearn.preprocessing as preprocessing
st_x = preprocessing.StandardScaler()
X_train = st_x.fit_transform(X_train)
X_test = st_x.transform(X_test)

# in ra các tập train và test
# print("X_train: ")
# print(X_train)
# print("y_train: ")
# print(y_train)
# print("X_test: ")
# print(X_test)
# print("y_test: ")
# print(y_test)

# thuật toán KNN
# Chọn số K  tốt nhất
error = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10, 10))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()







# Huấn luyện mô hình với tập train
knnc = KNeighborsClassifier(n_neighbors=7, p=2 , metric='euclidean')
knnc.fit(X_train, y_train)

# Dự đoán kết quả cho tập test
y_pred = knnc.predict(X_test)

# Đánh giá mô hình
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

#
# navie bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))







