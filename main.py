import numbers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('cleveland.csv',sep=',')
dataset = dataset.replace('?', np.nan)

print(dataset.info())

# thiếu 6 cái dữ liệu missing

data = dataset.dropna(subset=['ca','thal'])

print(data.info())

# đã drop dòng dữ liệu missing

# 0 là không bị bệnh tim, 1234 là bị bệnh tim
data['num'] = data['num'].apply(lambda x: 0 if x == 0 else 1)

# xem dữ liệu biểu đồ
# set cột y chỉ chứa giá trị nguyên
# suwr dungj plt để vẽ biểu đồ cho chol, thalach,oldpeak
#Sử dụng thư viện matplotlib để vẽ biểu đồ cho 3 thuộc tính:  chol, thalach,oldpeak

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.hist(data['chol'], bins=20)
plt.title('chol')
plt.subplot(3, 1, 2)
plt.hist(data['thalach'], bins=20)
plt.title('thalach')
plt.subplot(3, 1, 3)
plt.hist(data['oldpeak'], bins=20)
plt.title('oldpeak')
plt.show()









