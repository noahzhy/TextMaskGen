import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt

# 数据
data = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [43, 62, 183],
    [84, 38, 838],
    [94, 71, 760],
    [106, 103, 765],
    [96, 138, 907],
    [169, 12, 50],
    [178, 14, 85],
    [174, 85, 1405],
    [115, 172, 1193],
    [56, 211, 62],
    [114, 210, 1205],
    [65, 253, 125],
    [117, 241, 992]
]

# 转换为DataFrame
df = pd.DataFrame(data, columns=['x', 'y', 'size'])

# sort via x then y
df = df.sort_values(['x', 'y'])
print(df)

# 特征和目标变量
X = df[['x', 'y']]
y = df['size']

# 拟合模型
model = BayesianRidge()
model.fit(X, y)

y_pred = model.predict(X)

# 计算残差
residuals = y - y_pred

# 识别离群点
threshold = 1.3 * np.std(residuals)  # 设定残差的阈值
outliers = df[np.abs(residuals) > threshold]

# print the points without outliers
tx = df[np.abs(residuals) <= threshold]
# to list
print(tx.values.tolist())

# visualize the results and index the outliers
plt.scatter(df['x'], df['y'], s=df['size'], c='blue', alpha=0.5)
plt.scatter(outliers['x'], outliers['y'], s=outliers['size'], c='red')
# show std threshold space
# plt.fill_between(
#     # x_test, ymean - ystd, ymean + ystd, color="pink", alpha=0.5, label="predict std"
#     df['x'], y_pred - threshold, y_pred + threshold, color="pink", alpha=0.5, label="predict std"
# )
plt.xlabel('x')
plt.ylabel('y')
plt.title('Identifying Outliers')
plt.show()
