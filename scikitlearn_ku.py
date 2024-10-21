import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('bike.csv')
df = df.drop('id', axis=1)

shanghai_df = df[df['city'] == 1].drop('city', axis=1)

shanghai_df['hour'] = shanghai_df['hour'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

y = shanghai_df['y'].values.reshape(-1, 1)
shanghai_df = shanghai_df.drop('y', axis=1)

X = shanghai_df.values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建特征归一化器
feature_scaler = MinMaxScaler()

# 创建目标标签归一化器
label_scaler = MinMaxScaler()

# 对训练集数据进行归一化
X_train_scaled = feature_scaler.fit_transform(X_train)

# 对训练集标签进行归一化
y_train_scaled = label_scaler.fit_transform(y_train)

# 对测试集数据进行归一化，注意这里使用训练集的归一化参数
X_test_scaled = feature_scaler.transform(X_test)

# 对测试集标签进行归一化，同样使用训练集的归一化参数
y_test_scaled = label_scaler.transform(y_test)

# 构建线性回归模型
linear_model = LinearRegression()

# 利用训练集训练模型
linear_model.fit(X_train_scaled, y_train_scaled)

print("线性回归模型训练完成。")

# 使用测试集进行评估
y_pred = linear_model.predict(X_test_scaled)

# 计算均方误差（MSE）
mse = mean_squared_error(y_test_scaled, y_pred)

# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)
print(f"均方根误差（RMSE）: {rmse}")