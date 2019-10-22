#!/usr/bin/env python
# coding: utf-8

"""
去掉不相干的列id, 对city,brand进行编码（使用LabelEncoder），将编码的类别字段保存到数据库，以便预测使用，features数据归一化处理，
洗牌数据，按照训练集：测试集=8：2进行分组，定义神经网络(Linear)定义激活函数(Sigmod)，定义损失函数(MSELoss)，循环2000次训练，保存模型
"""

import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch
import torch.nn as nn
import pymysql
from sklearn import preprocessing
import json
import db
df = pd.read_csv('t_car_last.csv')
d_plt = df
df.info()


df = df.drop(['id'], axis=1)
le = preprocessing.LabelEncoder()
le_count = 0

label_code_map = {}  # 保存编码的数据，在预测时需要将数据根据训练数据编码
for col in df:
    if df[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(df[col].unique())) >= 10:
            # Train on the training data
            le.fit(df[col])
            # Transform both training and testing data
            orl = df[col]
            df[col] = le.transform(orl)
            label_code_map[col] = le.classes_
            # Keep track of how many columns were label encoded
            le_count += 1

c = json.dumps(label_code_map['city'].tolist())
b = json.dumps(label_code_map['brand'].tolist())
def get_cursor():
    c = db.config()
    conn = pymysql.connect(host=c['host'], port=c['port'], user=c['user'], password=c['password'], charset=c['charset'], database=c['database'])
    cursor = conn.cursor()
    return cursor, conn
# 将编码保存，在预测时使用
cursor, conn = get_cursor()
sql = 'insert into t_encoder(city, brand, output_volume) values(%s, %s)'

cursor.execute(sql, [c, b])
conn.commit()
cursor.close()
conn.close()
# 数据归一化处理:(原值-平均值)/数据标准差 Z-score标准化方法
def z_scrore_normalize_feature(df):
    # return df.apply(lambda column: (column - column.mean())/column.std())
    ss = preprocessing.StandardScaler().fit(df)
    return ss.transform(df), ss.mean_, ss.var_
dz, mean_, var_ = z_scrore_normalize_feature(df.iloc[:, :8])

import json
c = json.dumps(mean_.tolist())
b = json.dumps(var_.tolist())
# In[90]:

cursor, conn = get_cursor()
sql = 'insert into t_statistics(mean, var) values(%s, %s)'

cursor.execute(sql, [c, b])
conn.commit()
cursor.close()
conn.close()
# max和min的归一化操作
def min_max_normalize_feature(df):
    df = (df - df.min())/(df.max() - df.min())
    return df
dmm = min_max_normalize_feature(df.iloc[:, :8])

df.describe()


y_data = np.array(df[df.columns[-1]]).reshape(len(df), 1) # 提取label
X_data = np.array(dz) # 提取feature


# 将数据集分成训练集和测试集 0.8
def shuffle_data(features: np.array, labels: np.array, test_size=0.8):
    np.random.seed(0)
    index = list(range(len(features)))
    np.random.shuffle(index)

    data_size = len(features)
    train_data_size = int(test_size * data_size)

    train_features = X_data[:train_data_size, :]
    train_labels = y_data[:train_data_size, :]

    test_features = X_data[train_data_size:, :]
    test_labels = y_data[train_data_size:, :]

    return train_features, train_labels, test_features, test_labels



train_features, train_labels, test_features, test_labels = shuffle_data(X_data, y_data)



def transfer_torch_data(train_features, train_labels, test_features, test_labels):
    return torch.from_numpy(train_features).float(),torch.from_numpy(train_labels).float(), torch.from_numpy(test_features).float(),torch.from_numpy(test_labels).float()

train_features, train_labels, test_features, test_labels = transfer_torch_data(train_features, train_labels, test_features, test_labels)


model = nn.Sequential(
    nn.Linear(8, 128),
    nn.Sigmoid(),
    nn.Linear(128, 1)
)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

epochs = 2000
for each_epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(train_features)
    loss_value = loss(predictions, train_labels)
    loss_value.backward()
    optimizer.step()
    if each_epoch % 100 == 0:
        print(f"epoch: {each_epoch}, loss: {loss_value.data}")

model.eval()

torch.save(model, 'torch.model')
y_pred = model(test_features).detach().numpy()

# 输入数据，预测
a = np.array([0, 230, 2.0, 2015, 4.4, 0, 24.18, 0])
b = (a - mean_)/np.sqrt(var_)
b = torch.from_numpy(b).float()
y_b = model(b).detach().numpy()

print(y_b)



