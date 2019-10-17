# encoding=utf-8

import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import numpy as np

df = pd.read_csv('t_car.csv')
# df = df.fillna(df.mean())  # 均值替换
df = df.dropna(axis=0, how='any')  # 删除空值数据 axis=0行，axis=1列
df = df.drop(['title'], axis=1)
print(df.info())
a = ['gear_type']  # 需要OneHotEncoder的feature

# Create a label encoder object
le = preprocessing.LabelEncoder()
le_count = 0

# 将数量大于10的列以Label编码
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

print('%d columns were label encoded.' % le_count)
gear_type = df['gear_type']
df = df.drop(['gear_type'], axis=1)
print(df.columns)
# 数据归一化处理:(原值-平均值)/数据标准差
def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean())/column.std())
df = normalize_feature(df)

y_data = np.array(df[df.columns[-1]]).reshape(len(df), 1)
df = pd.concat([gear_type, df[df.columns[0:7]]], axis=1)
datax = pd.get_dummies(df[a])  # one_hot
df = df.join(datax)
df = df.drop(a, axis=1)
X_data = np.array(df)

print(X_data.shape, type(X_data))
print(y_data.shape, type(y_data))

alpha = 0.01 # 学习率 alpha
epoch = 500 # 训练全量数据集的轮数

with tf.name_scope('input'):
    # 输入 X，形状[47, 3]
    X = tf.placeholder(tf.float32, X_data.shape, name='X')
    # 输出 y，形状[47, 1]
    y = tf.placeholder(tf.float32, y_data.shape, name='y')

with tf.name_scope('hypothesis'):
    # 权重变量 W，形状[3,1]
    W = tf.get_variable("weights",
                        (X_data.shape[1], 1),
                        initializer=tf.constant_initializer())
    b = tf.get_variable('b', (y_data.shape[1], 1), initializer=tf.constant_initializer())
    # 假设函数 h(x) = w0*x0+w1*x1+w2*x2, 其中x0恒为1
    # 推理值 y_pred  形状[47,1]
    y_pred = tf.matmul(X, W, name='y_pred') + b

with tf.name_scope('loss'):
    # 损失函数采用最小二乘法，y_pred - y 是形如[47, 1]的向量。
    # tf.matmul(a,b,transpose_a=True) 表示：矩阵a的转置乘矩阵b，即 [1,47] X [47,1]
    # 损失函数操作 loss
    loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)
with tf.name_scope('train'):
    # 随机梯度下降优化器 opt
    train_op = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss_op)

with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    # 记录所有损失值
    loss_data = []
    # 开始训练模型
    # 因为训练集较小，所以采用批梯度下降优化算法，每次都使用全量数据训练
    for e in range(1, epoch + 1):
        _, loss, w ,b_ = sess.run([train_op, loss_op, W, b], feed_dict={X: X_data, y: y_data})
        # 记录每一轮损失值变化情况
        loss_data.append(float(loss))
        if e % 10 == 0:
            log_str = "Epoch %d \t Loss=%.4g \t Model: y = %.4gx1 + %.4gx2 + %.4gx3 + %.4gx4 + %.4gx5 + %.4gx6 + %.4gx7 + %.4gx8, + %.4gx9 + %.4g"
            print(log_str % (e, loss, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], b_[0]))

