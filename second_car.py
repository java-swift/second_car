#encoding=utf-8

from sklearn import preprocessing

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

df = pd.read_csv('t_car.csv')
# df = df.fillna(df.mean())  # 均值替换
df = df.dropna(axis=0, how='any')  # 删除空值数据 axis=0行，axis=1列
df = df.drop(['title'], axis=1)
print(df.info())
# vex = DictVectorizer()
# df = vex.fit_transform(df)
# encoder.fit_transform()

# city, brand, gear_type, output_volume需要转换成数值类型 LabelEncoder或OneHotEncoder
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

datax = pd.get_dummies(df[a])  # one_hot
df = df.join(datax)
df = df.drop(a, axis=1)
y_df = df.price
X_df = df.drop(['price'], axis=1)  # 删除标签
x_train, x_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.9, random_state=1)

print(x_train)
# model = XGBRegressor()
model = LinearRegression()  # 线性回归模型
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 兰州	雪佛兰	2.4L	2016	8.9	0	20.61	自动挡	8.1
# 上海	福特	1.8L	2009	5	0	14.64	自动挡	3.2
print(len(le.classes_))
# 待预测的数据feature 下例真实值3.2
t = ['上海', '福特',	'1.8L',	2009,	5, 0, 14.64, 0, 1]
output_volume_l = label_code_map['output_volume']
city_l = label_code_map['city']
brand_l = label_code_map['brand']


# 非数值类型数据转换为数值类型（以原转换为准则）
def re_assign(index, ls, tbd):
    for i in range(len(ls)):
        tbd[index] = i + 1
        if ls[i] == t[index]:
            tbd[index] = i
            break


re_assign(2, output_volume_l, t)
re_assign(0, city_l, t)
re_assign(1, brand_l, t)

tu = tuple(t)
t = list()
t.append(tu)
t = pd.DataFrame(t, columns=['city', 'brand', 'output_volume', 'launch_year', 'kilometres', 'is_import', 'old_price', 'gear_type_手动挡', 'gear_type_自动挡'])
print(t)
# print(model.coef_)  # 线性回归参数
# print(model.intercept_)  #线性回归参数
# t = xgb.DMatrix(t)
y_ = model.predict(t)
print(y_)
score = model.score(x_test, y_test)
print(score)
# print(df.columns)
