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
from sklearn.model_selection import GridSearchCV
import pymysql
import re

df_init = pd.read_csv('../t_car.csv')
# df = df.fillna(df.mean())  # 均值替换
df_init = df_init.dropna(axis=0, how='any')  # 删除空值数据 axis=0行，axis=1列
df_init = df_init.drop(['title'], axis=1)

###去除Unknown值
error_data = df_init[df_init['output_volume'].str.contains('Unknown')]
output_volume_error_data = list(error_data['output_volume'])
output_volume_all = list(df_init['output_volume'])
output_volume_ret = list(set(output_volume_all)^set(output_volume_error_data))
df_one_process = df_init[df_init.output_volume.isin(output_volume_ret)]

#extract output_volume real value
f1 = lambda x: re.findall(r'\d+\.?\d*', x)[0]
df_one_process['output_volume'] = df_one_process['output_volume'].map(f1)  ##output_volume unit is L

#process gear_type
f2 = lambda x: 0 if x == '自动挡' else 1
df_one_process['gear_type'] = df_one_process['gear_type'].map(f2)

#process output_volume
df_one_process['output_volume'] = df_one_process['output_volume'].astype('float')

df_last = df_one_process

#process city
# Create a label encoder object
city_le = preprocessing.LabelEncoder()
city_le.fit(df_last['city'])
city_list = df_last['city']
df_last['city'] = city_le.transform(city_list)
# print(str(city_le.inverse_transform([10, 20, 1])).decode("string_escape"))
# print(list(city_le.classes_).index('天津'))

#process band
# Create a label encoder object
brand_le = preprocessing.LabelEncoder()
brand_le.fit(df_last['brand'])
band_list = df_last['brand']
df_last['brand'] = brand_le.transform(band_list)
# print(str(brand_le.inverse_transform([116])).decode("string_escape"))
# print(list(brand_le.classes_).index('哈弗H9'))

y_train = df_last.price
x_train = df_last.drop(['price'], axis=1)  # 删除标签
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, shuffle=True)
# print(y_train)
model = XGBRegressor(max_depth=150, learning_rate=0.01, n_estimators=700,
                     silent=True, objective='reg:linear', booster='gblinear', n_jobs=50,
                     nthread=None, gamma=0.5, min_child_weight=7, max_delta_step=0, subsample=1,
                     colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                     scale_pos_weight=1, base_score=0.5, random_state=0, seed=None,
                     missing=None, importance_type='gain')
# model = LinearRegression()  # 线性回归模型
model.fit(x_train, y_train)
# model.save_model('xgboost.model')
cv_params = { # 'n_estimators': [500, 550, 600, 650, 700],
             'max_depth': [100, 150, 200, 250, 300, 350, 400, 450],
             #'min_child_weight': [6, 7, 8],
             # 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
             }
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=3, verbose=1, n_jobs=-1)
optimized_GBM.fit(x_train, y_train)
evalute_result = optimized_GBM.return_train_score
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
y_pred = model.predict(x_test)

# 兰州	雪佛兰	2.4L	2016	8.9	0	20.61	自动挡	8.1
# 上海	福特	1.8L	2009	5	0	14.64	自动挡	3.2
# 待预测的数据feature 下例真实值[8.1,3.2]
# t = ['兰州', '雪佛兰',	'2.4L',	2016,	8.9, 0, 20.61, 0]
# city_id = list(city_le.classes_).index('兰州')
# band_id = list(brand_le.classes_).index('雪佛兰')
# t = [city_id, band_id, 2.4, 2016, 8.9, 0, 20.61, 0]
t = ['上海', '福特',	'1.8',	2016,	8.9, 0, 20.61, 0]
city_id = list(city_le.classes_).index('上海')
band_id = list(brand_le.classes_).index('福特')
t = [city_id, band_id, 1.8,	2009, 5, 0, 14.64, 0]


tu = tuple(t)
t = list()
t.append(tu)
t = pd.DataFrame(t, columns=['city', 'brand', 'output_volume', 'launch_year', 'kilometres', 'is_import', 'old_price', 'gear_type'])

y_ = model.predict(t)
print(y_)

