#encoding=utf-8
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('t_car_last.csv')
print(type(df['city']))
df['num'] = 1
sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']})
# city = df[['num', 'city']].groupby(['city']).sum()
# sns.barplot(x='city', y='num', data=df, estimator=sum)
# plt.show()
#
# sns.jointplot(x='launch_year', y='price', data=df)
# plt.show()
#
# sns.barplot(x='is_import', y='num', data=df, estimator=sum)
# plt.show()

sns.barplot(x='gear_type', y='num', data=df, estimator=sum)
plt.show()


import re

df_init = pd.read_csv('../t_car.csv')
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
df_one_process['output_volume'] = df_one_process['output_volume'].astype('float')

#Partition interval for kilometres
bin = [0, 5, 10, 15, 20]
kilometres_step = pd.cut(df_one_process['kilometres'], bin)
df_one_process['kilometres_step'] = kilometres_step


画品牌直方图,一共300多个品牌，取前50名数量最多的
brand_groups = df_one_process.groupby(['brand']).count().nlargest(50, 'city')
print(brand_groups)
brand_groups['city'].plot('bar')
plt.show()

画里程分布直方图
kilometres_groups = df_one_process.groupby(['kilometres_step']).count()
kilometres_groups['city'].plot('bar')
plt.show()


画年份直方图
year_groups = df_one_process.groupby(['launch_year']).count()
year_groups['city'].plot('bar')
plt.show()

画手动档自动挡直方图
gear_groups = df_one_process.groupby(['gear_type']).count()
gear_groups['city'].plot('bar')
plt.show()

画进口与否直方图
import_groups = df_one_process.groupby(['is_import']).count()
import_groups['city'].plot('bar')
plt.show()
