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