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


##below:wangpei code 
import re
from matplotlib.font_manager import FontProperties



def get_chinese_font(): ###just for mac os 
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

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

plt.figure(figsize=(20, 30))
ax1 = plt.subplot(3,2,1)
ax2 = plt.subplot(3,2,2)
ax3 = plt.subplot(3,2,3)
ax4 = plt.subplot(3,2,4)
ax5 = plt.subplot(3,2,5)

#选择ax1
plt.sca(ax1)
#画品牌直方图,一共300多个品牌，取前50名数量最多的
brand_groups = df_one_process.groupby(['brand']).count().nlargest(30, 'city')
# print(brand_groups)
brand_groups['city'].plot('bar')
plt.xticks(rotation=90, fontproperties=get_chinese_font())
plt.xlabel(u'品牌', fontproperties=get_chinese_font())
plt.ylabel(u'数量', fontproperties=get_chinese_font())
plt.title(u'品牌数量排名前30', fontproperties=get_chinese_font())
ax_trick=plt.gcf()
ax_trick.patch.set_color("white")

plt.sca(ax2)
#画里程分布直方图
kilometres_groups = df_one_process.groupby(['kilometres_step']).count()
kilometres_groups['city'].plot('bar')
plt.xticks(rotation=90, fontproperties=get_chinese_font())
plt.xlabel(u'已行驶里程', fontproperties=get_chinese_font())
plt.ylabel(u'数量', fontproperties=get_chinese_font())
plt.title(u'已行驶里程', fontproperties=get_chinese_font())
ax_trick=plt.gcf()
ax_trick.patch.set_color("white")

plt.sca(ax3)
#画发行年份分布直方图
year_groups = df_one_process.groupby(['launch_year']).count()
year_groups['city'].plot('bar')
plt.xticks(rotation=90, fontproperties=get_chinese_font())
plt.xlabel(u'发行年份', fontproperties=get_chinese_font())
plt.ylabel(u'数量', fontproperties=get_chinese_font())
plt.title(u'发行年份', fontproperties=get_chinese_font())
ax_trick=plt.gcf()
ax_trick.patch.set_color("white")

plt.sca(ax4)
#画手动档自动挡分布直方图
gear_groups = df_one_process.groupby(['gear_type']).count()
gear_groups['city'].plot('bar')
plt.xticks(rotation=90, fontproperties=get_chinese_font())
plt.xlabel(u'自动手动挡', fontproperties=get_chinese_font())
plt.ylabel(u'数量', fontproperties=get_chinese_font())
plt.title(u'自动手动挡', fontproperties=get_chinese_font())
ax_trick=plt.gcf()
ax_trick.patch.set_color("white")

plt.sca(ax5)
#画进口与否分布直方图
gear_groups = df_one_process.groupby(['is_import']).count()
gear_groups['city'].plot('bar')
plt.xticks(rotation=90, fontproperties=get_chinese_font())
plt.ylabel(u'进口与否', fontproperties=get_chinese_font())
plt.ylabel(u'数量', fontproperties=get_chinese_font())
plt.title(u'进口与否', fontproperties=get_chinese_font())
ax_trick=plt.gcf()
ax_trick.patch.set_color("white")

# plt.show()
plt.savefig("./bar.png")
