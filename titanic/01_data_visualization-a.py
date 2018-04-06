import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

data_train = pd.read_csv('input/train.csv')
# print(data_train.columns)
# print(data_train.info())
# print(data_train.describe())
# 定义figure
fig = plt.figure()
# 设定图表颜色alpha参数
fig.set(alpha=0.2)
# 在一张大图里面分列几个小图
plt.subplot2grid((2,3),(0,0))
# 绘制获救与未获救人员的柱状图
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u'0-death  1-saved')
plt.ylabel(u'person num')
# 绘制乘客等级人员分布图
plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.ylabel('person num')
plt.title('passenger rank')
# 绘制年龄与获救关系图
plt.subplot2grid((2,3),(0,2))
# scatter函数用于绘制散点图
plt.scatter(data_train.Survived,data_train.Age)
plt.ylabel('age')
plt.grid(b=True,which='major',axis='y')
plt.title('saved with age 1-saved')
# 绘制不同等级的乘客年龄与密度关系分布曲线图
# colspan=2表示合并两列
plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass==1].plot(kind='kde')
data_train.Age[data_train.Pclass==2].plot(kind='kde')
data_train.Age[data_train.Pclass==3].plot(kind='kde')
plt.xlabel('age')
plt.ylabel('desity')
plt.title('the distribution of different age with each rank')
plt.legend(('1first calss','2second class','3third calss'),loc='best')
# 各登船口岸上船人数
plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title('The number of passengers at each boarding port')
plt.ylabel('person num')
plt.show()