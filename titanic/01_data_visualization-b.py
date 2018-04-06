import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
# 解决matplotlib绘图中文乱码问题
import matplotlib as mpb
mpb.rcParams['font.sans-serif'] = ['SimHei']

data_train = pd.read_csv('input/train.csv')
# print(data_train.columns)
# print(data_train.info())
# print(data_train.describe())
# 定义figure
# fig = plt.figure()
# # 设定图表颜色alpha参数
# fig.set(alpha=0.2)
Survived_0 = data_train.Pclass[data_train.Survived==0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived==1].value_counts()
df = pd.DataFrame({'saved':Survived_1,'unsaved':Survived_0})
# stacked=True使得图形拼接到一起
df.plot(kind='bar',stacked=True)
plt.title('Rescue status of all passengers with diff ranks')
plt.xlabel('passengers with diff ranks')
plt.ylabel('passenger num')
# 看看各登录港口的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)
Survived_0 = data_train.Embarked[data_train.Survived==0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived==1].value_counts()
df = pd.DataFrame({'saved':Survived_1,'unsaved':Survived_0})
df.plot(kind='bar',stacked=True)
plt.title('The rescue of passengers on each landing port')
plt.xlabel('landing port')
plt.ylabel('passenger num')
#看看各性别的获救情况
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
df.plot(kind='bar',stacked=True)
plt.title('The rescue of passengers on each gender')
plt.xlabel('性别')
plt.ylabel('passenger num')
plt.show()

#然后我们再来看看各种舱级别情况下各性别的获救情况
fig = plt.figure()
# 设置图像的透明度，可选
fig.set(alpha=0.65)
plt.title(u"根据舱等级和性别的获救情况")
ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
# rotation=30表示x的标注旋转30度
ax1.set_xticklabels([u"获救", u"未获救"], rotation=30)
ax1.legend([u"女性/高级舱"], loc='best')
# sharey=ax1子图共享坐标轴
ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')
plt.show()