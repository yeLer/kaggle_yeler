import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpb
mpb.rcParams['font.sans-serif'] = ['SimHei']

data_train = pd.read_csv('input/train.csv')
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
# print(df)

g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
# print(df)
# print(data_train.Cabin.value_counts())

#cabin的值计数太分散了，绝大多数Cabin值只出现一次。感觉上作为类目，加入特征未必会有效
#那我们一起看看这个值的有无，对于survival的分布状况，影响如何吧
# 设定图表颜色的alpha参数
Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({u'有':Survived_cabin,u'无':Survived_nocabin}).transpose()
df.plot(kind = 'bar',stacked=True)
plt.title(u'按Cabin有无看获救情况')
plt.xlabel(u'Cabin有无')
plt.ylabel(u'人数')
plt.show()