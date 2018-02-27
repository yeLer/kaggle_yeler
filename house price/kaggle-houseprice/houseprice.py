import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
# 1.读取训练集和测试集合
train_df = pd.read_csv('input/train.csv',index_col=0)
test_df = pd.read_csv('input/test.csv',index_col=0)
# 打印前5行数据
# print(train_df.head())

# 2.合并数据--主要是为了用DF进行数据预处理的时候更加方便。等所有的需要的预处理进行完之后，我们再把他们分隔开。
#   2.1绘制直方图信息
#   数据预处理
# prices = pd.DataFrame({"price":train_df["SalePrice"],"log(price+1)":np.log1p(train_df["SalePrice"])})
# #   绘制直方图
# prices.hist()
# #   显示数据
# plt.show()

# 2.2数据处理阶段,去掉训练集中销售价格（y）部分的数据,y_train则是SalePrice那一列
y_train = np.log1p(train_df.pop('SalePrice'))
# 将剩余的训练集部分与测试集合并
all_df = pd.concat((train_df,test_df),axis=0)
# 打印合并后的数据集合的维度
# print(all_df.shape)

# 3.变量转化--类似『特征工程』。就是把不方便处理或者不unify的数据给统一了。
# 3.1转变数据类型
# print(all_df['MSSubClass'].dtype)  # int64的数据类型
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str) #将数据转成str
# 3.2转变数据类别
# print(all_df['MSSubClass'].value_counts())
# 把category的变量转变成numerical表达形式(pandas自带的get_dummies方法，可以帮你一键做到One-Hot。)
pd.get_dummies(all_df['MSSubClass'],prefix='MSSubClass')
# 同理,我们把所有的category数据,都给One-Hot了
all_dummy_df = pd.get_dummies(all_df)
# print(all_dummy_df.head())
# 3.3解决数据缺失
#   查看数据缺失的情况
# (处理缺失数据的原则)
#   1缺失太少，使用平均值或者常用值代替)
#   2缺失中等，新增一个数据表示
#   3缺失过多，抛弃该数据
# print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))
# 这里全部使用均值填充
mean_cols =all_dummy_df.mean()
# print(mean_cols.head(10))
all_dummy_df = all_dummy_df.fillna(mean_cols)
# 检验是否存在空缺值
# print(all_dummy_df.isnull().sum().sum())
# 3.4标准化numerical数据--把源数据给放在一个标准分布内。不要让数据间的差距太大。
# 查看哪些是numerical的：
numeric_cols = all_df.columns[all_df.dtypes!='object']
# print(numeric_cols)
# 这里也是可以继续使用Log的，我只是给大家展示一下多种“使数据平滑”的办法。
numeric_cols_mean = all_dummy_df.loc[:,numeric_cols].mean()
numeric_cols_std = all_dummy_df.loc[:,numeric_cols].std()
all_dummy_df.loc[:,numeric_cols] = (all_dummy_df.loc[:,numeric_cols]-numeric_cols_mean)/numeric_cols_std
# 将混合的数据分开成训练集和测试集，依据索引分离
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
# print(dummy_train_df.shape,dummy_test_df.shape)
# .....数据预处理完毕.......

# 4.建立模型
# 把DF转化成Numpy Array，这跟Sklearn更加配
X_train = dummy_train_df.values
X_test = dummy_test_df.values
# 用Sklearn自带的cross validation方法来测试模型
# numpy.linspace用于创建等差数列，现在介绍logspac用于创建等比数列  起始值0.001  结束值100  总共50个数
alphas = np.logspace(-3, 2, 50)
# print(alphas)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
# 存下所有的CV值，看看哪个alpha值更好（也就是『调参数』）
# plt.plot(alphas, test_scores)
# plt.title("Alpha vs CV Error")
# plt.show()
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
# plt.plot(max_features, test_scores)
# plt.title("Max Features vs CV Error");
# plt.show()
# 5.用一个Stacking的思维来汲取两种或者多种模型的优点
ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)
ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)
y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))
y_final = (y_ridge + y_rf) / 2
submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_final})
#将DataFrame存储为csv,index表示是否显示行名，default=True
submission_df.to_csv("submision.csv",index=False,sep=',')
