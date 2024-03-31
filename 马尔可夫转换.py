import numpy as np
import pandas as pd

from hmmlearn import hmm
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(15, 10)})# 设置Seaborn的绘图风格和图形尺寸。

np.random.seed(42069)# 设置随机种子，以确保代码的可重复性

import warnings; warnings.simplefilter('ignore')

'''
知识介绍
https://www.adeveloperdiary.com/data-science/machine-learning/introduction-to-hidden-markov-model/
https://blog.csdn.net/weixin_51130521/article/details/119494594?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171128556416777224455710%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171128556416777224455710&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-119494594-null-null.142^v99^pc_search_result_base9&utm_term=%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB&spm=1018.2226.3001.4187
代码参考
https://github.com/LouisSugunasabesan/Volatility-Modelling-Using-HiddenMarkovModels?tab=readme-ov-file
'''


# 定义包含数据的文件路径，单独运行下面的文件一二三，模型是公用的
#%%文件一，数据量太少，无法得到合适的结果
DATA_PATH = "D:\.a桌面\工作\国泰君安\宏观时钟\宏观时钟\资产化因子.xlsx"

df = pd.read_excel(DATA_PATH, index_col=0, parse_dates=True)# 设置日期索引
df.sort_index()

df.drop(['Inflation', 'IntRate', 'Credit', 'ExchRate','Liquidity'],axis=1, inplace=True)
df.columns = ['Close']# 列重命名为“Close”。

nullvaluecheck = pd.DataFrame(df.isna().sum().sort_values(ascending=False)*100/df.shape[0],columns=['missing %']).head(60)
nullvaluecheck.style.background_gradient(cmap='PuBu')# 检查数据帧中的缺失值并显示。
#%%文件二，黄金数据测试
DATA_PATH = "D:\.a桌面\工作\国泰君安\Volatility-Modelling-Using-HiddenMarkovModels-main/GLD.csv"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)# 设置日期索引
df.sort_index()

df.drop([' Volume', ' Open', ' High', ' Low'],axis=1, inplace=True)
df.columns = ['Close']# 列重命名为“Close”。

nullvaluecheck = pd.DataFrame(df.isna().sum().sort_values(ascending=False)*100/df.shape[0],columns=['missing %']).head(60)
nullvaluecheck.style.background_gradient(cmap='PuBu')# 检查数据帧中的缺失值并显示。
#%%文件三，工业数据测试
DATA_PATH = "D:\.a桌面\工作\国泰君安\马尔可夫\美国_道琼斯工业平均指数.xlsx"

df = pd.read_excel(DATA_PATH, index_col=0, parse_dates=True)# 设置日期索引
df.sort_index()

#df.drop(['Inflation', 'IntRate', 'Credit', 'ExchRate','Liquidity'],axis=1, inplace=True)
df.columns = ['Close']# 列重命名为“Close”。

nullvaluecheck = pd.DataFrame(df.isna().sum().sort_values(ascending=False)*100/df.shape[0],columns=['missing %']).head(60)
nullvaluecheck.style.background_gradient(cmap='PuBu')# 检查数据帧中的缺失值并显示。
#%%公共建模部分

returns = np.log(df['Close']).diff()# 对数差分

returns.dropna(inplace=True)# 删除NaN值。

returns.plot(kind='hist',bins=150)
plt.title(label='Distribution of Growth Returns', size=15)
plt.show()# 绘制分布的直方图。

split = int(0.2*len(returns))
X = returns[:-split]
X_test = returns[-split:]# 将数据集分为训练集和测试集。

pd.DataFrame(X).plot()
plt.title(label='Growth Training Set', size=15)
plt.show()# 绘制训练集中的Growth。

pd.DataFrame(X_test).plot()
plt.title(label='Growth Testing Set', size=15)
plt.show()# 绘制测试集中的Growth。

X = X.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy().reshape(-1, 1)# 将训练集和测试集转换为NumPy数组并重塑形状以满足模型的需求。
# 创建一个高斯HMM模型，指定2个隐藏状态并使用对角协方差。
model = hmm.GaussianHMM(n_components=2, covariance_type="diag", verbose=True)
'''
高斯HMM是指在隐马尔可夫模型中，观测数据的概率分布是高斯分布（正态分布）。
指定了模型中的隐藏状态数量为2，在这个模型中，系统在任何时刻都可以处于两种不同的状态之一。
'''

model.transmat_ = np.array([
                            [0.8, 0.2],
                            [0.2, 0.8]
                           ])# 设置状态转移矩阵。
model.fit(X)


Z = model.predict(X_test)# 对测试集和训练集进行状态预测。
Z_train = model.predict(X)

# 计算训练状态转换
returns_train0 = np.empty(len(Z_train))
returns_train1 = np.empty(len(Z_train))
returns_train0[:] = np.nan
returns_train1[:] = np.nan# 为每个状态更改创建序列。

# 创建储存
returns_train0[Z_train == 0] = returns[:-split][Z_train == 0]
returns_train1[Z_train == 1] = returns[:-split][Z_train == 1]

# 作图
fig, ax = plt.subplots(figsize=(15,10))

plt.subplot(211)
plt.plot(Z)
plt.title(label='Growth Training Volatility Regime', size=15)

plt.subplot(212)
plt.plot(returns_train0, label='State_0 (High Volatility)', color='r')
plt.plot(returns_train1, label='State_1 (Low Volatility)', color='b', )
plt.title(label='Growth Training Volatility Clusters', size=15)
plt.legend()
plt.tight_layout()


# 计算测试状态转换
returns0 = np.empty(len(Z))
returns1 = np.empty(len(Z))
returns0[:] = np.nan
returns1[:] = np.nan

# 创建储存
returns0[Z == 0] = returns[-split:][Z == 0]
returns1[Z == 1] = returns[-split:][Z == 1]

# 作图
fig, ax = plt.subplots(figsize=(15,10))

plt.subplot(211)
plt.plot(Z)
plt.title(label='Growth Volatility Regime', size=15)

plt.subplot(212)
plt.plot(returns0, label='State_0 (High Volatility)', color='r')
plt.plot(returns1, label='State_1 (Low Volatility)', color='b')
plt.title(label='Growth Volatility Clusters', size=15)

plt.legend()
plt.tight_layout()




