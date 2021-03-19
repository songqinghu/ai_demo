import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 读取数据
data = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/otto/train.csv")
print(data.head())
print(data.shape)
print(data.describe())
##查看数据分布 不均匀
sns.countplot(data.target)
plt.show()
##截取部分数据
new_data = data[:10000]
print(new_data.shape)
##截取数据做图
sns.countplot(new_data.target)
plt.show()
##效果不好.丢失很多分类且数据分布还是不均匀
# 随机欠采样获取数据
# 首先需要确定特征值\标签值
y = data['target']
x = data.drop(['id', 'target'], axis=1)
print(x.head())
ru = RandomUnderSampler(random_state=0)
x_sampler, y_sampler = ru.fit_resample(x, y)
##看看效果
print(x_sampler.shape)
print(y_sampler.shape)
sns.countplot(y_sampler)
plt.show()
# 把标签值转为数字
label = LabelEncoder()
y_sampler = label.fit_transform(y_sampler)
print(y_sampler)
# 分割数据
train_x, test_x, train_y, test_y = train_test_split(x_sampler, y_sampler)
# 数据标准化  --  已经是处理过的数据了

# 模型训练
model = RandomForestClassifier(oob_score=True)
model.fit(train_x, train_y)
# 预测结果和计算准确率
pre_y = model.predict(test_x)
print("预测结果为 : \n", pre_y)
print("模型准确率为: \n", model.score(test_x, test_y))
# 预测结果数据分布
sns.countplot(pre_y)
plt.show()

# logloss模型评估 one - hot 编码

ohe = OneHotEncoder(sparse=False)

test_y1 = ohe.fit_transform(test_y.reshape(-1, 1))
pre_y1 = ohe.fit_transform(pre_y.reshape(-1, 1))

print(log_loss(test_y1, pre_y1, eps=1e-15, normalize=True))
# 改变预测值的输出模式,让输出结果为百分占比,降低logloss值
y_pre_proba = model.predict_proba(test_x)
print(log_loss(test_y1, y_pre_proba, eps=1e-15, normalize=True))
# 模型调优  n_estimators, max_feature, max_depth, min_samples_leaf

# 确定最优 n_estimators

estimators_param = range(10, 200, 10)
# 精度度
accuracy_t = np.zeros(len(estimators_param))
# 错误率
error_t = np.zeros(len(estimators_param))

for i, estimator in enumerate(estimators_param):
    rfc = RandomForestClassifier(n_estimators=estimator,
                                 max_depth=10, max_features=10,
                                 min_samples_leaf=10, oob_score=True, random_state=0, n_jobs=-1)
    rfc.fit(train_x, train_y)
    accuracy_t[i] = rfc.oob_score_
    ##计算logloss
    e_prd_pro_y = rfc.predict_proba(test_x)
    error_t[i] = log_loss(test_y1, e_prd_pro_y, eps=1e-15, normalize=True)
    print(error_t)

# 将数据可视化展示效果

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4), dpi=100)

axes[0].plot(estimators_param, accuracy_t)
axes[1].plot(estimators_param, error_t)

axes[0].set_xlabel("n_estimators")
axes[0].set_ylabel("accuracy_t")
axes[1].set_xlabel("n_estimators")
axes[1].set_ylabel("error_t")

axes[0].grid(True)
axes[1].grid(True)

plt.show()

##看图知道选取 n_estimators = 175效果较好.后续几个参数按照此方法调试即可
