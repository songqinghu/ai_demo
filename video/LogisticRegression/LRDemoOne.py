import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##加载数据
names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
         'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
         'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
    names=names)
print(data.head())

##数据预处理
##缺失数据处理
data = data.replace(to_replace="?", value=np.NaN)
data = data.dropna()
##划分特征值和目标值
data_x = data.iloc[:, 1:-1]
data_y = data["Class"]
print(data_x.head())
print(data_y.head())

##划分数据集
train_data, test_data, train_target, test_target = train_test_split(data_x, data_y, random_state=22, test_size=0.2)

##数据标准化
standard = StandardScaler()
train_data = standard.fit_transform(train_data)
test_data = standard.fit_transform(test_data)

##模型训练
model = LogisticRegression()
model.fit(train_data, train_target)

##模型评价
print(model.score(test_data, test_target))
test_predict = model.predict(test_data)
print(test_predict)

##模型评估
ret = classification_report(test_target, test_predict, labels=(2, 4), target_names=("良性", "恶性"))
print(ret)
##AUC指标 -- 原始只能是0,1
auc_test_target = np.where(test_target > 2, 1, 0)

auc = roc_auc_score(test_target, test_predict)

print("auc结果:\n", auc)
