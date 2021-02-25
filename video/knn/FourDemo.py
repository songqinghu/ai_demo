#鸢尾花种类预测—流程实现
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#获取数据集
iris = load_iris()
#数据基本处理
train_data,test_data,train_target,test_target = train_test_split(iris.data,iris.target,test_size=0.3,random_state=20)
#特征工程
transfer = StandardScaler()
train_data_format = transfer.fit_transform(train_data)
test_data_format = transfer.fit_transform(test_data)
#机器学习(模型训练)
model = KNeighborsClassifier(n_neighbors=6)
model.fit(train_data_format,train_target)
#模型评估
test_target_predict = model.predict(test_data_format)
print("模型预测结果为:\n",test_target_predict)
print("对比真实结果和预测值:\n",test_target == test_target_predict)
##模型预测准确率
score = model.score(test_data_format,test_target)
print("模型预测准确率: ",score)