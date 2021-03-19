import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

##读取数据展示
data = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/tree/train.csv")
print(data.head())
# 2.数据基本处理

# 2.1 确定特征值,目标值
x = data[["Pclass", "Age", "Sex"]]
y = data["Survived"]
# 2.2 缺失值处理  age数据存在为空的情况
x['Age'].fillna(x['Age'].mean(), inplace=True)
# 2.3 分割数据集
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=20)

# 3.特征工程(字典特征抽取)
# 特征中出现类别符号，需要进行one-hot编码处理(DictVectorizer)
# train_x.to_dict(orient="records")  #需要将数组特征转换成字典数据

# 3.1 数据标准化
standard = DictVectorizer()
train_x = standard.fit_transform(train_x.to_dict(orient="records"))
test_x = standard.fit_transform(test_x.to_dict(orient="records"))

# 3.2 随机森林训练和模型评估

#
rf = RandomForestClassifier()
# 定义参数列表
param = {"n_estimators": [100, 120, 300], "max_depth": [3, 7, 11]}
# 使用网格搜索
model = GridSearchCV(rf, param_grid=param, cv=2)
model.fit(train_x, train_y)
pre_y = model.predict(test_x)
print("模型预测结果 : \n", pre_y)
print("模型准确率打分 : \n", model.score(test_x, test_y))
