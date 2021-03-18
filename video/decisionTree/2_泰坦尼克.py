import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

##读取数据展示
data = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/tree/train.csv")
print(data.head())
# 2.数据基本处理

# 2.1 确定特征值,目标值
x = data[["Pclass", "Age", "Sex"]]
y = data["Survived"]
print(x)
print(y.head())
# 2.2 缺失值处理  age数据存在为空的情况
print(x[x['Age'].isna()])
# print(x.describe())
x['Age'].fillna(x['Age'].mean(), inplace=True)
print(x)
# 2.3 分割数据集
train_x, test_x, train_y, test_y = train_test_split(x, y)

# 3.特征工程(字典特征抽取)
# 特征中出现类别符号，需要进行one-hot编码处理(DictVectorizer)
# train_x.to_dict(orient="records")  #需要将数组特征转换成字典数据

# 3.1 数据标准化
standard = DictVectorizer()
train_x = standard.fit_transform(train_x.to_dict(orient="records"))
test_x = standard.fit_transform(test_x.to_dict(orient="records"))

# 3.2.决策树模型训练和模型评估

# 决策树API当中，如果没有指定max_depth那么会根据信息熵的条件直到最终结束。这里我们可以指定树的深度来进行限制树的大小
model = DecisionTreeClassifier(criterion="entropy", max_depth=5)
model.fit(train_x, train_y)
pre_y = model.predict(test_x)
print("模型预测结果 : \n", pre_y)
print("模型准确率打分 : \n", model.score(test_x, test_y))

# 4 决策树可视化
# 4.1 保存树的结构到dot文件
# sklearn.tree.export_graphviz()
export_graphviz(model, out_file='./tree.dot', feature_names=standard.get_feature_names())
## 效果查看 https://dreampuf.github.io/GraphvizOnline
