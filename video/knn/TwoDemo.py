from sklearn.datasets import load_iris
from sklearn.datasets import fetch_20newsgroups
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import  train_test_split

## 小数据集
#iris = load_iris()
#print(iris)
##大数据集
#news = fetch_20newsgroups()
#print(news)
##加载鸢尾花数据集
iris = load_iris()
##获取属性
print("鸢尾花的特征值:\n", iris["data"])
print("鸢尾花的目标值：\n", iris.target)
print("鸢尾花特征的名字：\n", iris.feature_names)
#print("鸢尾花目标值的名字：\n", iris.target_names)
#print("鸢尾花的描述：\n", iris.DESCR)

##查看数据集分布
iris_d = pd.DataFrame(iris['data'],columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris_d['Species'] = iris.target
#print(iris_d)

def plot_iris(iris,col1,col2):
    sns.lmplot(x=col1,y=col2,data=iris,hue='Species',fit_reg=False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("鸢尾花种类分布图")
    plt.show()

plot_iris(iris_d,'Petal_Width', 'Sepal_Length')

###数据集的划分
x_train,y_train,x_terget,y_terget = train_test_split(iris['data'],iris.target,random_state=22)
print("训练集数据:",x_train)
print("训练集目标:",x_terget)
print("测试集数据:",y_train)
print("测试集目标:",y_terget)