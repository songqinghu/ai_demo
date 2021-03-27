import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

##读取数据
data = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/svm/train.csv")
print(data.head())
print(data.shape)
##确定特征值目标值
x = data.iloc[:, 1:]
y = data.iloc[:, 0]
print(x.head())
print(y.head())


##展示图像效果
def show_image(n):
    tmp = x.iloc[n,].values.reshape(28, 28)
    plt.imshow(tmp)
    plt.axis("off")
    plt.show()


n = 40
print(y[n])
show_image(n)

##数据处理 将数字变为0,1形式
x = x.values / 255
y = y.values
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=0)

print(train_x.shape, test_x.shape)


##特征降维和模型训练
# 多次使用pca,确定最后的最优模型

def get_max_good_pca(n, train_x, train_y, test_x, test_y):
    start_time = time.time()
    print("特征降维,传递的参数为:{}".format(n))
    pca = PCA(n_components=n)
    pca.fit(train_x)
    pca_train_x = pca.transform(train_x)
    pca_test_x = pca.transform(test_x)

    print("使用svm进行模型训练")
    svc = svm.SVC()
    svc.fit(pca_train_x, train_y)
    ##预测准确率
    accuracy = svc.score(pca_test_x, test_y)
    end_time = time.time()
    print("准确率 {}  消耗时间 {} ".format(accuracy, (end_time - start_time)))
    return accuracy


n_s = np.linspace(0.70, 0.85, num=5)
accuracys = []
for n in n_s:
    accuracy = get_max_good_pca(n, train_x, train_y, test_x, test_y)
    accuracys.append(accuracy)
print(n_s)
print(accuracys)

##可视化展示效果

plt.plot(n_s, np.array(accuracys), "r")
plt.show()

##确定最优模型
pca = PCA(n_components=0.80)
pca.fit(train_x)
print(pca.n_components_)
pca_train_x = pca.transform(train_x)
pca_test_x = pca.transform(test_x)
print(pca_train_x.shape, pca_test_x.shape)
svc = svm.SVC()
svc.fit(pca_train_x, train_y)
print(svc.score(pca_test_x, test_y))
