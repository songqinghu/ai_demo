import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs
from sklearn.metrics import calinski_harabasz_score

# 创建数据
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，
# 簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2, 0.2]
x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=22)

print("数据集:\n", x)

# 数据可视化 散点图
plt.scatter(x[:, 0], x[:, 1], marker='o')
plt.show()

##使用k-means进行聚类,并使用CH方法评估
for i in range(2, 5, 1):
    pre_y = KMeans(n_clusters=i, random_state=22).fit_predict(x)
    ##查看效果
    plt.scatter(x[:, 0], x[:, 1], c=pre_y)
    plt.show()
    ##评估
    print("当前聚类核心 : ", i)
    print(calinski_harabasz_score(x, pre_y))
