import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


def var_demo():
    # 读取数据
    data = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/cluster/factor_returns.csv")

    print(data.head())
    print(data.shape)
    # 实例化转化器和设置阀值
    transform = VarianceThreshold(threshold=1)
    data = transform.fit_transform(data.iloc[:, 1:10])
    print("新特征: \n", data)
    print(data.shape)


var_demo()


def pear_demo():
    # 皮尔逊相关系数
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]
    result = pearsonr(x1, x2)
    print(result)


pear_demo()


def spea_demo():
    # 斯皮尔曼相关系数
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]
    result = spearmanr(x1, x2)
    print(result)


spea_demo()


def pca_dmeo():
    # 对数据进行PCA降维
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]

    pca = PCA(n_components=0.9)
    data1 = pca.fit_transform(data)
    print("保留90%的信息 : \n", data1)

    pca1 = PCA(n_components=3)
    data2 = pca1.fit_transform(data)
    print("保留3个特征 : \n", data2)


pca_dmeo()
