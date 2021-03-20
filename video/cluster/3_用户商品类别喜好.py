import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1.获取数据
order_product = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/cluster/instacart/order_products__prior.csv")
products = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/cluster/instacart/products.csv")
orders = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/cluster/instacart/orders.csv")
aisles = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/cluster/instacart/aisles.csv")
# 2.数据基本处理
# 2.1 合并表格
temp_table1 = pd.merge(order_product, products, on=['product_id', 'product_id'])
temp_table2 = pd.merge(temp_table1, orders, on=['order_id', 'order_id'])
table = pd.merge(temp_table2, aisles, on=['aisle_id', 'aisle_id'])
print(table.head())
print(table.shape)
# 2.2 交叉表合并
data = pd.crosstab(table['user_id'], table['aisle'])
print(data.head())
print(data.shape)
# 2.3 数据截取
new_data = data[:1000]
# 3.特征工程 — pca
pca = PCA(n_components=0.8)
train_data = pca.fit_transform(new_data)
# 4.机器学习（k-means）
model = KMeans(n_clusters=5)
pre = model.fit_predict(train_data)
print("聚类结果: \n", pre)
# 5.模型评估
# sklearn.metrics.silhouette_score(X, labels)
#  计算所有样本的平均轮廓系数
# X：特征值
# labels：被聚类标记的目标值
result = silhouette_score(train_data, pre)
print("模型评估 : \n", result)
