from sklearn.neighbors import KNeighborsClassifier

#构造数据
x = [[0],[1],[10],[20]]
y = [0,0,1,1]

##模型训练
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)
##预测数据
print(model.predict([[1]]))
print(model.predict([[100]]))