from sklearn.linear_model import LinearRegression

##线性回归初步使用

##数据
x = [[80, 86],
     [82, 80],
     [85, 78],
     [90, 90],
     [86, 82],
     [82, 90],
     [78, 80],
     [92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

##模型预估

model = LinearRegression()

##训练模型

model.fit(x, y)

##预测结果
print(model.predict([[80, 100]]))

print("线性回归相关系数 : ", model.coef_)
