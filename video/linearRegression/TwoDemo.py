##波士顿房价预测
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


##正规方式方式
def linear_model1():
    ##加载数据
    data = load_boston()
    # print(data)
    ##分割数据
    data_x, data_y, x_target, y_target = train_test_split(data.data, data.target, test_size=0.2)
    ##数据标准化
    standard = StandardScaler()
    data_x_format = standard.fit_transform(data_x)
    data_y_format = standard.fit_transform(data_y)
    ##模型训练
    model = LinearRegression()
    model.fit(data_x_format, x_target)
    print("特性参数:\n", model.coef_)
    ##模型预测
    y_predict = model.predict(data_y_format)
    # print("预测结果:\n",y_predict)
    ##模型评估
    data_error = mean_squared_error(y_target, y_predict)
    print("标准平方差:\n", data_error)


##梯度下降方式
def linear_model2():
    ##加载数据
    data = load_boston()
    # print(data)
    ##分割数据
    data_x, data_y, x_target, y_target = train_test_split(data.data, data.target, test_size=0.2)
    ##数据标准化
    standard = StandardScaler()
    data_x_format = standard.fit_transform(data_x)
    data_y_format = standard.fit_transform(data_y)
    ##模型训练
    model = SGDRegressor()
    model.fit(data_x_format, x_target)
    print("特性参数:\n", model.coef_)
    ##模型预测
    y_predict = model.predict(data_y_format)
    # print("预测结果:\n",y_predict)
    ##模型评估
    data_error = mean_squared_error(y_target, y_predict)
    print("标准平方差:\n", data_error)


##岭回归
def linear_model3():
    ##加载数据
    data = load_boston()
    # print(data)
    ##分割数据
    data_x, data_y, x_target, y_target = train_test_split(data.data, data.target, test_size=0.2)
    ##数据标准化
    standard = StandardScaler()
    data_x_format = standard.fit_transform(data_x)
    data_y_format = standard.fit_transform(data_y)
    ##模型训练
    model = Ridge(alpha=1.0)
    model.fit(data_x_format, x_target)
    print("特性参数:\n", model.coef_)
    ##模型预测
    y_predict = model.predict(data_y_format)
    # print("预测结果:\n",y_predict)
    ##模型评估
    data_error = mean_squared_error(y_target, y_predict)
    print("标准平方差:\n", data_error)


linear_model1()
linear_model2()
linear_model3()
