##波士顿房价预测--模型保存和加载
from joblib import load
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


##岭回归--模型保存和加载
def linear_model():
    ##加载数据
    data = load_boston()
    # print(data)
    ##分割数据
    data_x, data_y, x_target, y_target = train_test_split(data.data, data.target, test_size=0.2, random_state=22)
    ##数据标准化
    standard = StandardScaler()
    data_x_format = standard.fit_transform(data_x)
    data_y_format = standard.fit_transform(data_y)
    ##模型训练
    # model = Ridge(alpha=1.0)
    # model.fit(data_x_format,x_target)
    ##保存模型
    # dump(model,"./data/ridge.pkl")
    ##加载模型
    model = load("./data/ridge.pkl")
    print("特性参数:\n", model.coef_)
    ##模型预测
    y_predict = model.predict(data_y_format)
    # print("预测结果:\n",y_predict)
    ##模型评估
    data_error = mean_squared_error(y_target, y_predict)
    print("标准平方差:\n", data_error)


linear_model()
