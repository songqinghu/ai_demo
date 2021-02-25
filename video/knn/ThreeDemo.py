##特征预处理
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

##归一化过程
def minmax_scaler():
    data = pd.read_csv("./data/dating.txt")
    print(data)
    transfer = MinMaxScaler(feature_range=(2,3))
    data_res = transfer.fit_transform(data[["milage","Liters","Consumtime"]])
    print(data_res)

def standard_scaler():
    data = pd.read_csv("./data/dating.txt")
    print(data)
    transfer = StandardScaler()
    data_res = transfer.fit_transform(data[["milage","Liters","Consumtime"]])
    print("标准化的结果: \n",data_res)
    print("每一列特征的平均值: \n",transfer.mean_)
    print("每一列特征的方差: \n",transfer.var_)

minmax_scaler()
standard_scaler()