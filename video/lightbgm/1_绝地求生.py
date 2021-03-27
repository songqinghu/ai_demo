import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

##获取数据
data = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/lightbgm/train_V2.csv")
print(data.head())
print(data.shape)  # 记录数 (4446966, 29)
print(np.unique(data['matchId']).shape)  # 多少场比赛
print(np.unique(data['groupId']).shape)  # 有多少支队伍

##数据处理
# 查看缺失值字段
print(np.any(data.isnull()))
# 查找缺失值
print(data[data['winPlacePerc'].isnull()])
# 删除缺失值
data = data.drop(2744604)
print(data.shape)
print(np.any(data.isnull()))

##特征数据规范化
# 每场比赛的参加人数
count = data.groupby("matchId")['matchId'].transform('count')
data['playersJoined'] = count
print(data.head())

print(data['playersJoined'].sort_values().head())

##画图查看参赛人数分布
plt.figure(figsize=(20, 8))
sns.countplot(data['playersJoined'])
plt.grid()
plt.show()

##规范化输出部分数据
data['killsNorm'] = data['kills'] * ((100 - data['playersJoined']) / 100 + 1)
data["damageDealtNorm"] = data["damageDealt"] * ((100 - data["playersJoined"]) / 100 + 1)
data["maxPlaceNorm"] = data["maxPlace"] * ((100 - data["playersJoined"]) / 100 + 1)
data["matchDurationNorm"] = data["matchDuration"] * ((100 - data["playersJoined"]) / 100 + 1)
# 比较经过规范化的特征值和原始特征值的值
to_show = ['Id', 'kills', 'killsNorm', 'damageDealt', 'damageDealtNorm', 'maxPlace', 'maxPlaceNorm', 'matchDuration',
           'matchDurationNorm']
print(data[to_show][0:11])
# 部分变量合成
data["healsandboosts"] = data["heals"] + data["boosts"]
###异常数据的处理
# 没有移动且杀人的数据
data["totalDistance"] = data["rideDistance"] + data["walkDistance"] + data["swimDistance"]
data["killwithoutMoving"] = (data["kills"] > 0) & (data["totalDistance"] == 0)
# 删除没有移动且杀人的数据
data.drop(data[data["killwithoutMoving"] == True].index, inplace=True)
# 删除驾车杀人异常
data.drop(data[data["roadKills"] > 10].index, inplace=True)
# 删除杀人数超过30人一局中
data.drop(data[data["kills"] > 30].index, inplace=True)
# 删除爆头率异常
data["headshot_rate"] = data["headshotKills"] / data["kills"]
data["headshot_rate"] = data["headshot_rate"].fillna(0)
data.drop(data[(data["headshot_rate"] == 1) & (data["kills"] > 9)].index, inplace=True)
# 删除超远距离杀人
data.drop(data[data["longestKill"] >= 1000].index, inplace=True)
# 删除运动信息异常
# 行走
data.drop(data[data["walkDistance"] >= 10000].index, inplace=True)
# 载具
data.drop(data[data["rideDistance"] >= 20000].index, inplace=True)
# 游泳
data.drop(data[data["swimDistance"] >= 20000].index, inplace=True)
# 武器收集异常
data.drop(data[data["weaponsAcquired"] >= 80].index, inplace=True)
# 异常使用药品
data.drop(data[data["heals"] >= 80].index, inplace=True)

##类别型数据处理
data = pd.get_dummies(data, columns=["matchType"])
matchType_encoding = data.filter(regex="matchType")
data["groupId"] = data["groupId"].astype("category")
data["groupId_cat"] = data["groupId"].cat.codes
data["matchId"] = data["matchId"].astype("category")
data["matchId_cat"] = data["matchId"].cat.codes
data.drop(["groupId", "matchId"], axis=1, inplace=True)

# 数据截取:使用部分数据
df_sample = data.sample(100000)

# 确定特征值和目标值
df = df_sample.drop(["winPlacePerc", "Id"], axis=1)

y = df_sample["winPlacePerc"]

# 分割数据集
X_train, X_valid, y_train, y_valid = train_test_split(df, y, test_size=0.2)

# 模型训练和评估

# 使用随机森林进行训练
m1 = RandomForestRegressor(n_estimators=40,
                           min_samples_leaf=3,
                           max_features='sqrt',
                           n_jobs=-1)

m1.fit(X_train, y_train)

y_pre = m1.predict(X_valid)
print(m1.score(X_valid, y_valid))
print(mean_absolute_error(y_valid, y_pre))

# 使用lightbgm训练
X_train, X_valid, y_train, y_valid = train_test_split(df, y, test_size=0.2)

gbm = lgb.LGBMRegressor(objective="regression", num_leaves=31, learning_rate=0.05, n_estimators=20)

gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="l1", early_stopping_rounds=5)

y_pre = gbm.predict(X_valid, num_iteration=gbm.best_iteration_)

print(mean_absolute_error(y_valid, y_pre))

# 模型调优

estimator = lgb.LGBMRegressor(num_leaves=31)
param_grid = {
    "learning_rate": [0.01, 0.1, 1],
    "n_estimators": [40, 60, 80, 100, 200, 300]
}

gbm = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1)

gbm.fit(X_train, y_train)

y_pre = gbm.predict(X_valid)
print(mean_absolute_error(y_valid, y_pre))
print(gbm.best_params_)

# 模型三次调优
scores = []
n_estimators = [100, 300, 500, 800]

for nes in n_estimators:
    lgbm = lgb.LGBMRegressor(boosting_type='gbdt',
                             num_leaves=31,
                             max_depth=5,
                             learning_rate=0.1,
                             n_estimators=nes,
                             min_child_samples=20,
                             n_jobs=-1)

    lgbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="l1", early_stopping_rounds=5)

    y_pre = lgbm.predict(X_valid)

    mae = mean_absolute_error(y_valid, y_pre)

    scores.append(mae)
    print("本次结果输出的mae值是:\n", mae)
plt.plot(n_estimators, scores, 'o-')
plt.ylabel("mae")
plt.xlabel("n_estimator")
print("best n_estimator {}".format(n_estimators[np.argmin(scores)]))

# max_depth

scores = []
max_depth = [3, 5, 7, 9, 11]

for md in max_depth:
    lgbm = lgb.LGBMRegressor(boosting_type='gbdt',
                             num_leaves=31,
                             max_depth=md,
                             learning_rate=0.1,
                             n_estimators=500,
                             min_child_samples=20,
                             n_jobs=-1)

    lgbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="l1", early_stopping_rounds=5)

    y_pre = lgbm.predict(X_valid)

    mae = mean_absolute_error(y_valid, y_pre)

    scores.append(mae)
    print("本次结果输出的mae值是:\n", mae)

plt.plot(max_depth, scores, 'o-')
plt.ylabel("mae")
plt.xlabel("max_depths")
print("best max_depths {}".format(max_depth[np.argmin(scores)]))

print(scores)
