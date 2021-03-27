import math

import numpy as np
from hmmlearn import hmm

# 设定隐藏状态的集合
states = ['box1', 'box2', 'box3']
n_states = len(states)

# 设定观察状态的集合
observations = ['red', 'white']
n_observations = len(observations)

# 设定初始状态分布
start_probability = np.array([0.2, 0.4, 0.4])

# 设定状态转移概率分布矩阵
transition_probability = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

# 设定观测状态概率矩阵
emission_probability = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
])

# 设定模型参数
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

seen = np.array([[0, 1, 0]]).T
box = model.predict(seen)

print("球的观测顺序为：\n", ",".join(map(lambda x: observations[x], seen.flatten())))
print("最可能的隐藏状态序列为:\n", ",".join(map(lambda x: states[x], box)))
print(math.exp(model.score(seen)))
