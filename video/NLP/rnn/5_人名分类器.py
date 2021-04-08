import glob
import math
import os
import random
import string
import time
import unicodedata
from io import open

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# /Users/songqinghu/Desktop/baiduyun/data/nlp/rnn/name/names

# 常量定义
data_path = "/Users/songqinghu/Desktop/baiduyun/data/nlp/rnn/name/names/"
category_lines = {}
all_categorys = []

# 常用字符
all_letters = string.ascii_letters + " ,.;'"
n_letters = len(all_letters)


# s = "À l'aide"
# unicode转ascii
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if (unicodedata.category(c) != 'Mn' and c in all_letters))


# 读取文件
def readLines(filename):
    lines = open(filename, encoding='UTF-8').read().strip().split("\n")
    return [unicodeToAscii(line) for line in lines]


# filename = data_path + "Chinese.txt"

# 构建类别到关系的词典
for filename in glob.glob(data_path + "*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]

    all_categorys.append(category)

    category_lines[category] = readLines(filename)

n_categorys = len(all_categorys)


# print("n_categorys : ",n_categorys)
# print(category_lines['German'][:10])

# 将人名转为onehot张量形式
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    # 将人名和张量的位置对应上
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.index(letter)] = 1

    return tensor


# line = 'Bai'
# print(lineToTensor(line))

# 构建RNN模型
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # input 张量形状 [1,n_letters]
        # hidden 张量形状 [hidden_layers,1,hidden_size]
        input = input.unsqueeze(0)
        rr, hn = self.rnn(input, hidden)
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        """初始化隐藏张量h0"""
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# LSTM模型
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, c):
        # input 张量形状 [1,n_letters]
        # hidden 张量形状 [hidden_layers,1,hidden_size]
        input = input.unsqueeze(0)
        rr, (hn, c) = self.lstm(input, (hidden, c))
        return self.softmax(self.linear(rr)), hn, c

    def initHiddenAndC(self):
        """初始化隐藏张量h0"""
        c = h = torch.zeros(self.num_layers, 1, self.hidden_size)
        return h, c


# GRU模型   先跳过
class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # input 张量形状 [1,n_letters]
        # hidden 张量形状 [hidden_layers,1,hidden_size]
        input = input.unsqueeze(0)
        rr, hn = self.gru(input, hidden)
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        """初始化隐藏张量h0"""
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# 实例化张量
# 输入张量最后一维为字符集大小
input_size = n_letters
# 隐藏层大小自定义
hidden_size = 128
# 输出层张量为要预测的类别
output_size = n_categorys
# 隐藏层数默认设置1
num_layers = 1

# 输入测试
input = lineToTensor('B').squeeze(0)
hidden = c = torch.zeros(1, 1, hidden_size)

rnn = RNN(input_size, hidden_size, output_size, num_layers)
lstm = LSTM(input_size, hidden_size, output_size, num_layers)
gru = GRU(input_size, hidden_size, output_size, num_layers)


# rnn_output,next_hidden = rnn(input,hidden)
# print("rrn_output:\n",rnn_output)


# 构建训练模型进行训练

# 从输出结果中获取指定类别
def categoryFromOutput(output):
    # 获取概率最大的索引
    top_n, top_i = output.topk(1)
    # print(top_n,top_i)
    category_i = top_i[0].item()
    return all_categorys[category_i], category_i


# x = torch.arange(1,6)
# print(x)
# print(x.topk(2)[0],x.topk(2)[1][0])

# category_name,category_i = categoryFromOutput(rnn_output)
# print(category_name,category_i)

# 随机生成训练数据
def randomTrainingExample():
    # 随机类别
    category = random.choice(all_categorys)
    # 随机名字
    line = random.choice(category_lines[category])
    # 获取类别张量
    category_tensor = torch.tensor([all_categorys.index(category)], dtype=torch.long)
    # 将名字转为onehot
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


# 构建RNN训练函数
criterion = nn.NLLLoss()

# 学习率
learning_rate = 0.005


def trainRNN(category_tensor, line_tensor):
    # 初始化h0
    hidden = rnn.initHidden()

    # 梯度归零
    rnn.zero_grad()

    # 开始训练 行的长度进行第一个轮划分
    for i in range(line_tensor.size()[0]):
        # 这里每次取第一个字符的二维张量
        output, hidden = rnn(line_tensor[i], hidden)

    # 计算该轮次的损失
    loss = criterion(output.squeeze(0), category_tensor)

    # 反向传播
    loss.backward()

    # 更新模型中参数
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


# LSTM训练暂时略过

def trainLSTM(category_tensor, line_tensor):
    # 初始化h0
    hidden, c = lstm.initHiddenAndC()

    # 梯度归零
    lstm.zero_grad()

    # 开始训练 行的长度进行第一个轮划分
    for i in range(line_tensor.size()[0]):
        # 这里每次取第一个字符的二维张量
        output, hidden, c = lstm(line_tensor[i], hidden, c)

    # 计算该轮次的损失
    loss = criterion(output.squeeze(0), category_tensor)

    # 反向传播
    loss.backward()

    # 更新模型中参数
    for p in lstm.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


# GRU训练暂时略过
def trainGRU(category_tensor, line_tensor):
    # 初始化h0
    hidden = gru.initHidden()

    # 梯度归零
    gru.zero_grad()

    # 开始训练 行的长度进行第一个轮划分
    for i in range(line_tensor.size()[0]):
        # 这里每次取第一个字符的二维张量
        output, hidden = gru(line_tensor[i], hidden)

    # 计算该轮次的损失
    loss = criterion(output.squeeze(0), category_tensor)

    # 反向传播
    loss.backward()

    # 更新模型中参数
    for p in gru.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


# 时间计算函数
def timeSince(since):
    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s = - m * 60

    return "%dm %ds" % (m, s)


# 打印相关参数
n_iter = 5000
# 打印间隔轮次
print_every = 500
# 设置制图间隔
plot_every = 500


def train(train_type_fn):
    all_losses = []
    start = time.time()
    current_loss = 0
    for iter in range(1, n_iter + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train_type_fn(category_tensor, line_tensor)
        current_loss += loss

        # 是否打印
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            print("iter {}  category {}  guess {}  cost time {}".format(iter, category, guess, time.time() - start))

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    return all_losses, time.time() - start


# 开始训练
all_losses1, period1 = train(trainRNN)
all_losses2, period2 = train(trainLSTM)
all_losses3, period3 = train(trainGRU)

# 画图
plt.figure(0)

plt.plot(all_losses1, label="RNN")
plt.plot(all_losses2, label="LSTM")
plt.plot(all_losses3, label="GRU")
plt.legend(loc="upper left")

plt.figure(1)
x_data = ["RNN", "LSTM", "GRU"]
y_data = [period1, period2, period3]
plt.bar(range(len(x_data)), y_data, tick_label=x_data)

plt.show()


# 构建评估模型进行评估


def evaluateRNN(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output.squeeze(0)


def evaluateLSTM(line_tensor):
    hidden, c = lstm.initHiddenAndC()
    for i in range(line_tensor.size()[0]):
        output, hidden, c = lstm(line_tensor[i], hidden, c)

    return output.squeeze(0)


def evaluateGRU(line_tensor):
    hidden = gru.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i], hidden)

    return output.squeeze(0)


# 构建预测函数
def predict(input_line, evaluate, n_predictions=3):
    print('\n %s' % input_line)

    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print("(%.2f) %s" % (value, all_categorys[category_index]))
            predictions.append([value, all_categorys[category_index]])

    return predictions


# 调用
for evaluate_fn in [evaluateRNN, evaluateLSTM, evaluateGRU]:
    print("-" * 18)
    predict("Song", evaluate_fn)
