import torch
import torch.nn as nn

# 定义rnn模型 第一个参数为输入的特征纬度  第二个参数为隐藏层神经元数 第三个参数为隐藏层层数
model = nn.RNN(5, 6, 1)
# 创建初始输入 第一个参数为该批次样本个数 第二个参数为seq_length  第三个参数为词嵌入纬度
input1 = torch.randn(1, 3, 5)
print(input1)
# 创建隐藏层  第一个参数为 隐藏层层数  第二个参数为当前样本 seq_length   第三个参数为每层神经元数
h0 = torch.randn(1, 3, 6)
print(h0)

output, hn = model(input1, h0)

print(output)
print(hn)
