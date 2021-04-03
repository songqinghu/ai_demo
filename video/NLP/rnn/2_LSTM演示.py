import torch
import torch.nn as nn

# 模型  词嵌入纬度   隐藏层神经元数量  隐藏层层数
model = nn.LSTM(5, 6, 2)
# 输入  该批次的样本数量   seq_length  词纬度
input1 = torch.randn(1, 3, 5)
# 隐藏层 层数 seq_length  神经元数
h0 = torch.randn(2, 3, 6)
# 细胞状态
c0 = torch.randn(2, 3, 6)
# 训练
output, (hn, cn) = model(input1, (h0, c0))
print(output)
print(hn)
print(cn)
