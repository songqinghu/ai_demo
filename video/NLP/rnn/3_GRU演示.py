import torch
import torch.nn as nn

# 模型  词嵌入纬度  隐藏层纬度  隐藏层数
model = nn.GRU(5, 6, 2)
# 输入  批次样本数  词长   词嵌入纬度
input1 = torch.randn(1, 3, 5)
# 隐藏层
h0 = torch.randn(2, 3, 6)

output, hn = model(input1, h0)
print(output)
print(hn)
