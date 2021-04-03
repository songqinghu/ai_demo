import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):

    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super(Attn, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        # 初始化注意力
        self.attn = nn.Linear(self.query_size + self.key_size, self.value_size1)

        # 注意力第三步
        self.attn_combine = nn.Linear(self.query_size + self.value_size2, self.output_size)

    def forward(self, Q, K, V):
        # 线性变换
        attn_weights = F.softmax(self.attn(torch.cat((Q[0], K[0]), 1)), dim=1)

        # bmm
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), V)

        output = torch.cat((Q[0], attn_applied[0]), 1)

        output = self.attn_combine(output).unsqueeze(0)

        return output, attn_weights


query_size = 32
key_size = 32
value_size1 = 32
value_size2 = 64
output_size = 64

attn = Attn(query_size, key_size, value_size1, value_size2, output_size)

Q = torch.randn(1, 1, query_size)
K = torch.randn(1, 1, key_size)
V = torch.randn(1, value_size1, value_size2)

output, attn_weights = attn(Q, K, V)
print(output)
print(output.size())
print(attn_weights)
print(attn_weights.size())
