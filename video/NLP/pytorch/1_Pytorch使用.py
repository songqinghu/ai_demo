import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 版本信息
print(torch.__version__)

##张量 tensor
# 1.1 创建
a = torch.tensor([1, 2, 3])
print(a)
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(b)
c = torch.ones([3, 4])
print(c)
d = torch.zeros([3, 4])
print(d)
e = torch.empty([3, 4])
print(e)
f = torch.linspace(1, 10, 9)
print(f)
g = torch.rand([3, 4])
print(g)
h = torch.randint(1, 10, [3, 4])
print(h)
aa = torch.randn([3, 4])
print(aa)

ab = torch.tensor(np.arange(1))
print(ab)

ac = ab
print(ac)
ac = ab.item()  # 必须单元素才可以
print(ac)
ab = torch.tensor([[1, 2], [2, 3], [3, 4]])
ac = ab.numpy()
print(ac)

print(ab.size())
print(ab.dtype)
print(ab.dim())

ac = ab.view(2, 3)
print(ac)
print(ac.size())
print(ab.t())
print(ab)
print(ab.transpose(1, 0))
x = torch.tensor([[[1., 2., 3.],
                   [4., 5., 6.]],

                  [[2., 2., 3.],
                   [4., 5., 6.]],

                  [[3., 2., 3.],
                   [4., 5., 6.]],

                  [[4., 2., 3.],
                   [4., 5., 6.]]]
                 )
print(x.size())
y = x.transpose(1, 2)
print(y.size())
z = y.transpose(0, 1)
print(z.size())
xx = x.permute(2, 0, 1)
print(xx.size())

a = torch.tensor([[[1], [2], [3]]])

print(a.size())
b = a.squeeze()
print(b)
print(b.size())
c = b.unsqueeze(0)
print(c)
print(c.size())
c = b.unsqueeze(1)  # 指定位置
print(c)
print(c.size())

a = torch.ones([3, 4])
print(a.dtype)
a = torch.ones([3, 4], dtype=torch.int32)
print(a.dtype)

a = torch.randn([2, 3, 4])
print(a)
print(a[:, :1, :2].size())
print(a[:, :1, :2].reshape(1, 4))

# gpu
print(torch.cuda.is_available())

a = torch.randn([3, 4])
b = torch.randn([4, 5])

print(a.mm(b).size())

a = torch.tensor([1, 2, 3])
b = torch.tensor([1])
print(a.add(b))
print(a)
print(a.add_(b))
print(a)

x = torch.ones([2, 2])
print(x)
print(x.requires_grad)
y = x + 2
print(y)
z = y * y * 3
print(z)
out = z.mean()
print(out)
print(out.item())

x = torch.ones([2, 2], requires_grad=True)
print(x)
print(x.requires_grad)
y = x + 2
print(y)
z = y * y * 3
print(z)
out = z.mean()
print(out)
print(out.item())

a = torch.randn([2, 2])
print(a)
print(a.requires_grad)
a = ((a * 3) / (a + 1))
print(a)
print(a.requires_grad)
a.requires_grad = True
print(a)
b = (a * a).sum()
print(b)
print(b.requires_grad)
print(b.grad_fn)
with torch.no_grad():
    c = (a * a).sum()
print(c)
print(b.data)
c1 = c.numpy()
print(c1)
print(type(c1))
b1 = b.detach().numpy()
print(b1)
print(type(b1))

##手工模拟线性回归
x = torch.rand([50])
y = 3 * x + 0.8
# print(x,y)
w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)
print(w, b)


def loss_fn(y, y_pre):
    loss = (y_pre - y).pow(2).mean()
    for i in [w, b]:
        if i.grad is not None:
            i.grad.data.zero_()
    loss.backward()  # 这一步做了啥??
    return loss.data


def optimize(lr):
    w.data -= lr * w.grad.data
    b.data -= lr * b.grad.data


# 训练
for i in range(30000):
    y_pre = w * x + b
    loss = loss_fn(y, y_pre)
    if i % 1000 == 0:
        print(i, loss)
    optimize(0.01)
print(w, b)
predict = w * x + b
plt.scatter(x.data.numpy(), y.data.numpy(), c='r')
plt.plot(x.data.numpy(), predict.data.numpy())
plt.show()

##模型使用
model = nn.Sequential(nn.Linear(2, 64), nn.Linear(64, 1))
x = torch.randn(10, 2)
print(x)
out = model(x)
print(out)

# 正规方式处理
x = torch.rand([50, 1])
y = 3 * x + 0.8


class Lr(nn.Module):

    def __init__(self):
        super(Lr, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


model = Lr()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
for i in range(30000):
    out = model(x)
    loss = criterion(y, out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % 1000 == 0:
        print('Epoch{}/{}, Loss: {:.6f}'.format(i, 30000, loss.data))

model.eval()  # 开始预测模式
pre = model(x)
output = pre.data.numpy()
plt.scatter(x.data.numpy(), y.data.numpy(), c="r")
plt.plot(x.data.numpy(), output)
plt.show()

# 判断和使用gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x, y = x.to(device), y.to(device)
model = Lr().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
for i in range(30000):
    out = model(x)
    loss = criterion(y, out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % 1000 == 0:
        print('Epoch {}/{}, Loss: {:.6f}'.format(i, 30000, loss.data))

model.eval()
predict = model(x)
predict = predict.cpu().detach().numpy()
plt.scatter(x.cpu().data.numpy(), y.cpu().data.numpy(), c='r')
plt.plot(x.cpu().data.numpy(), predict)
plt.show()
