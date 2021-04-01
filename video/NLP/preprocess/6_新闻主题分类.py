import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

# load_data_dir = './data'

# if not os.path.exists(load_data_dir):
#    os.mkdir(load_data_dir)

# train_dataset,test_dataset = DATASETS['AG_NEWS'](root=load_data_dir)
##/Users/songqinghu/Desktop/baiduyun/data/nlp/proprecess/data/ag_news_csv

train_dataset = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/nlp/proprecess/data/ag_news_csv/train.csv")
test_dataset = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/nlp/proprecess/data/ag_news_csv/test.csv")
print(train_dataset.head())
##构建embedding层的文本分类模型

# 指定 BATCH_SIZE
BATCH_SIZE = 16

# 进行可用设备检测,有GPU就是用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 分类模型类
class TextSentiment(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        """
        :param vocab_size: 整个语料包含不同词汇总数
        :param embed_dim:  指定词嵌入的纬度数
        :param num_class:  文本分类的类别总数
        """
        super().__init__()
        # 实例化embedding层 sparse=True 表示每次只对部分权重更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        # 实例化线性层 输入特征和输入特征
        self.fc = nn.Linear(embed_dim, num_class)
        # 为各层初始化权重
        self.init_weights()

    def init_weights(self):
        init = 0.5
        self.embedding.weight.data.uniform_(-init, init)
        self.fc.weight.data.uniform_(-init, init)
        # 偏置初始置为零
        self.fc.bias.data.zero_()

    def forward(self, text):
        """
        :param text: 文本数值映射后的结果
        :return: 与类别数尺寸相同的的张量,用来判断文本类型
        """
        # 获取embedding的结果
        # (m,32) m为该批次中词汇的总数
        embedded = self.embedding(text)
        # 将(m,32)转化为 (batch_size,32)
        # 已知 m 远大于 batch_size
        # m整除获取有多少个 batch_size
        c = embedded.size(0) // BATCH_SIZE
        # 获取新的embedded
        embedded = embedded[: c * BATCH_SIZE]
        # 使用平均池化求取指定行数列的平均数
        embedded = embedded.transpose(1, 0).unsqueeze(0)
        embedded = F.avg_pool1d(embedded, kernel_size=c)
        return self.fc(embedded[0].transpose(1, 0))


# 实例化模型

# 获得整个语料包含的不同词汇总数
VOCAB_SIZE = len(train_dataset.get_vocab())
print(VOCAB_SIZE)
# 指定词嵌入纬度
EMBED_DIM = 32
# 获取类别总数
NUM_CLASS = len(train_dataset.get_labels())
# 实例化模型
model = TextSentiment(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_class=NUM_CLASS).to(device)


# 对数据进行batch处理
def generate_batch(batch):
    # 从batch中获取标签张量
    label = torch.tensor([entry[1] for entry in batch])
    # 从batch中获取样本张量
    text = torch.tensor([entry[0] for entry in batch])
    text = torch.cat(text)
    return text, label


# 构建训练和验证函数
def train(train_data):
    # 初始化损失率和准确率
    train_loss = 0
    train_acc = 0

    # 使用数据加载器生成batch_size大小的数据进行批次训练
    data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    # 对data进行遍历更新参数
    for i, (text, cls) in enumerate(data):
        optimizer.zero()
        output = model(text)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    scheduler.step()

    return train_loss / len(train_data), train_acc / len(train_data)


def valid(valid_data):
    # 初始化损失率和准确率
    valid_loss = 0
    valid_acc = 0

    # 使用数据加载器生成batch_size大小的数据进行批次训练
    data = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    # 对data进行遍历更新参数
    for text, cls in enumerate(data):
        # 验证阶段不在求解梯度
        with torch.no_grad():
            output = model(text)
            loss = criterion(output, cls)
            valid_loss += loss.item()
            valid_acc += (output.argmax(1) == cls).sum().item()

    return valid_loss / len(valid_data), valid_acc / len(valid_data)


# 进行模型训练和验证

# 训练轮次
N_EPOCHS = 10

# 定义初始的验证损失
min_valid_loss = float("inf")
# 选择损失函数
criterion = nn.CrossEntropyLoss().to(device)
# 随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
# 步长优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

# 获取部分数据的长度用户训练
train_len = int(len(train_dataset) * 0.95)
# 进行数据切割
sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

# 开始每一轮训练
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(sub_train_)
    valid_loss, valid_acc = valid(sub_valid_)

    cost_time = int(time.time() - start_time)
    print("耗时 : ", cost_time)
    print("训练 : ", train_loss, train_acc)
    print("验证 : ", valid_loss, valid_acc)

# 查看embedding层嵌入的词向量
print(model.state_dict()['embedding.weight'])
