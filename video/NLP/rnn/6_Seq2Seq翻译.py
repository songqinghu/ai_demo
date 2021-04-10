import math
import random
import re
import time
import unicodedata
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据
data_path = "/Users/songqinghu/Desktop/baiduyun/data/nlp/rnn/name/eng-fra.txt"

# 起始结束标识符
SOS_TOKEN = 0
EOS_TOKEN = 1


class Lang:

    def __init__(self, name):
        # 语言名称
        self.name = name
        # 词汇到数值得映射
        self.word2index = {}
        # 自然值到词汇
        self.index2word = {0: "SOS", 1: "EOS"}
        # 初始化对应的索引指针
        self.n_words = 2

    def addSentence(self, sentence):
        """将句子转为数值序列"""
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        """将词语转为对应的索引值"""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words = self.n_words + 1


# 测试
# name = "eng"
# eng = Lang(name)
# eng.addSentence("hello I am Jay")
# print(eng.index2word)
# print(eng.word2index)
# print(eng.n_words)

# 字符规范化 去除非法字符
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if (unicodedata.category(c) != 'Mn'))


# 规范化
def normalizeString(s):
    # 非法字符
    s = unicodeToAscii(s.lower().strip())
    # 在.!?前加空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 将非标准字符转为空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# 测试效果
# s = normalizeString("Are you kidding me?")
# print(s)

# 将数据读取到内容
def readLangs(lang1, lang2):
    lines = open(data_path, encoding='UTF-8').read().strip().split("\n")
    # 获取语言对
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]
    # 放入语言中
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


# 测试
# input_lang,output_lang,pairs = readLangs("eng","fra")
# print(input_lang)
# print(output_lang)
# print(pairs[:5])

# 过滤键值对
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m "
             "he is", "he s "
                      "she is", "she s "
                                "you are", "you re "
                                           "we are", "we re "
                                                     "they are", "they re "
)


def filterPair(p):
    """过滤函数 p为键值对"""
    return len(p[0].split(" ")) < MAX_LENGTH \
           and len(p[1].split(" ")) < MAX_LENGTH \
           and p[0].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# 测试前五个
# fpairs = filterPairs(pairs)
# print(fpairs[:5])


# 整合数据用Lang映射
def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    pairs = filterPairs(pairs)
    # 遍历整合到语言中
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData("eng", "fra")


# print(input_lang.n_words)
# print(output_lang.n_words)
# print(random.choice(pairs))


# 将语言对转换为张量
def tensorFromSentence(lang, sentence):
    # 单词序列转为 数字序列
    indexes = [lang.word2index[word] for word in sentence.split(" ")]
    indexes.append(EOS_TOKEN)  # 结束符号
    # 封装成张量
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    output_tensor = tensorFromSentence(output_lang, pair[1])

    return (input_tensor, output_tensor)


# 测试

# pair_tensor = tensorFromPair(input_lang, output_lang, pairs[1])
# print(pair_tensor)

# GRU编码器
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # 输入为每个次所以 input为 [ 1,embedding]形状 而到gru需要三维张量
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


# hidden_size = 25
# input_size = 20
# print(pair_tensor[0][0]) #第一个词
# encoderRnn = EncoderRNN(input_size,hidden_size)
# hidden = encoderRnn.initHidden()
# output,hidden = encoderRnn(pair_tensor[0][0],hidden)
# print(output)


# 解码器
class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 构建注意力解码器
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        # 注意力
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=-1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 训练模型
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # input_tensor: 代表源语言的输入张量
    # target_tensor: 代表目标语言的输入张量
    # encoder: 代表编码器的实例化对象
    # decoder: 代表解码器的实例化对象
    # encoder_optimizer: 代表编码器优化器
    # decoder_optimizer: 代表解码器优化器
    # criterion: 损失函数
    # max_length: 代表句子的最大长度
    # 初始化编码器的隐藏层张量
    encoder_hidden = encoder.initHidden()

    # 训练前将编码器和解码器的优化器梯度归零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 根据源文本和目标文本张量获得对应的长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # 初始化编码器的输出矩阵张量, 形状是max_length * encoder.hidden_size
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # 设置初始损失值
    loss = 0

    # 遍历输入张量
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # 每一个轮次的输出encoder_output是三维张量, 使用[0,0]进行降维到一维列表, 赋值给输出张量
        encoder_outputs[ei] = encoder_output[0, 0]

    # 初始化解码器的第一个输入字符
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

    # 初始化解码器的隐藏层张量, 赋值给最后一次编码器的隐藏层张量
    decoder_hidden = encoder_hidden

    # 判断是否使用teacher_forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 如果使用teacher_forcing
    if use_teacher_forcing:
        # 遍历目标张量, 进行解码
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # 使用损失函数计算损失值, 并进行累加
            loss += criterion(decoder_output, target_tensor[di])
            # 因为使用了teacher_forcing, 所以将下一步的解码器输入强制设定为“正确的答案”
            decoder_input = target_tensor[di]
    # 如果不适用teacher_forcing
    else:
        # 遍历目标张量, 进行解码
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # 预测值变成输出张量中概率最大的那一个
            topv, topi = decoder_output.topk(1)
            # 使用损失函数计算损失值, 并进行累加
            loss += criterion(decoder_output, target_tensor[di])
            # 如果某一步的解码结果是句子终止符号, 则解码直接结束, 跳出循环
            if topi.squeeze().item() == EOS_TOKEN:
                break
            # 下一步解码器的输入要设定为当前步最大概率值的那一个
            decoder_input = topi.squeeze().detach()

    # 应用反向传播进行梯度计算
    loss.backward()
    # 利用编码器和解码器的优化器进行参数的更新
    encoder_optimizer.step()
    decoder_optimizer.step()

    # 返回平均损失
    return loss.item() / target_length


# 构建时间计算的辅助函数
def timeSince(since):
    # since: 代表模型训练的开始时间
    # 首先获取当前时间
    now = time.time()
    # 计算得到时间差
    s = now - since
    # 将s转换为分钟, 秒的形式
    m = math.floor(s / 60)
    # 计算余数的秒
    s -= m * 60
    # 按照指定的格式返回时间差
    return '%dm %ds' % (m, s)


# since = time.time() - 620

# period = timeSince(since)
# print(period)


import matplotlib.pyplot as plt


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    # encoder: 编码器的实例化对象
    # decoder: 解码器的实例化对象
    # n_iters: 训练的总迭代步数
    # print_every: 每隔多少轮次进行一次训练日志的打印
    # plot_every: 每隔多少轮次进行一次损失值的添加, 为了后续绘制损失曲线
    # learning_rate: 学习率
    # 获取训练开始的时间
    start = time.time()
    # 初始化存放平均损失值的列表
    plot_losses = []
    # 每隔打印间隔的总损失值
    print_loss_total = 0
    # 每个绘制曲线损失值的列表
    plot_loss_total = 0

    # 定义编码器和解码器的优化器
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # 定义损失函数
    criterion = nn.NLLLoss()

    # 按照设定的总迭代次数进行迭代训练
    for iter in range(1, n_iters + 1):
        # 每次从语言对的列表中随机抽取一条样本作为本轮迭代的训练数据
        training_pair = tensorFromPair(random.choice(pairs))
        # 依次将选取出来的语句对作为输入张量, 和输出张量
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # 调用train()函数获得本轮迭代的损失值
        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        # 将本轮迭代的损失值进行累加
        print_loss_total += loss
        plot_loss_total += loss

        # 如果到达了打印的轮次
        if iter % print_every == 0:
            # 首先获取本次打印的平均损失值
            print_loss_avg = print_loss_total / print_every
            # 为了下一个打印间隔的累加, 这里将累加器清零
            print_loss_total = 0
            # 打印若干信息
            print('%s (%d %d%%) %.4f' % (timeSince(start),
                                         iter, iter / n_iters * 100, print_loss_avg))

        # 如果到达了绘制损失曲线的轮次
        if iter % plot_every == 0:
            # 首先获取本次损失添加的平均损失值
            plot_loss_avg = plot_loss_total / plot_every
            # 将平均损失值添加进最后的列表中
            plot_losses.append(plot_loss_avg)
            # 为了下一个添加损失值的累加, 这里将累加器清零
            plot_loss_total = 0

    # 绘制损失曲线
    plt.figure()
    plt.plot(plot_losses)
    plt.savefig("./s2s_loss.png")


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)

attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout=0.1).to(device)

n_iters = 5000
print_every = 500

if __name__ == '__main__':
    trainIters(encoder1, attn_decoder1, n_iters, print_every=print_every)
