from itertools import chain

import jieba
import jieba.posseg as pseg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

# 设置现实风格
plt.style.use("fivethirtyeight")

# 读取训练集和测试集
train_data = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/nlp/proprecess/cn_data/train.tsv", sep="\t")
valid_data = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/nlp/proprecess/cn_data/dev.tsv", sep="\t")

print(train_data.shape, valid_data.shape)
print(train_data.head())
print(valid_data.head())


##查看训练数据和验证数据标签分布
def show_lable():
    sns.countplot(train_data['label'])
    plt.title("train_data_label")
    plt.show()

    sns.countplot(valid_data['label'])
    plt.title("valid_data_label")
    plt.show()


##查看训练集和验证集上句子长度的数据分布规律
def show_sentence_length():
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))
    ##绘制训练集句子长度分布
    sns.countplot(train_data["sentence_length"])
    plt.xticks([])
    plt.show()
    # 绘制dist长度分布图
    sns.displot(train_data["sentence_length"])
    plt.yticks([])
    plt.show()

    valid_data['sentence_length'] = list(map(lambda x: len(x), valid_data['sentence']))
    ##绘制训练集句子长度分布
    sns.countplot(valid_data["sentence_length"])
    plt.xticks([])
    plt.show()
    # 绘制dist长度分布图
    sns.displot(valid_data["sentence_length"])
    plt.yticks([])
    plt.show()


##查看正负样本长度散点图
def show_strip():
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))
    valid_data['sentence_length'] = list(map(lambda x: len(x), valid_data['sentence']))

    sns.stripplot(x='label', y='sentence_length', data=train_data)
    plt.show()

    sns.stripplot(x='label', y='sentence_length', data=valid_data)
    plt.show()


# 词汇统计

##对句子分词并统计出不同词的总数
def word_count():
    train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data['sentence'])))
    valid_vocab = set(chain(*map(lambda x: jieba.lcut(x), valid_data['sentence'])))
    print("训练集词总量: ", len(train_vocab))
    print("验证集词总量: ", len(valid_vocab))


##获得训练集和测试集上正负样本的高频形容词词云

# 获取形容词
def get_adj_list(text):
    adj = []
    for g in pseg.lcut(text):
        if g.flag == "a":
            adj.append(g.word)
    return adj


# 绘制词云
def gen_word_cloud(keyword_list):
    # 实例化词云生成器
    wordcloud = WordCloud(max_words=100, background_color='white', font_path="./SimHei.ttf")
    # 将list改为字符串
    keyword_string = " ".join(keyword_list)
    # 画词云
    wordcloud.generate(keyword_string)
    # 绘制图像并显示
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# 获得训练集和测试集上正负样本每个句子的形容词并展示词云
def show_cloud():
    p_train_data = train_data[train_data['label'] == 1]['sentence']
    trian_p_a_vocab = chain(*map(lambda x: get_adj_list(x), p_train_data))
    # print(trian_p_a_vocab)
    n_train_data = train_data[train_data['label'] == 0]['sentence']
    trian_n_a_vocab = chain(*map(lambda x: get_adj_list(x), n_train_data))
    # print(trian_n_a_vocab)
    gen_word_cloud(trian_n_a_vocab)
    gen_word_cloud(trian_p_a_vocab)

    p_valid_data = valid_data[valid_data['label'] == 1]['sentence']
    valid_p_a_vocab = chain(*map(lambda x: get_adj_list(x), p_valid_data))

    n_valid_data = valid_data[valid_data['label'] == 0]['sentence']
    valid_n_a_vocab = chain(*map(lambda x: get_adj_list(x), n_valid_data))

    gen_word_cloud(valid_p_a_vocab)
    gen_word_cloud(valid_n_a_vocab)


if __name__ == '__main__':
    # show_cloud()
    pass
