import fileinput

import fasttext
import joblib
import torch
from keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter


# one hot 编码
def onehot_demo1():
    # 词语集合
    vocab = ["周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"]
    # 实例化词汇映射器
    t = Tokenizer(num_words=None, char_level=False)
    # 拟合数据
    t.fit_on_texts(vocab)

    # 数据获取
    for token in vocab:
        zero_list = [0] * len(vocab)
        ##从1开始的 [[2]] 二维数据
        print(t.texts_to_sequences([token]))
        token_index = t.texts_to_sequences([token])[0][0] - 1
        zero_list[token_index] = 1
        print(token, " one hot 编码为", zero_list)

    # 保存映射器
    joblib.dump(t, "./Tokenizer")


# 加载映射器
def onehot_demo2():
    t = joblib.load("./Tokenizer")
    # 词语集合
    vocab = ["周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"]
    # 数据获取
    for token in vocab:
        zero_list = [0] * len(vocab)
        ##从1开始的 [[2]] 二维数据
        print(t.texts_to_sequences([token]))
        token_index = t.texts_to_sequences([token])[0][0] - 1
        zero_list[token_index] = 1
        print(token, " one hot 编码为", zero_list)


# fasttext实现word2vec
def word2vec_demo():
    # /Users/songqinghu/Desktop/baiduyun/data/nlp/proprecess/fil9
    # 使用fasttext训练词向量
    model = fasttext.train_unsupervised("/Users/songqinghu/Desktop/baiduyun/data/nlp/proprecess/fil9")
    # 查看指定词的词向量
    word_vec = model.get_word_vector("the")
    print(word_vec)


# 设置word2ver相关参数和验证及保存模型
def word2vec_demo1():
    # 使用fasttext训练词向量
    model = fasttext.train_unsupervised("/Users/songqinghu/Desktop/baiduyun/data/nlp/proprecess/fil9", "cbow", dim=300,
                                        epoch=1, lr=1)
    # 效果检验
    near = model.get_nearest_neighbors('dog')
    print(near)
    # 模型保存和重新加载
    model.save_model("fil9.bin")
    # 模型重加载
    new_model = fasttext.load_model("fil9.bin")
    near1 = new_model.get_nearest_neighbors('dog')
    print(near1)


# word embedding可视化
def wordEmbedding_demo():
    # 实例化写入对象
    writer = SummaryWriter()
    # 随机初始化一个100*5的矩阵用于演示
    embedded = torch.randn(100, 5)

    # 导入准备好的100个词汇
    meta = list(map(lambda x: x.strip(),
                    fileinput.FileInput("/Users/songqinghu/Desktop/baiduyun/data/nlp/proprecess/vocab100.csv")))
    print(meta)
    writer.add_embedding(embedded, metadata=meta)
    writer.close()
    # 使用生成文件启动终端查看效果 tensorboard --logdir runs --host 0.0.0.0


if __name__ == '__main__':
    wordEmbedding_demo()
