import hanlp
import jieba
import jieba.posseg as pseg


# jieba精确分词
def jieba_demo1():
    text = "以“母阵”为根基，可以演化一切阵法，阴阳五行、地风水火雷，以及这十一种大阵延伸的三百六十种小阵，皆可依靠母阵，随心所欲的施展。"
    tokenizer = jieba.cut(text, cut_all=False)
    print(tokenizer)
    content = jieba.lcut(text)
    print(content)


# jieba全模式
def jieba_demo2():
    text = "以“母阵”为根基，可以演化一切阵法，阴阳五行、地风水火雷，以及这十一种大阵延伸的三百六十种小阵，皆可依靠母阵，随心所欲的施展。"
    tokenizer = jieba.cut(text, cut_all=True)
    print(tokenizer)
    content = jieba.lcut(text, cut_all=True)
    print(content)


# 搜索引擎模式
def jieba_demo3():
    text = "以“母阵”为根基，可以演化一切阵法，阴阳五行、地风水火雷，以及这十一种大阵延伸的三百六十种小阵，皆可依靠母阵，随心所欲的施展。"
    tokenizer = jieba.cut_for_search(text)
    print(tokenizer)
    content = jieba.lcut_for_search(text)
    print(content)


# 用户自定义词典
def jieba_demo4():
    text = "以“母阵”为根基，可以演化一切阵法，阴阳五行、地风水火雷，以及这十一种大阵延伸的三百六十种小阵，皆可依靠母阵，随心所欲的施展。"
    content = jieba.lcut(text)
    print(content)
    jieba.load_userdict("./dict")
    print(jieba.lcut(text))


# 中文分词
def hanlp_demo1():
    tokenizer = hanlp.load("CTB6_CONVSEG")
    content = tokenizer("以“母阵”为根基，可以演化一切阵法，阴阳五行、地风水火雷，以及这十一种大阵延伸的三百六十种小阵，皆可依靠母阵，随心所欲的施展。")
    print(content)


# 英文分词 -- 这个没找到
def hanlp_demo2():
    tokenizer = hanlp.load("CTB6_CONVSEG")
    content = tokenizer("A Silly BoyMum: Jack, what's the weather like today")
    print(content)


# 中文实体命名
def hanlp_demo3():
    tokenizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ALBERT_BASE_ZH)
    content = tokenizer("中新网北京新闻3月31日电 (陈杭)“今天距北京冬奥会和冬残奥会开幕仅有310天，筹办工作日益紧迫。")
    print(content)


# 英文实体命名
def hanlp_demo4():
    tokenizer = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_UNCASED_EN)
    content = tokenizer(["President", "Obama", "said", "the", "White", "House", "is", "my", "office"])
    print(content)


##jieba词性标注
def jieba_demo5():
    content = pseg.lcut("中新网北京新闻3月31日电 (陈杭)“今天距北京冬奥会和冬残奥会开幕仅有310天，筹办工作日益紧迫。")
    print(content)


##hanlp中文词性标注
def hanlp_demo5():
    tokenizer = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
    content = tokenizer("中新网北京新闻3月31日电 (陈杭)“今天距北京冬奥会和冬残奥会开幕仅有310天，筹办工作日益紧迫。")
    print(content)


# hanlp英文词性标注
def hanlp_demo6():
    tokenizer = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
    content = tokenizer(["President", "Obama", "said", "the", "White", "House", "is", "my", "office"])
    print(content)


if __name__ == '__main__':
    hanlp_demo5()
