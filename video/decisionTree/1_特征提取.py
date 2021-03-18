import jieba
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def dict_demo():
    """
    对字典类型数据进行特征提取
    :return:
    """
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    ##这里为了方便查看设置parse=False 实际使用中还是要用默认值True
    dictVectorizer = DictVectorizer(sparse=False)
    new_data = dictVectorizer.fit_transform(data)
    print("特征名称: ", dictVectorizer.feature_names_)
    print(new_data)


def text_count_demo():
    """
    英文文本数据特征提取
    :return:
    """
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    vectorizer = CountVectorizer()
    new_data = vectorizer.fit_transform(data)
    print("特征名称: ", vectorizer.get_feature_names())
    print("特征值: ", new_data.toarray())


def han_text_count_demo():
    """
    中文文本数据特征提取
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将原始数据转换成分好词的形式
    text_list = []
    for content in data:
        text_list.append(analysisText(content))
    vectorizer = CountVectorizer()
    new_data = vectorizer.fit_transform(text_list)
    print("特征名称: ", vectorizer.get_feature_names())
    print("特征值: ", new_data.toarray())


def analysisText(content):
    """
    使用jieba分词器对文本进行分割和组合成可以按照count统计的方式
    :param content:
    :return:
    """
    next_word = jieba.cut(content)
    new_content = " ".join(next_word)
    return new_content


def han_text_tf_demo():
    """
    中文文本数据特征提取
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将原始数据转换成分好词的形式
    text_list = []
    for content in data:
        text_list.append(analysisText(content))
    vectorizer = TfidfVectorizer(stop_words=['一种', '不会', '不要'])
    new_data = vectorizer.fit_transform(text_list)
    print("特征名称: ", vectorizer.get_feature_names())
    print("特征值: ", new_data.toarray())


if __name__ == '__main__':
    # dict_demo()
    # text_count_demo()
    # han_text_count_demo()
    han_text_tf_demo()
