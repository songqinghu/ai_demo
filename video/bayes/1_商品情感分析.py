import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

##读取数据
data = pd.read_csv("/Users/songqinghu/Desktop/baiduyun/data/bayes/书籍评价.csv", encoding="gbk")
print(data)
print(data.shape)
##数据处理
##取出内容用户分词处理
content = data['内容']
##将结果分类转换为数字
print(data.loc[:, '评价'])
data.loc[data.loc[:, '评价'] == '好评', '评论编号'] = 1
data.loc[data.loc[:, '评价'] == '差评', '评论编号'] = 0
print(data)
##停用词
stopwords = []
with open("/Users/songqinghu/Desktop/baiduyun/data/bayes/stopwords.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    # print(lines)
    for tmp in lines:
        line = tmp.strip()
        # print(line)
        stopwords.append(line)

print(stopwords)

##分词
content_list = []

for line in content:
    seg = jieba.cut(line, cut_all=False)
    temp = ",".join(seg)
    content_list.append(temp)

##获取词频
cv = CountVectorizer(stop_words=stopwords)
x = cv.fit_transform(content_list)

##训练集和测试集
train_x = x.toarray()[:10, :]
train_y = data['评价'][:10]

x_test = x.toarray()[10:, :]
y_test = data["评价"][10:]

##训练
model = MultinomialNB(alpha=1.0, )
model.fit(train_x, train_y)
##预测
pre = model.predict(x_test)
print("预测结果\n", pre)
print("真实结果\n", y_test)
# 模型评价
score = model.score(x_test, y_test)
print("模型打分: \n", score)
