from keras.preprocessing import sequence

##ngram特征提取
ngram_num = 2


def create_ngram_set(input_list):
    return set(zip(*[input_list[i:] for i in range(ngram_num)]))


output = create_ngram_set([1, 3, 2, 1, 5, 3])
print(output)

##文本长度规范

# 限定文本长度
cutlen = 10


def padding(x_trian):
    return sequence.pad_sequences(x_trian, cutlen)


x_train = [[1, 34, 23, 2, 3, 22, 45, 21, 5, 6, 7, 39, 99], [4, 6, 9, 10, 20, 40]]

print(padding(x_train))
