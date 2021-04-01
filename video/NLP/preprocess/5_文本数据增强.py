from googletrans import Translator

##原始文本
p_simple1 = "酒店设施非常不错"
p_simple2 = "这家价格很便宜"
n_simple1 = "拖鞋都发霉了,太差了"
n_simple2 = "电视不好用,没看到足球"

# 实例化翻译器  这里已经不工作了.后面要用可能需要从新找方法了
translator = Translator(service_urls=['translate.google.cn'])

# 第一次翻译
output = translator.translate([p_simple1, p_simple2, n_simple1, n_simple2], dest='ja')
# 获取结果
ja_res = list(map(lambda x: x.text, output))

print(ja_res)

# 将翻译转会中文
zn_output = translator.translate(ja_res, dest='zh-cn')
# 获取结果
zb_res = list(map(lambda x: x.text, zn_output))

print(zb_res)
