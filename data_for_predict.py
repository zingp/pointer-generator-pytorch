# 分词
# 变成example
# 处理成batch对象
import jieba
import config

def proprecession_article(article):
    words = jieba.cut(article)
    if len(words) > config.max_enc_steps:
        words = words[ :config.max_enc_steps]