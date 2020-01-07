#!/usr/bin/env python
# coding: utf-8
import os
import glob

def read_file(filename):
    with open(filename, "r") as f:
        cont = f.read()
    return cont

def gen_sentence(filename):
    cont = read_file(filename)
    cont = "".join(cont.split(" "))
    return cont

def compute_rouge_n(text1, text2, n):
    def ngram(text, n):
        leng = len(text)
        word_dic = {}
        for i in range(0, leng, n):
            start = i
            words = ""
            if leng - start < n:
                break
            else:
                words = text[start: start+n]
                word_dic[words] = 1
        return word_dic
    dic1 = ngram(text1, n)
    dic2 = ngram(text2, n)
    x = 0
    y = len(dic2)
    for w in dic1:
        if w in dic2:
            x += 1
    rouge = x / y
    return rouge if rouge <=1.0 else 1.0


def avg_rouge(ref_dir, dec_dir, n):
    ref_files = os.path.join(ref_dir, "*reference.txt")
    filelist = glob.glob(ref_files)
    scores_list = []
    for ref_file in filelist:
        basename = os.path.basename(ref_file)
        number = basename.split("_")[0]
        dec_file = os.path.join(dec_dir, "{}_decoded.txt".format(number))
        dec_cont = gen_sentence(dec_file)
        ref_cont = gen_sentence(ref_file)
        score = compute_rouge_n(dec_cont, ref_cont, n)
        scores_list.append(score)
    return sum(scores_list) / len(scores_list)


if __name__ == "__main__":
    #root_dir = "./logs/weibo_adam/decode_model_494000_20200106_110724"
    root_dir = "./logs/weibo_adagrad/decode_model_704000_20200107_093803"
    ref_dir = os.path.join(root_dir, "rouge_ref")
    dec_dir = os.path.join(root_dir, "rouge_dec_dir")
    for i in range(1,3):
        print("ROUGE-{} : {:.4}".format(i, avg_rouge(ref_dir, dec_dir, i)))

