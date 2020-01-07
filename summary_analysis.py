"""
分析和展示生成的摘要与参考摘要
"""
import os
import glob

from rouge import load_word_sents

def print_summary(ref_dir, dec_dir):
    ref_files = os.path.join(ref_dir, "*reference.txt")
    filelist = glob.glob(ref_files)
    n = 0
    for ref_file in filelist:
        basename = os.path.basename(ref_file)
        number = basename.split("_")[0]
        dec_file = os.path.join(dec_dir, "{}_decoded.txt".format(number))
        dec_cont = load_word_sents(dec_file)
        ref_cont = load_word_sents(ref_file)
        n += 1
        print("参考摘要：", ref_cont)
        print("\033[36;1m生成摘要：\033[0m", dec_cont)
        print("- "*50)
        if n == 1000:
            print("已经输出1000条")
            break

if __name__ == "__main__":
    root_dir = "./logs/weibo_adagrad/decode_model_704000_20200107_093803"
    ref_dir = os.path.join(root_dir, "rouge_ref")
    dec_dir = os.path.join(root_dir, "rouge_dec_dir")
    print_summary(ref_dir, dec_dir)
    