python3 eval.py /root/liuyouyuan/pyproject/pointer-generator-pytorch/log/train_1569842742/model/model_5000_1569846703


nohup python decode.py ./weibo_log/train_1570885369/model/model_415000_20191013_104358 >weibo_no_coverage.log 2>&1 &

nohup python train.py -m weibo_log/train_1570973725/model/model_495000_20191014_111027 >wb_coverage_log  2>&1 &

# 20200103
nohup python train.py >logs/log_gen_nocover_adam.log 2>&1 &

nohup python train.py -m logs/weibo_adam/train_20200105_094159/model/model_490000_20200105_165432 >logs/log_gen_cover.log 2>&1 &

nohup python decode.py logs/weibo_adam/train_20200106_110339/model/model_494000_20200106_110724 >logs/log_gen_cover_494_dec.log 2>&1 &

"""
(pt-tf-env) [dc@gz_6237_gpu pointer-generator-pytorch]$ python rouge_zh.py
ROUGE-1 : 0.4933
ROUGE-2 : 0.234
(pt-tf-env) [dc@gz_6237_gpu pointer-generator-pytorch]$ python rouge.py
{'rouge_1/f_score': 0.23396664178905938,
 'rouge_1/p_score': 0.23396664295893074,
 'rouge_1/r_score': 0.23396664295893074,
 'rouge_2/f_score': 0.15545508691115734,
 'rouge_2/p_score': 0.1554550876884549,
 'rouge_2/r_score': 0.1554550876884549,
 'rouge_l/f_score': 0.23396664295884423,
 'rouge_l/p_score': 0.23396664295893074,
 'rouge_l/r_score': 0.23396664295893074}
"""


# 20200106 20:29
nohup python train.py >logs/log_gen_nocover_adagrad.log 2>&1 &

nohup python train.py -m logs/weibo_adagrad/train_20200106_202839/model/model_700000_20200107_062401 >logs/log_gen_cover_adagrad.log 2>&1 &

nohup python decode.py logs/weibo_adagrad/train_20200107_093425/model/model_704000_20200107_093803 >log_gen_cover_adagrad_704_dec.log 2>&1 &

"""
(pt-tf-env) [dc@gz_6237_gpu pointer-generator-pytorch]$ python rouge_zh.py
ROUGE-1 : 0.4745
ROUGE-2 : 0.2187
(pt-tf-env) [dc@gz_6237_gpu pointer-generator-pytorch]$ python rouge.py
{'rouge_1/f_score': 0.22685453430889232,
 'rouge_1/p_score': 0.22685453544319967,
 'rouge_1/r_score': 0.22685453544319967,
 'rouge_2/f_score': 0.1545234846036691,
 'rouge_2/p_score': 0.15452348537630375,
 'rouge_2/r_score': 0.15452348537630375,
 'rouge_l/f_score': 0.22685453544311474,
 'rouge_l/p_score': 0.22685453544319967,
 'rouge_l/r_score': 0.22685453544319967}
"""
