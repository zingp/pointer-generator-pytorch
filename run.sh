python3 eval.py /root/liuyouyuan/pyproject/pointer-generator-pytorch/log/train_1569842742/model/model_5000_1569846703


nohup python decode.py ./weibo_log/train_1570885369/model/model_415000_20191013_104358 >weibo_no_coverage.log 2>&1 &

nohup python train.py -m weibo_log/train_1570973725/model/model_495000_20191014_111027 >wb_coverage_log  2>&1 &

# 20200103
nohup python train.py >logs/log_gen_nocover_adam.log 2>&1 &

nohup python train.py -m logs/weibo_adam/train_20200105_094159/model/model_490000_20200105_165432 >logs/log_gen_cover.log 2>&1 &
