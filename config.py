import os
import torch
root_dir = "/search/odin/liuyouyuan/pyproject/data/finished_files"
#root_dir = "/search/odin/liuyouyuan/pyproject/data/weibo/finished_files"
train_data_path = os.path.join(root_dir, "chunked/train_*")
eval_data_path = os.path.join(root_dir, "val.bin")
decode_data_path = os.path.join(root_dir, "test.bin")
vocab_path = os.path.join(root_dir, "vocab")
log_root = "./cnn_coverage_log"
# log_root = "./weibo_log"

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 16
max_enc_steps=400
#max_enc_steps=200
max_dec_steps=100
#max_dec_steps=40
beam_size=4
min_dec_steps=35
#min_dec_steps=20
vocab_size=50_000

lr=0.15
adam_lr = 0.001    # 使用adam时候的学习率
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
# is_coverage = False
is_coverage = True
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 520_000

lr_coverage=0.15

# 使用GPU相关
use_gpu=True
USE_CUDA = use_gpu and torch.cuda.is_available()     # 是否使用GPU
NUM_CUDA = torch.cuda.device_count()
DEVICE = torch.device("cuda:4" if USE_CUDA else 'cpu')
