import os
import sys

# 添加project目录至环境变量
base_dir = os.path.abspath(os.path.dirname(__file__))
print(base_dir)
sys.path.append(base_dir)

import time
import argparse

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

import config
from data import Vocab
from utils import calc_running_avg_loss
from config import USE_CUDA, DEVICE
from batcher import Batcher
from batcher import get_input_from_batch
from batcher import get_output_from_batch
from adagrad_custom import AdagradCustom


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        train_dir = os.path.join(config.log_root, 'train_{}'.format(stamp))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter_step):
        """保存模型"""
        state = {
            'iter': iter_step,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
        model_save_path = os.path.join(self.model_dir, 'model_{}_{}'.format(iter_step, stamp))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        """模型初始化或加载、初始化迭代次数、损失、优化器"""
        # 初始化模型
        self.model = Model(model_file_path)
        # 模型参数的列表
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        # 定义优化器
        # self.optimizer = optim.Adam(params, lr=config.adam_lr)
        # 使用AdagradCustom做优化器
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        # 初始化迭代次数和损失
        start_iter, start_loss = 0, 0
        # 如果传入的已存在的模型路径，加载模型继续训练
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location = lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if USE_CUDA:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(DEVICE)

        return start_iter, start_loss

    def train_one_batch(self, batch):
        """
        训练一个batch，返回该batch的loss。
        enc_batch:             torch.Size([16, 400]), 16篇文章的编码，不足400词的用pad的编码补足, oov词汇用0编码；
        enc_padding_mask:      torch.Size([16, 400]), 对应pad的位置为0，其余为1；
        enc_lens:              numpy.ndarray, 列表内每个元素表示每篇article的单词数；
        enc_batch_extend_vocab:torch.Size([16, 400]), 16篇文章的编码;oov词汇用超过词汇表的编码；
        extra_zeros:           torch.Size([16, 文章oov词汇数量]) zero tensor;
        c_t_1:                 torch.Size([16, 512]) zero tensor;
        coverage:              Variable(torch.zeros(batch_size, max_enc_seq_len)) if is_coverage==True else None;coverage模式时后续有值
        ----------------------------------------
        dec_batch:             torch.Size([16, 100]) 摘要编码含有开始符号编码以及PAD；
        dec_padding_mask:      torch.Size([16, 100]) 对应pad的位置为0，其余为1；
        max_dec_len:           标量，摘要词语数量，不包含pad
        dec_lens_var:          torch.Size([16] 摘要词汇数量         
        target_batch:          torch.Size([16, 100]) 目标摘要编码含有STOP符号编码以及PAD
        """
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch)
        self.optimizer.zero_grad()
        """
        # 记得修改Batch类添加vocab属性
 
        print("模型输入文章编码:", "*"*100)
        print("enc_batch:", enc_batch, enc_batch.size())
        print("enc_batch[-1]:", enc_batch[-1])
        # print("batch._id_to_word:", batch.vocab._id_to_word)
        print("enc_batch[-1]原文:", [batch.vocab.id2word(idx) for idx in enc_batch[-1].cpu().numpy()])
        print("-"*50)
        print("enc_padding_mask:", enc_padding_mask, enc_padding_mask.size())
        print("-"*50)
        print("enc_lens:", enc_lens, enc_lens.shape)
        print("-"*50)
        print("enc_batch_extend_vocab", enc_batch_extend_vocab, enc_batch_extend_vocab.size())
        print("enc_batch_extend_vocab[-1]:", enc_batch_extend_vocab[-1])
        print("enc_batch_extend_vocab[-1]的原文:", [batch.vocab.id2word(idx) if idx<50000 else '[UNK]+{}'.format(idx-50000) for idx in enc_batch_extend_vocab[-1].cpu().numpy()])
        print("-"*50)
        print("extra_zeros:", extra_zeros, extra_zeros.size())
        print("-"*50)
        print("c_t_1:", c_t_1, c_t_1.size())
        print("-"*50)
        print("coverage:", coverage)
        print("*"*100)
        
        print("模型输入摘要编码，包括源和目标：", "*"*100)
        print("dec_batch:", dec_batch, dec_batch.size())
        print("dec_batch[0]:", dec_batch[0])
        # print("batch._id_to_word:", batch.vocab._id_to_word)
        print("dec_batch[0]原文:", [batch.vocab.id2word(idx) for idx in dec_batch[0].cpu().numpy()])
        print("-"*50)
        print("dec_padding_mask:", dec_padding_mask, dec_padding_mask.size())
        print("-"*50)
        print("max_dec_len:", max_dec_len)
        print("-"*50)
        print("dec_lens_var", dec_lens_var, dec_lens_var.size())
        print("-"*50)
        print("target_batch:", target_batch, target_batch.size())
        print("-"*50)
        print("target_batch[0]:", target_batch[0], target_batch[0].size())
        print("target_batch[0]的原文:", [batch.vocab.id2word(idx) if idx<50000 else '[UNK]+{}'.format(idx-50000) for idx in target_batch[0].cpu().numpy()])
        print("*"*100)
        input("任意键继续>>>")
        """
        # [B, max(seq_lens), 2*hid_dim], [B*max(seq_lens), 2*hid_dim], tuple([2, B, hid_dim], [2, B, hid_dim])
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)   # (h,c) = ([1, B, hid_dim], [1, B, hid_dim])
        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]      # 摘要的一个单词，batch里的每个句子的同一位置的单词编码
            # print("y_t_1:", y_t_1, y_t_1.size())
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]  # 摘要的下一个单词的编码
            # print("target-iter:", target, target.size())
            # print("final_dist:", final_dist, final_dist.size())
            # input("go on>>")
            # final_dist 是词汇表每个单词的概率，词汇表是扩展之后的词汇表，也就是大于预设的50_000
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()   # 取出目标单词的概率gold_probs
            step_loss = -torch.log(gold_probs + config.eps)  # 最大化gold_probs，也就是最小化step_loss（添加负号）
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        # 训练设置，包括
        iter_step, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter_step < n_iters:
            # 获取下一个batch数据
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter_step)
            iter_step += 1

            if iter_step % 100 == 0:
                self.summary_writer.flush()
            
            # print_interval = 1000
            if iter_step % 1000 == 0:
                # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                print('steps %d, seconds for %d steps: %.2f, loss: %f' % (iter_step, 1000,
                                                                          time.time() - start, loss))
                start = time.time()
            # 5000次迭代就保存一下模型
            if iter_step % 1000 == 0:
                self.save_model(running_avg_loss, iter_step)

def init_print():
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    print("时间:{}".format(stamp))
    print("***参数:***")
    for k, v in config.__dict__.items():
        if not k.startswith("__"):
            print(":".join([k, str(v)]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    init_print()
    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_path)
