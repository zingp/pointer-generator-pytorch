from torch.autograd import Variable
import numpy as np
import torch
import config
from config import USE_CUDA, DEVICE
# 需要看一下batch对象到底是什么？
def get_input_from_batch(batch, USE_CUDA):
    """处理batch为模型的输入数据"""
    batch_size = len(batch.enc_lens)

    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
    # max_art_oovs is the max over all the article oov list in the batch
    if batch.max_art_oovs > 0:
        extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

    coverage = None
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    if USE_CUDA:
        enc_batch = enc_batch.to(DEVICE)
        enc_padding_mask = enc_padding_mask.to(DEVICE)

    if enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab = enc_batch_extend_vocab.to(DEVICE)
    if extra_zeros is not None:
        extra_zeros = extra_zeros.to(DEVICE)
    c_t_1 = c_t_1.to(DEVICE)

    if coverage is not None:
        coverage = coverage.to(DEVICE)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

def get_output_from_batch(batch, USE_CUDA):
    dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
    dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

    target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

    if USE_CUDA:
        dec_batch = dec_batch.to(DEVICE)
        dec_padding_mask = dec_padding_mask.to(DEVICE)
        dec_lens_var = dec_lens_var.to(DEVICE)
        target_batch = target_batch.to(DEVICE)

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

