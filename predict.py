"""
预处理：构建一个Batch对象
    1 首先得到article的分词
    构建一个example Example(art_str, ["",], vocab)
        art_str 是分词后用空格连起来的字符串
        Example的第二个参数是abstract的分词空格连接在一起的各个句子列表[句子1，句子2.。。]
    2 构建一个Batch
    [ex1,ex2...] + vocab + bach_size 构建一个Batch
"""
import jieba
import os
import sys
import time

import torch
from torch.autograd import Variable

import data, config
from config import USE_CUDA, DEVICE
from data import Vocab
from model import Model
from batcher import Example
from batcher import Batch
from batcher import get_input_from_batch

def build_batch_by_article(article, vocab):
    words = jieba.cut(article)
    art_str = " ".join(words)
    example = Example(art_str, ["",], vocab)
    ex_list = [example for _ in range(config.beam_size)]
    batch  = Batch(ex_list, vocab, config.beam_size)
    return batch

"""
decode阶段使用 beam search 算法
"""
class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens = self.tokens + [token],
                            log_probs = self.log_probs + [log_prob],
                            state = state,
                            context = context,
                            coverage = coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path, vocab):
        self.vocab = vocab
        # 加载模型
        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self, batch):
        best_summary = self.beam_search(batch)

        # Extract the output ids from the hypothesis and convert back to words
        output_ids = [int(t) for t in best_summary.tokens[1:]]
        decoded_words = data.outputids2words(output_ids, self.vocab,
                                                (batch.art_oovs[0] if config.pointer_gen else None))

        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING)
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words
        print("decode_words:", decoded_words)
        return "".join(decoded_words)


    def beam_search(self, batch):
        # batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if USE_CUDA:
                y_t_1 = y_t_1.to(DEVICE)
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

if __name__ == '__main__':
    article = "近日，一段消防员用叉子吃饭的视频在网上引起热议。原来是因为训练强度太大，半天下来，大家拿筷子的手一直在抖，甚至没法夹菜。于是，用叉子吃饭，渐渐成了上海黄浦消防车站中队饭桌上的传统。转发，向消防员致敬！"
    model_path = sys.argv[1]
    
    vocab = Vocab(config.vocab_path, config.vocab_size)
    batch = build_batch_by_article(article, vocab)
    beam_processor = BeamSearch(model_path, vocab)
    beam_processor.decode(batch)