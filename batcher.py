import time
import random
import torch
import numpy as np
import tensorflow as tf
from random import shuffle
from queue import Queue
from threading import Thread
from torch.autograd import Variable

import data
import config
from config import USE_CUDA, DEVICE

random.seed(1234)

class Example(object):

    def __init__(self, article, abstract_sentences, vocab):
        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        # 处理article，如果超过配置文件中的长度，截断。
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(article_words) # store the length after truncation but before padding
        # 编码 article，包括oov单词也得跟着编码
        self.enc_input = [vocab.word2id(w) for w in article_words]
        # 处理 abstract
        abstract = ' '.join(abstract_sentences)  # string
        abstract_words = abstract.split()        # list of strings
        # 编码 abstract
        abs_ids = [vocab.word2id(w) for w in abstract_words] # 

        # 构建解码阶段的输入序列和输出序列“strat w1 w2”, "w1 w2 end",要一样长
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # 如果使用pointer-generator模式, 需要一些额外信息
        if config.pointer_gen:
            # 编码时需要输入原文编码和oov单词的编码
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

            # 获取参考摘要的id，其中oov单词由原文中的oov单词编码表示
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            # 目标编码和处理oov
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)

        # 存储原始数据
        self.original_article = article
        self.original_abstract = abstract
        # 编码前的摘要，单词列表
        self.original_abstract_sents = abstract_sentences


    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # 截断
            inp = inp[:max_len]
            target = target[:max_len] # 没有结束标志
        else:   # 无截断
            target.append(stop_id)    # 结束标志
        assert len(inp) == len(target)
        return inp, target


    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)


    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        # self.vocab = vocab               # 添加这个用来测试
        self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list) # initialize the input to the encoder
        self.init_decoder_seq(example_list) # initialize the input and targets for the decoder
        self.store_orig_strings(example_list) # store the original strings


    def init_encoder_seq(self, example_list):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list] # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists


class Batcher(object):
    # _batch_queue队列的最大长度
    BATCH_QUEUE_MAX = 100 

    def __init__(self, data_path, vocab, mode, batch_size, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        # 初始化一个存放Batch的队列，和一个存放Examples的队列。后面会用，注意队列的大小关系。
        self._batch_queue = Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue(self.BATCH_QUEUE_MAX * self.batch_size)
        # 是否使用single_pass模式
        if single_pass:
            self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1   # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False  # 这个标志是否已经读完数据，True表示已经读完
        else:
            self._num_example_q_threads = 1  # 16 # num threads to fill example queue
            self._num_batch_q_threads = 1    #4  # num threads to fill batch queue
            self._bucketing_cache_size = 1   #100 # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            # 这里相当于循环一次
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            # 设置为守护线程，只有守护线程都终结，整个python程序才会退出
            self._example_q_threads[-1].daemon = True
            # 开始执行，相当于调用self.fill_example_queue()
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        # 如果_batch_queue队列为空，则打印警告，并返回None，结束训练
        if self._batch_queue.qsize() == 0:
            tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get() # get the next Batch
        return batch

    def fill_example_queue(self):
        # 创建一个生成器对象
        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

        while True:
            try:
                (article, abstract) = input_gen.__next__() # read the next example from file. article and abstract are both strings.
                article, abstract = article.decode(), abstract.decode()
            except StopIteration: # if there are no more examples:
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] # 编码abstract
            #print("abstract_sentences:", abstract_sentences)
            example = Example(article, abstract_sentences, self._vocab)  # 处理成一个Example.
            self._example_queue.put(example)  # 放处理成一个Example对象至example queue.
            """
            print("*****Example*****")
            print("="*100)
            print("Article", example.original_article, type(example.original_article))
            print("-"*80)
            print("enc_input:", example.enc_input)
            print("-"*80)
            print("Abstract Words", example.original_abstract, type(example.original_abstract))
            print("-"*80)
            print("dec_input:", example.dec_input, len(example.dec_input))
            print("-"*80)
            print("Target:", example.target, len(example.target))
            print("="*100)
            """

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                # beam search decode mode single example repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
            else:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True) # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size))

    def watch_threads(self):
        while True:
            tf.logging.info(
            'Bucket queue size: %i, Input queue size: %i',
            self._batch_queue.qsize(), self._example_queue.qsize())

            time.sleep(60)
            for idx,t in enumerate(self._example_q_threads):
                if not t.is_alive(): # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx,t in enumerate(self._batch_q_threads):
                if not t.is_alive(): # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()


    def text_generator(self, example_generator):
        while True:
            e = example_generator.__next__()     # e 是一个 tf.Example对象
            try:
                article_text = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
                abstract_text = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue

            if len(article_text) == 0: # article为空的example对象就跳过
                # tf.logging.warning('Found an example with empty article text. Skipping it.')
                continue
            else:
                yield (article_text, abstract_text)


# 解析Batch对象
def get_input_from_batch(batch):
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

def get_output_from_batch(batch):
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