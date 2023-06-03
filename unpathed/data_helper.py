# -*- coding:UTF-8 -*-
import numpy as np
import pickle



def load_embedding(dstPath):
    #加载词向量矩阵
    with open(dstPath, 'rb') as fin:
        embeddings = pickle.load(fin)
    return embeddings


class Iterator(object):
    def __init__(self, x):
        self.x = x
        self.sample_num = len(x)

    def next_batch(self, batch_size, shuffle = True):
        if shuffle:
            np.random.shuffle(self.x)
        l = np.random.randint(0,self.sample_num - batch_size + 1)
        r = l + batch_size
        x_part = self.x[l:r]
        return x_part

    def next(self, batch_size, shuffle = False):
        if shuffle:
            np.random.shuffle(self.x)
        l = 0
        while l < self.sample_num:
            r = min(l + batch_size, self.sample_num)
            x_part = self.x[l:r]
            l += batch_size
            yield x_part


class Batch:
    def __init__(self):
        self.q_id  =[]
        self.query= []
        self.c_id = []
        self.code = []
        self.q_mask=[]
        self.c_mask=[]
        self.label= []


def transform(fin_path, vocab_path, unk_id =1):
    '''
    load the vocab txt and build a word2id dict.
    then change the corpus from word to index using the word2id dict.

    :return: list-like data containg [q_id,q_idx,a_id,a_idx,int(label)]
    '''
    transformed_data=[]
    word2id = {}

    # 首先构造word2id的字典~
    with open(vocab_path,'r',encoding='utf-8')as f:
        for line in f:
            id = int(line.strip('\n').split('\t')[0])
            word = line.strip('\n').split('\t')[1]
            word2id[word] = id

    with open(fin_path,'r',encoding='utf-8')as f2:
        for line in f2:
            q_id, query, c_id, code,label = line.split('\t')
            #不在词典中返回1 视为unk
            q_idx = [word2id.get(word,unk_id) for word in query.split()]
            c_idx = [word2id.get(word,unk_id) for word in code.split()]
            transformed_data.append([q_id,q_idx,c_id,c_idx,int(label)])

    return transformed_data



def transform_train(fin_path, vocab_path, unk_id =1):
    transformed_data = []
    word2id = {}

    with open(vocab_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            id = int(line.strip('\n').split('\t')[0])
            word = line.strip('\n').split('\t')[1]
            word2id[word] = id

    with open(fin_path, 'r', encoding='utf-8')as f2:
        for line in f2:
            # 语料三元组（正样本，查询，负样本）
            pos_code, query, neg_code = line.split('\t')
            # 不在词典中返回1 视为unk
            pos_c_idx = [word2id.get(word, unk_id) for word in pos_code.split()]  #代码正样本
            q_idx = [word2id.get(word, unk_id) for word in query.split()]         #查询
            neg_c_idx = [word2id.get(word, unk_id) for word in neg_code.split()]  #代码负样本
            transformed_data.append([pos_c_idx,q_idx,neg_c_idx])

    return transformed_data



def padding(sent, max_length):
    #若max_lengt大于序列长度同样满足
    if len(sent) > max_length:
        sent = sent[:max_length]
    #包含截断和延长
    pad_length = max_length - len(sent)
    #pad之处补0
    sent2idx= sent + [0] * pad_length
    #pad之后句子的id及原始句子的长度
    return sent2idx, len(sent)


def load_data(transformed_corpus, query_len, code_len, keep_ids = False):

    padded_corpus = []

    if keep_ids:
        for sample in transformed_corpus:
            q_id, query, c_id, code, label = sample
            query, q_mask=padding(query, query_len)
            code, c_mask =padding(code, code_len)
            padded_corpus.append((q_id, query, c_id, code, q_mask, c_mask, label))
    else:
        for sample in transformed_corpus:
            q_id, query, c_id, code, label = sample
            query, q_mask=padding(query, query_len)
            code, c_mask =padding(code, code_len)
            padded_corpus.append((query, code, q_mask, c_mask, label))

    return padded_corpus


def load_train_data(transformed_corpus,query_len,code_len):

    padded_corpus =[]

    for sample in transformed_corpus:
        pos_code, query, neg_code = sample

        pos_code, pos_c_mask = padding(pos_code, code_len)
        query, q_mask = padding(query, query_len)
        neg_code, neg_c_mask = padding(neg_code, code_len)

        padded_corpus.append((pos_code,query,neg_code,pos_c_mask,q_mask,neg_c_mask))

    return padded_corpus


