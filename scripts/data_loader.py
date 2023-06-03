import os
import re
import torch
import tqdm
import random
import pickle
import pandas as pd

from torch.utils.data import Dataset


class Vocab():
    def __init__(self, my_dict, seq_max_len=None, pad_index=0,
                 id_vocab=None):
        self.pad_index = pad_index
        self.unknow_token = 1
        self.has_id = id_vocab is not None
        if seq_max_len is not None:
            self.seq_max_len = seq_max_len
            for key, value in my_dict.items():
                value_list = [int(a) for a in value.split()]
                if self.has_id:
                    value_list = [id_vocab.sos_index] + value_list + [id_vocab.eos_index]
                    value_list = value_list[:seq_max_len]
                    padding = [id_vocab.pad_index for _ in  range(seq_max_len - len(value_list))]
                    value_list.extend(padding)
                    my_dict[key] = value_list
                else:
                    padding = [self.pad_index for _ in   range(seq_max_len - len(value_list))]
                    value_list.extend(padding)
                    my_dict[key] = value_list
        self.itos = my_dict
        if isinstance(my_dict[1], str) and not self.has_id:
            self.stoi = {value: key for key, value in my_dict.items()}

    def to_id(self, sequence, unk=1):
        if isinstance(sequence, list):
            return [self.stoi.get(s, unk) for s in sequence]
        else:
            return self.stoi.get(sequence, unk)

    def to_str(self, sequence, is_path=False):
        unknown = [self.unknow_token] + [0 for _ in range(self.seq_max_len - 1)]  if is_path | self.has_id else self.unknow_token
        if isinstance(sequence, list):
            return [self.itos.get(i, unknown)
                    for i in sequence]
        else:
            return self.itos.get(sequence, unknown)


class Pairdata(Dataset):
    def __init__(self, descr_path, descr_vocab, patch_vocab, descr_max_len, path_max_len, code_path, k, is_train=True):
        super(Pairdata, self).__init__()


        self.descr_path  = descr_path
        self.descr_vocab = descr_vocab
        self.descr_max_len = descr_max_len

        self.path_vocab = patch_vocab
        self.path_max_len = path_max_len

        self.code_path = code_path

        self.k = k

        self.descr_lines = None

        # 设置代码的词典
        self.code = {}
        self.is_train = is_train

        #code_path = {}/path_data/test/
        def file_path(name, flag):
            if flag:
                return os.path.join(self.code_path, '..', 'train', 'java', '{}.csv'.format(name))
            else:
                return os.path.join(self.code_path, 'java', '{}.csv'.format(name))

        # path_contexts  {}/path_data/test/path_contexts
        f = open(file_path('path_contexts', False), 'r')
        path_contexts = f.read()
        f.close()
        self.code['path_contexts'] = self.context_process(path_contexts)

        # node_types   {}/path_data/train/node_types
        node_types = pd.read_csv(file_path('node_types', True), sep=',')
        self.code['node_types'] = self.df2vocab(node_types)
        # paths        {}/path_data/train/paths
        paths = pd.read_csv(file_path('paths', True), sep=',')
        self.code['paths'] = self.df2vocab(paths, max_len=self.path_max_len, id_vocab=self.path_vocab)
        # tokens       {}/path_data/train/tokens
        tokens = pd.read_csv(file_path('tokens', True), sep=',')
        self.code['tokens'] = self.df2vocab(tokens, id_vocab=self.descr_vocab)
        self.token_map = self.df2vocab(tokens[['id', 'token_cut']], max_len=self.descr_max_len, id_vocab=self.descr_vocab)

        with open(descr_path, "r", encoding='utf-8') as f:
            # line[:-1] 去掉换行符\n
            self.descr_lines = [line[:-1] for line in tqdm.tqdm(f, desc="Loading Dataset")]

        print('descr长度：',len(self.descr_lines))
        print('path_contexts长度：',len(self.code['path_contexts']))

        # 判断数据长度是否一致
        assert len(self.descr_lines) == len(self.code['path_contexts'])

        self.descr_tokenized = self.tokenize(self.descr_lines, self.descr_vocab, self.descr_max_len)

        self.corpus_length = len(self.code['path_contexts'])

    def __len__(self):
        return self.corpus_length

    def __getitem__(self, item):
        data = self.get_data(item)

        if self.is_train:
            trg_item = self.get_random_item(item)
            trg_data = self.get_data(trg_item)
            return data, trg_data
        else:
            return data

    '''
    self.pad_index = 0
    self.unk_index = 1
    self.eos_index = 2
    self.sos_index = 3
    '''

    def tokenize(self, corpus, vocab,  maxlen):
        rtn = []
        for line in corpus:
            # 字符串以空格拼接在切分即为词组
            tokens = line.split()
            for i in range(len(tokens)):
                # 词典里取索引，不存在就unk
                tokens[i] = vocab.stoi.get(tokens[i], vocab.unk_index)
            # [3,.....2]
            value_list = [vocab.sos_index] + tokens + [vocab.eos_index]
            # 长了截断短了补0
            value_list = value_list[:maxlen]
            # 不足的补0
            padding = [vocab.pad_index for _ in range(maxlen - len(value_list))]
            value_list.extend(padding)
            rtn.append(value_list)
        return rtn

    def get_random_item(self, item):
        rand_item = item
        while rand_item is item:
            # 从[0,corpus_length]任意取一个数字
            rand_item = random.randrange(self.corpus_length)
        return rand_item

    def get_data(self, item):

        code_id = self.code['path_contexts'][item][0]
        path_context = self.code['path_contexts'][item][1]

        descr = self.descr_tokenized[code_id]

        start_tokens = []
        paths = []
        end_tokens = []

        for i in random.sample(path_context, self.k):

            start_token, path, end_token = i.split(',')

            paths.append(self.code['paths'].to_str(int(path), is_path=True))
            start_tokens.append(self.token_map.to_str(int(start_token)))
            end_tokens.append(self.token_map.to_str(int(end_token)))

        data = {"descr": torch.tensor(descr),
                "paths": torch.tensor(paths),
                "start_tokens": torch.tensor(start_tokens),
                "end_tokens": torch.tensor(end_tokens),
                }

        return data

    def df2vocab(self, d, max_len=None, id_vocab=None):
        return Vocab({int(row[0]): row[1] for i, row in d.iterrows()},
                     max_len, id_vocab=id_vocab)

    def context_process(self, context):
        context = context.split('\n')
        rtn = []
        match_num = re.compile(r'(?<=_)[0-9]+(?=\.)')
        for i, row in enumerate(context[:-1]):
            row = row.split(' ')
            id = int(re.search(match_num, row[0]).group())
            values = row[1:]
            if len(values) < self.k:
                padding = ['0,0,0' for _ in
                           range(self.k - len(values))]
                values.extend(padding)
            rtn.append((id, values))
        return rtn


def main():
    # 输入数据的路径
    lang_path= '/data/hugang/DeveCode/mydata/PathCS/github/java'
    # lang_path= '/data/hugang/DeveCode/mydata/PathCS/example/java'

    f = open('{}/processed/descr_vocab.pickle'.format(lang_path), 'rb')
    descr_vocab = pickle.load(f)
    f.close()
    f = open('{}/processed/path_vocab.pickle'.format(lang_path), 'rb')
    path_vocab = pickle.load(f)
    f.close()

    print('-------------------------训练数据集！！-------------------------')
    # 训练数据的加载 train_set和train_data_loader 都可以用len
    train_set = Pairdata('{}/processed/descr_train.txt'.format(lang_path),
                         descr_vocab,path_vocab,
                        20 ,12,'{}/code_path/train'.format(lang_path),
                        40, is_train=True)

    train_iter = torch.utils.data.DataLoader(dataset=train_set, batch_size=100, shuffle=True, num_workers=12)

    print(len(train_iter))

    print('-------------------------验证数据集！！-------------------------')
    # 训练数据的加载 valid_set和valid_data_loader 都可以用len
    valid_set = Pairdata('{}/processed/descr_test.txt'.format(lang_path),
                         descr_vocab,path_vocab,
                        20, 12, '{}/code_path/test'.format(lang_path),
                        40, is_train=False)

    valid_iter = torch.utils.data.DataLoader(dataset=valid_set, batch_size=100, shuffle=True, num_workers=1)

    print(len(valid_iter))


if __name__ == '__main__':
    main()




