import os
import pickle
import argparse
import pandas as pd

from vocab import WordVocab

from utils import split_camel


def parse_args():

    parser = argparse.ArgumentParser("Path Vocab Generation!!")

    # 数据的路径
    parser.add_argument('--data_path', type=str, default='/.../APathCS/path_data/',help="data location")
    # 数据的类型
    parser.add_argument('--data_name', type=str, default='example', help="dataset name")
    # 语言的类型
    parser.add_argument('--lang_type', type=str, default='java', help="different code type")
    # 词典的大小
    parser.add_argument("--vocab_size", default=None, type=int)
    # 最小频率
    parser.add_argument("--min_freq", default=1, type=int)

    return parser.parse_args()


def main():
    # 参数配置
    args = parse_args()

    # /path_data/exmaple/java/
    lang_path = os.path.join(args.data_path, args.data_name, args.lang_type)

    # /data/hugang/DeveCode/PSCS/data/exmaple/java/code_path/test/java
    input_path = os.path.join(lang_path,os.path.join('code_path','train',args.lang_type))
    # /data/hugang/DeveCode/PSCS/data/exmaple/java/processed
    out_path = os.path.join(lang_path, 'processed')

    # 根据词典将token转为数字，
    f = open(os.path.join(out_path,'descr_vocab.pickle'), 'rb')
    descrs_vocab = pickle.load(f)
    print(descrs_vocab)
    f.close()

    def protoken_func(x):
        #  字符串切成列表
        split_list = split_camel(x)
        #  列表转成数字
        num_list = descrs_vocab.to_seq(split_list)
        #  拼接数字串
        return ' '.join([str(i) for i in num_list])

    '''
    id	token
    14	empty
    78	concatMapEagerDelayError
    70	offer
    23	CheckReturnValue
    '''

    #  对代码的tokens做分词处理
    tokens_paths = [os.path.join(lang_path,os.path.join('code_path', i, args.lang_type, 'tokens.csv')) for i in ['train','test']]

    for tokens_path in tokens_paths:
        # 读取csv数据
        tokens = pd.read_csv(tokens_path)
        tokens['token_cut'] = tokens['token'].apply(protoken_func)
        tokens['token_cut'] = tokens['token_cut'].astype(str)
        tokens.to_csv(tokens_path, index=False)

    #node_types 包含 path_contexts和path指定的词
    node_path = os.path.join(input_path, 'node_types.csv')
    nodes_vocab = pd.read_csv(node_path)
    nodes_vocab['node_type'] = nodes_vocab.apply(lambda x: '_'.join(x['node_type'].split()), axis=1)

    '''
               id                          node_type
           0   43             ExpressionStatement_UP
           1    3       SingleVariableDeclaration_UP
    '''

    node_idx = nodes_vocab.set_index('id').to_dict(orient='dict')
    node_dict = node_idx['node_type']

    '''
    {'node_type': {43: 'ExpressionStatement_UP',... 3: 'SingleVariableDeclaration_UP'}
    '''

    paths_path = os.path.join(input_path, 'paths.csv')
    paths = pd.read_csv(paths_path)
    #path为数字转字符，先构建字典在映射
    paths['path_tokens'] = paths.apply(lambda x: ' '.join([node_dict.get(int(i), '<unk>') for i in x['path'].split(' ')]), axis=1)

    '''
        id         path                                path_tokens
    0    308       1 2 0 0 7 5           SimpleName_UP SimpleType_UP <unk> <unk> Method...
    1    309       1 6 39 40 7 5         SimpleName_UP MethodInvocation_UP ClassInstanc...
    '''
    path_words= paths['path_tokens'].values.tolist()

    vocab = WordVocab(path_words, max_size=args.vocab_size, min_freq=args.min_freq)
    print("VOCAB SIZE:", len(vocab))

    vocab.save_vocab(os.path.join(out_path, 'path_vocab.pickle'))


if __name__ == '__main__':
    main()







