import os
import tqdm
import argparse
import pandas as pd


max_row = 0

def trans(value, dict, unknown=0):
    new_value = dict.get(int(value), unknown)

    if pd.isna(new_value):
        new_value = unknown
    return str(int(new_value))


def transform_paths(row, map_dict):
    paths_ids = row['path'].split()
    new_ids = []
    for id in paths_ids:
        new_id = trans(id, map_dict)
        new_ids.append(new_id)
    new_path_id = ' '.join(new_ids)
    row['path'] = new_path_id
    return row


def transform_paths_content(row, token_map, path_map):
    row = row.split(',')
    start_token_id = row[0]
    path_id = row[1]
    end_token_id = row[2]
    new_start_token_id = trans(start_token_id, token_map)
    new_path_id = trans(path_id, path_map)
    new_end_token_id = trans(end_token_id, token_map)
    return '{},{},{}'.format(new_start_token_id, new_path_id,
                             new_end_token_id)


def fill_nan(vocab_map):
    global max_row
    max_row = vocab_map['id_y'].max()

    def apply_new_id(row):
        global max_row
        if pd.isna(row['id_y']):
            row['id_y'] = int(max_row + 1)
            max_row = row['id_y']
        return row
    vocab_map = vocab_map.apply(apply_new_id, axis=1)
    return vocab_map


def vocab_merge(vocab_a, vocab_b, on, method):
    vocab = vocab_a.merge(vocab_b, on=on, how=method)
    if method == 'outer':
        vocab = fill_nan(vocab)
    return vocab


def save_vocab(vocab, path, columns=None):
    vocab = vocab.iloc[:, 1:]
    if columns is not None:
        vocab.columns = columns
    try:
        vocab = vocab[[columns[1], columns[0]]].astype({'id': 'int32'})
    except ValueError:
        print(vocab)
    vocab.to_csv(path, index=False)


def map2dict(vocab_map):
    map_dict = {}
    for i, row in vocab_map.iterrows():
        if pd.isna(row[0]):
            continue
        map_dict[int(row[0])] = row[2]
    return map_dict


def parse_args():

    parser = argparse.ArgumentParser("CodePath Vocab Generation!!")
    # 数据的路径
    parser.add_argument('--data_path', type=str, default='/data/hugang/DeveCode/mydata/PathCS/',help="data location")
    # 数据的类型
    parser.add_argument('--data_name', type=str, default='example', help="dataset name")
    # 语言的类型
    parser.add_argument('--lang_type', type=str, default='java', help="different code type")
    # 数据的分片
    parser.add_argument('--train_path', type=str, default='train_i', help="train path dataset")
    # 语言的类型
    parser.add_argument('--test_path', type=str, default='test', help="test path dataset")
    # 输出的目录
    parser.add_argument('--out_path', type=str, default=' ', help="path output")

    parser.add_argument("--merge_vocab", type=bool, default=False, help="need merge vocab")

    return parser.parse_args()



def main():
    # 配置
    args = parse_args()
    # /data/hugang/DeveCode/mydata/PathCS/XXX/java
    lang_path = os.path.join(args.data_path, args.data_name, args.lang_type)

    #  训练数据的路径
    train_path = os.path.join(lang_path, args.train_path)
    #  测试数据的路径
    test_path  = os.path.join(lang_path,  args.test_path)
    #  输出文件的目录
    out_path  =  os.path.join(lang_path, args.out_path)

    if not os.path.exists(out_path):
        # 创建 code_path/train
        os.makedirs(out_path)

    #  训练集
    token_vocab_train = pd.read_csv(os.path.join(train_path, 'tokens.csv'))
    node_vocab_train =  pd.read_csv(os.path.join(train_path, 'node_types.csv'))
    path_vocab_train =  pd.read_csv(os.path.join(train_path, 'paths.csv'))
    #  测试集
    token_vocab_test = pd.read_csv(os.path.join(test_path, 'tokens.csv'))
    node_vocab_test  = pd.read_csv(os.path.join(test_path, 'node_types.csv'))
    path_vocab_test  = pd.read_csv(os.path.join(test_path, 'paths.csv'))

    need_merge = args.merge_vocab

    method = 'outer' if need_merge else 'left'
    
    node_vocab_map  = vocab_merge(node_vocab_test, node_vocab_train, on=['node_type'], method=method)
    token_vocab_map = vocab_merge(token_vocab_test,token_vocab_train, on=['token'], method='outer')

    node_dict  = map2dict(node_vocab_map)
    token_dict = map2dict(token_vocab_map)

    path_vocab_test = path_vocab_test.apply(lambda row: transform_paths(row, node_dict), axis=1)

    path_vocab_map = vocab_merge(path_vocab_test, path_vocab_train, on=['path'], method='outer')

    path_dict = map2dict(path_vocab_map)

    path_context_test = []
    for root, dirs, files in os.walk(test_path):
        for f_name in tqdm.tqdm(files):
            if 'path_contexts' in f_name:
                f_path = os.path.join(root, f_name)
                with open(f_path) as f:
                    f_list = f.readlines()
                for row in f_list:
                    path_list = row.split()
                    id = path_list[0]
                    paths = path_list[1:]
                    new_paths = []
                    for path_item in paths:
                        new_path = transform_paths_content(path_item, token_dict, path_dict)
                        new_paths.append(new_path)
                    new_row = ' '.join([str(id)] + new_paths) + '\n'
                    path_context_test.append(new_row)
    if need_merge:
        path_context_train = []
        for root, dirs, files in os.walk(train_path):
            for f_name in tqdm.tqdm(files):
                if 'path_contexts' in f_name:
                    f_path = os.path.join(root, f_name)
                    with open(f_path) as f:
                        f_list = f.readlines()
                    path_context_train = path_context_train + f_list
        path_context_train = path_context_test + path_context_train
        f = open(os.path.join(out_path, 'path_contexts.csv'), 'w')
        f.write(''.join(path_context_train))
        f.close()
        save_vocab(node_vocab_map, os.path.join(out_path, 'node_types.csv'),
                   columns=['node_type', 'id'])
        save_vocab(token_vocab_map, os.path.join(out_path, 'tokens.csv'),
                   columns=['token', 'id'])
        save_vocab(path_vocab_map, os.path.join(out_path, 'paths.csv'),
                   columns=['path', 'id'])
    else:
        f = open(os.path.join(out_path, 'path_contexts.csv'), 'w')
        f.write(''.join(path_context_test))
        f.close()
        save_vocab(path_vocab_map, os.path.join(train_path, 'paths.csv'),
                   columns=['path', 'id'])
        save_vocab(token_vocab_map, os.path.join(train_path,'tokens.csv'),
                   columns=['token', 'id'])


if __name__ == '__main__':
    main()
