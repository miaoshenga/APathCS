
import os
import re
import argparse
import pickle
from tqdm import tqdm
from utils import split_camel
from utils import read_jsonl

from vocab import WordVocab


def get_1stsent(docstring):
    # 只取第一个的句子 [VARIABLE "my_destination_table"]
    if '.' in docstring:
        docstring = docstring.split('.')[0]
    elif '\n' in docstring:
        docstring = docstring.split('\n')[0]
    # [VARIABLE "my_destination_table"]
    return docstring


def str_filter(docstring):
    # [VARIABLE "my_destination_table"]
    # 若含有中文，直接返回' '
    if re.search(r'[\u4e00-\u9fa5@]', docstring) is not None:
        return ''
    # 去除一些特定字符
    docstring = docstring.replace('\n', ' ')
    docstring = re.sub(r'\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>', '', docstring)
    docstring = docstring.replace('_', ' ')
    docstring = re.sub(r'[^a-zA-Z0-9 ]', '', docstring)
    # VARIABLE my destination table
    # 分词处理
    docstring_tokens = split_camel(docstring)
    # 骆驼命名法分割拼接成字符串
    docstring = ' '.join(docstring_tokens) if len(docstring_tokens) > 2 else ' '
    return docstring


def make_descrs(jsonl_list):
    descrs = []
    length_count = []
    njsonl_list = []
    for line in jsonl_list:
        # 只取第一个句子  # 'docstring': 查询原始字符串
        first_sentence = get_1stsent(line['docstring'])
        docstring = str_filter(first_sentence)
        if len(docstring) < 2: # 为真跳过本次循环
            continue
        descrs.append(docstring)
        # 统计token个数
        length_count.append(len(re.findall(r' ', docstring)) + 1)
        # 过滤无用的字典
        njsonl_list.append(line)
    #  有效的的注释描述用\n拼接
    descrs = '\n'.join(descrs)

    # 把长度查询长度满足要求的line存储
    return descrs, njsonl_list


def save_corpus(lang_path, file_content, file_name):
    #  在java文件下创建processed
    if not os.path.exists(
            os.path.join(lang_path, 'processed')):
        os.mkdir(os.path.join(lang_path, 'processed'))
    # \n保留在txt文件中
    f = open(os.path.join(lang_path, 'processed', file_name),
             'w', encoding='utf8')
    f.write(file_content)
    f.close()



def save_file(lang_path, data_dict, prefix, type, start_point=0):
    for i, snippet in tqdm(enumerate(data_dict)):
        # 取出字典中的代码 ’code':代码的原始字符串
        code_str = snippet['code']
        # pathminer v0.3 do not support single function
        code_str = "package nothing; class Hi {%s}" % code_str

        if not os.path.exists(os.path.join(lang_path, 'code_files')):
            os.mkdir(os.path.join(lang_path, 'code_files'))

        file_path = os.path.join(lang_path, 'code_files', type)
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        f = open(os.path.join(file_path, '%s_%d.java'%(prefix, i+start_point)), 'w', encoding='utf8')
        f.write(code_str)
        f.close()


def str2file(lang_path, data_dict, prefix, type, k=None):
    if k is not None:
        if len(data_dict) > k:
            # 数据分片
            for i in range(0, len(data_dict), k):
                data_piece = data_dict[i:i+k]
                save_file(lang_path, data_piece, prefix, '%s_%d'%(type, i),start_point=i)
        else:
            save_file(lang_path, data_dict,  prefix, '%s_%d'%(type, 0), start_point=0)
    else:
        save_file(lang_path, data_dict, prefix, type)



def parse_args():

    parser = argparse.ArgumentParser("NL Extract Processing!!")
    # 数据的路径
    parser.add_argument('--data_path', type=str, default='/data/hugang/DeveCode/mydata/PathCS/',help="data location")
    # 数据的类型
    parser.add_argument('--data_name', type=str, default='github', help="dataset name")
    # 数据的分片
    parser.add_argument('--chunk_size', type=int, default=100000, help="data split size")
    # 语言的类型
    parser.add_argument('--lang_type', type=str, default='java', help="different code type")

    return parser.parse_args()


def main():
    # 配置
    args = parse_args()

    lang_path = os.path.join(args.data_path,args.data_name, args.lang_type)

    # jsons = {'test': [], 'train': [], 'valid': []}
    # jsons['train'], jsons['valid'], jsons['test']
    train, valid, test  = read_jsonl(os.path.join(lang_path, 'json_data'))

    # 前面注释描述拼接\n，后面是数据字典的列表
    descr_train, njsonl_train = make_descrs(train)
    descr_valid, njsonl_valid = make_descrs(valid)
    descr_test,  njsonl_test  = make_descrs(test)

    # 处理以后的
    print('train len:', len(njsonl_train))  #329968
    print('valid len:', len(njsonl_valid))  #8891
    print('test len:', len(njsonl_test))    #19016

    # 将注释描述按行写入txt文件
    save_corpus(lang_path, descr_train, 'descr_train.txt')
    save_corpus(lang_path, descr_valid, 'descr_valid.txt')
    save_corpus(lang_path, descr_test,  'descr_test.txt')

    str2file(lang_path, njsonl_train, 'train', 'train', k=args.chunk_size)
    str2file(lang_path, njsonl_valid, 'valid', 'valid')
    str2file(lang_path, njsonl_test,   'test', 'test')

    descr_corpus = descr_train.split('\n') + descr_valid.split('\n') + descr_test.split('\n')

    #  max_size 最大词数, min_freq 最小词频
    descr_vocab = WordVocab(descr_corpus, max_size=50000, min_freq=1)

    print("NL Vocab Size:", len(descr_vocab)) #42756


    # 以字典形式保存
    descr_vocab.save_vocab(os.path.join(lang_path, 'processed', 'descr_vocab.pickle'))
    print('des',descr_vocab)

    import pickle

    f = open(os.path.join(lang_path, 'processed', 'descr_vocab.pickle'), 'rb')
    nl_vocab = pickle.load(f)




if __name__ == '__main__':
    main()


