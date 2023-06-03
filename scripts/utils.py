import os
import re
import jsonlines

import numpy as np


def read_jsonl(dir_path):
    #  语料集的类别
    dir_name = ['test', 'train', 'valid']
    jsons = {'test': [], 'train': [], 'valid': []}
    for type in dir_name:
        files = os.listdir(os.path.join(dir_path, type))
        # 对文件名排序
        nfiles =  sorted(files, key=lambda x: int(x.split('_')[2].rstrip('.jsonl')))
        # 分别遍历['test', 'train', 'valid'] 文件夹
        for file_name in nfiles:
            f = open(os.path.join(dir_path, type, file_name), 'r+',encoding='utf8')
            for line in jsonlines.Reader(f):
                # 'language': 程序语言类型，’partition: 数据类型
                # ‘path': 原始文件的完整路径
                # ‘func_name'：函数名和方法名，
                # 'original_string’:原始字符串
                # 'docstring': 查询原始字符串 （顶级注释或文档字符串（如果原始字符串中存在）
                # 'docstring_tokens'： 查询的分词
                # ’code':代码的原始字符串，
                # ‘code_tokens' : 代码的分词
                '''{'repo': 'google/guava', 'path': 'android/guava/src/com/google/common/io/CharSource.java', 'func_name': 'CharSource.isEmpty', 'original_string': 'public boolean isEmpty() throws IOException {\n    Optional<Long> lengthIfKnown = lengthIfKnown();\n    if (lengthIfKnown.isPresent()) {\n      return lengthIfKnown.get() == 0L;\n    }\n    Closer closer = Closer.create();\n    try {\n      Reader reader = closer.register(openStream());\n      return reader.read() == -1;\n    } catch (Throwable e) {\n      throw closer.rethrow(e);\n    } finally {\n      closer.close();\n    }\n  }', 'language': 'java', 'code': 'public boolean isEmpty() throws IOException {\n    Optional<Long> lengthIfKnown = lengthIfKnown();\n    if (lengthIfKnown.isPresent()) {\n      return lengthIfKnown.get() == 0L;\n    }\n    Closer closer = Closer.create();\n    try {\n      Reader reader = closer.register(openStream());\n      return reader.read() == -1;\n    } catch (Throwable e) {\n      throw closer.rethrow(e);\n    } finally {\n      closer.close();\n    }\n  }', 'code_tokens': ['public', 'boolean', 'isEmpty', '(', ')', 'throws', 'IOException', '{', 'Optional', '<', 'Long', '>', 'lengthIfKnown', '=', 'lengthIfKnown', '(', ')', ';', 'if', '(', 'lengthIfKnown', '.', 'isPresent', '(', ')', ')', '{', 'return', 'lengthIfKnown', '.', 'get', '(', ')', '==', '0L', ';', '}', 'Closer', 'closer', '=', 'Closer', '.', 'create', '(', ')', ';', 'try', '{', 'Reader', 'reader', '=', 'closer', '.', 'register', '(', 'openStream', '(', ')', ')', ';', 'return', 'reader', '.', 'read', '(', ')', '==', '-', '1', ';', '}', 'catch', '(', 'Throwable', 'e', ')', '{', 'throw', 'closer', '.', 'rethrow', '(', 'e', ')', ';', '}', 'finally', '{', 'closer', '.', 'close', '(', ')', ';', '}', '}'], 'docstring': "Returns whether the source has zero chars. The default implementation first checks {@link\n#lengthIfKnown}, returning true if it's known to be zero and false if it's known to be\nnon-zero. If the length is not known, it falls back to opening a stream and checking for EOF.\n\n<p>Note that, in cases where {@code lengthIfKnown} returns zero, it is <i>possible</i> that\nchars are actually available for reading. This means that a source may return {@code true} from\n{@code isEmpty()} despite having readable content.\n\n@throws IOException if an I/O error occurs\n@since 15.0", 'docstring_tokens': ['Returns', 'whether', 'the', 'source', 'has', 'zero', 'chars', '.', 'The', 'default', 'implementation', 'first', 'checks', '{', '@link', '#lengthIfKnown', '}', 'returning', 'true', 'if', 'it', 's', 'known', 'to', 'be', 'zero', 'and', 'false', 'if', 'it', 's', 'known', 'to', 'be', 'non', '-', 'zero', '.', 'If', 'the', 'length', 'is', 'not', 'known', 'it', 'falls', 'back', 'to', 'opening', 'a', 'stream', 'and', 'checking', 'for', 'EOF', '.'], 'sha': '7155d12b70a2406fa84d94d4b8b3bc108e89abfd', 'url': 'https://github.com/google/guava/blob/7155d12b70a2406fa84d94d4b8b3bc108e89abfd/android/guava/src/com/google/common/io/CharSource.java#L334-L348', 'partition': 'valid'}
                '''
                #print(line)
                jsons[type].append(line)
            f.close()

    #  存储成字典形式
    return jsons['train'], jsons['valid'], jsons['test']


def split_camel(camel_str):
    # [VARIABLE "gs://my_bucket/PartitionB_*.csv"]
    try:
        split_str = re.sub(r'(?<=[a-z]|[0-9])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\s+', '_',camel_str)
        #print(split_str) [VARIABLE_"gs://my_bucket/Partition_B_*.csv"]
    except TypeError:
        return ['']
    try:
        if split_str[0] == '_':
            return [camel_str]
    except IndexError:
        return []
    # ['[variable', '"gs://my', 'bucket/partition', 'b', '*.csv"]']
    return split_str.lower().split('_')  # 骆驼命名法处理并分词


def npvecs_similar(np_vec1, np_vec2, sim_measure='cos'):
    # sigmoid 激活函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 矩阵再列向量归一化
    def normalize(x):
        # 按行进行二范数处理
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    if sim_measure == 'cos':
        # 一维矩阵是内积，多维矩阵是矩阵相乘
        # 2*3 和 3*1 相乘， 2*1 [[2],[3]]
        # [:0] = [2,3]
        return np.dot(normalize(np_vec1), normalize(np_vec2).T)[:,0]

    elif sim_measure == 'poly':
        return (0.5 * np.dot(np_vec1, np_vec2.T).diagonal() + 1) ** 2

    elif sim_measure == 'sigmoid':
        return np.tanh(np.dot(np_vec1, np_vec2.T).diagonal() + 1)

    elif sim_measure in ['euc', 'gesd', 'aesd']:
        euc_dist = np.linalg.norm(np_vec1 - np_vec2, axis=1)
        euc_sim = 1 / (1 + euc_dist)
        sigmoid_sim = sigmoid(np.dot(np_vec1, np_vec2.T).diagonal() + 1)
        if sim_measure == 'euc':
            return euc_sim
        if sim_measure == 'gesd':
            return euc_sim * sigmoid_sim
        elif sim_measure == 'aesd':
            return 0.5 * (euc_sim + sigmoid_sim)