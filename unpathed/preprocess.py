# -*- coding: utf-8 -*-
import pickle
import numpy as np


#-----------------------------------训练集处理成三元组模式---------------------------------

def gen_train_samples(data_type):
    #处理训练集
    train_ipath='./corpus/data/%s_parse_train.txt'%data_type
    train_opath='./corpus/pair/%s_train_triplets.txt'%data_type
    with open (train_ipath, 'r',encoding='utf-8') as fin:
        with open(train_opath,'w',encoding='utf-8') as fout:
            #正样本
            pos_list = []
            count=0  #计数
            # 循环读取每一行
            while True:
                line=fin.readline()
                # 如果不存在跳出循环
                if not line:
                    break
                #元素拆分
                line_info=line.split('\t')
                c_id=line_info[2]
                id=int(c_id.split('_')[2])
                if id==0:
                    count += 1
                    pos_code=line_info[3]
                    pos_list.append(pos_code)
                else:
                    query=line_info[1]
                    neg_code=line_info[3]
                    #写入（正样本，查询，负样本）
                    fout.write('\t'.join([pos_list[count-1],query,neg_code]) + '\n')
    print('triplets数据转换完毕！')

#-----------------------------------收集所有数据的词汇--------------------------------

def gen_vocab(data_type):

    # 词典包含了查询和代码的词
    words = []
    # 训练、验证、测试
    data_sets = ['%s_parse_train.txt'%data_type,'%s_parse_valid.txt'%data_type,'%s_parse_test.txt'%data_type]

    for set_name in data_sets:
        fin_path ='./corpus/data/%s'%set_name
        with open(fin_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line_in =line.strip('\n').split('\t')
                query=line_in[1].split(' ')
                code =line_in[3].split(' ')
                for r1 in query:
                    if r1 not in words:
                        words.append(r1)
                for r2 in code:
                    if r2 not in words:
                        words.append(r2)

    fout_path = './vocab/%s_vocab.txt'%data_type
    with open(fout_path,'w',encoding='utf-8') as fout:
        for i, j in enumerate(words):
            fout.write('{}\t{}\n'.format(i, j))

    print('vocab词典数据转换完毕！')


#-----------------------------------根据词表生成对应的embedding--------------------------------

def data_transform(embed_folder,data_type,embedding_size):

    vocab_in = './vocab/%s_vocab.txt'%data_type
    # add 2 words: <PAD> and <UNK>
    clean_vocab_out = './vocab/%s_clean_vocab.txt'%data_type

    struc2vec_in  = '../use_embeddings/java_glove.txt'
    struc2vec_out = embed_folder+'%s_struc2vec.pkl'%data_type

    words = []
    with open(vocab_in, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip('\n').split('\t')[1]
            words.append(word)
    print('%s类型的java语言中的vocab.txt总共有%d个词'%(data_type,len(words)))

    ##############################################embedding####################################
    rng = np.random.RandomState(None)

    pad_embedding = rng.uniform(-0.25, 0.25, size=(1, embedding_size))
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, embedding_size))

    struc2vecs = []
    # 加载pad和unk词
    clean_words = ['<pad>', '<unk>']
    struc2vecs.append(pad_embedding.reshape(-1).tolist())
    struc2vecs.append(unk_embedding.reshape(-1).tolist())
    print('pad和unk服从均匀分布的随机变量初始化......')

    # 打开词向量
    with open(struc2vec_in, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line_info = line.strip('\n').split(' ')
                word = line_info[0]
                embedding = [float(val) for val in line_info[1:]]
                if word in words:
                    #词在词典中
                    clean_words.append(word)
                    struc2vecs.append(embedding)
            except:
                print('这行加载失败: %s'%line.strip())

    print('%s类型的java语言中的clean_vocab.txt总共有%d个词'%(data_type, len(clean_words)))

    ##############################################embedding#####################################
    print('%s类型的java语言在glove词表加上pad和unk总共有%d个词'%(data_type,len(clean_words)))

    print('%s类型的java语言的embeddings总共有%d个词'%(data_type,len(struc2vecs)))

    print('{}类型的java语言的embedding的维度为:{}'.format(data_type,np.shape(struc2vecs)))

    # 保存在词库的的词
    with open(clean_vocab_out, 'w', encoding='utf-8') as f:
        for i, j in enumerate(clean_words):
            f.write('{}\t{}\n'.format(i, j))

    # 保存embedding为pickle文件
    with open(struc2vec_out, 'wb') as f:
        pickle.dump(struc2vecs, f)





'''
origin类型的java语言中的vocab.txt总共有861385个词
pad和unk服从均匀分布的随机变量初始化......
origin类型的java语言中的clean_vocab.txt总共有858940个词
origin类型的java语言在glove词表加上pad和unk总共有858940个词
origin类型的java语言的embeddings总共有858940个词
origin类型的java语言的embedding的维度为:(858940, 300)
triplets数据转换完毕！




'''

#--------参数配置----------
struc2vec_folder='./struc2vec/'

#数据类型
origin_type  ='origin'
cleaned_type ='cleaned'
nlqfpro_type ='nlqfpro'

# 向量维度
embedding_size=300


if __name__ == '__main__':

    # origin
    gen_train_samples(origin_type)
    gen_vocab(origin_type)
    data_transform(struc2vec_folder,origin_type, embedding_size)

    # cleaned
    gen_train_samples(cleaned_type)
    gen_vocab(cleaned_type)
    data_transform(struc2vec_folder,cleaned_type, embedding_size)

    # nlqfpro
    gen_train_samples(nlqfpro_type)
    gen_vocab(nlqfpro_type)
    data_transform(struc2vec_folder,nlqfpro_type, embedding_size)






