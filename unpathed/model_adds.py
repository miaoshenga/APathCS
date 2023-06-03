# -*- coding: utf-8 -*-
import tensorflow as tf

import math

def eval_metric(qids, cids, preds, labels):

    pre_dict = {}
    dic = {}

    # 首先把相同qid的用字典的方式聚集起来~
    for qid, cid, pred, label in zip(qids, cids, preds, labels):
        #字典中包含若给定键，则返回该键对应的值，否则返回设置的值
        pre_dict.setdefault(qid, [])
        pre_dict[qid].append([cid, pred, label])

    #查找键值
    for qid in pre_dict.keys():
        # 按照pred排序 对list排序,已经对分数排名
        dic[qid] = sorted(pre_dict[qid], key=lambda x: x[1], reverse=True)
        # 得到rank
        cid2rank = {cid: [label, rank] for (rank, (cid, pred, label)) in enumerate(dic[qid])}
        #  排名
        dic[qid] = cid2rank

        #{’Q0':{'C00':[0,0],'C01':[0,1],'C02':[1,2],'C03':[0,3]}....,’Q1':}

    recall_1 = 0.0
    recall_2 = 0.0
    recall_10 = 0.0

    mrr = 0.0
    l_mrr = 10.0
    h_mrr = 0.0

    frank = 0.0
    l_frank = 10.0
    h_frank = 0.0

    prec_1 = 0.0
    prec_3 = 0.0
    prec_5 = 0.0

    dcg = 0.0   # 初始化dcg

    idcg = 0.0

    acc = 0.0


    dic_length = 0

    for qid in dic.keys():
        #判断字典的长度
        dic_length += 1
        # 按dic[qid]里面的 rank排序 ,从小到大（label,rank) C02是正确样本
        sort_rank = sorted(dic[qid].items(), key=lambda x:x[1][1], reverse=False)
        #[('C00', [0, 0]), ('C01', [0, 1]), ('C02', [1, 2]), ('C03', [0, 3])]

        # 循环遍历最佳候选位置
        for i in range(len(sort_rank)):

            if sort_rank[i][1][0] == 1:

                if i == 0:
                    # 计算recall@1的值
                    recall_1 += 1.0

                if i <= 1:
                    # 计算recall@2的值
                    recall_2 += 1.0

                if i <= 9:
                    # 计算recall@10的值
                    recall_10 += 1.0

                # 计算 MRR值
                mrr += 1.0 / float(i+1)
                #mrr += 1.0
                # 最小值
                if l_mrr > 1.0 / float(i+1):
                    l_mrr = 1.0 / float(i+1)
                # 最大值
                if h_mrr < 1.0 / float(i+1):
                    h_mrr = 1.0 / float(i+1)

                # 计算 ACC 值
                acc += 1.0

                # 计算 NDCG 值
                predictItem = sort_rank[i][1][0]
                if predictItem in [i]:
                    itemRelevance = 1
                    rank = i + 1
                    dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))

                idcg += (math.pow(2, 1) - 1.0) * (math.log(2) / math.log(i + 2))
                    # idcg += (math.pow(2, 1) - 1.0) * (math.log(2) / math.log(i + 2))



                # 计算FRank
                frank += i
                # 最小值
                # if l_frank > i:
                #       l_frank = i
                # # 最大值
                # if h_frank < i:
                #     h_frank = i

                # 计算k =1,3,5的Precison
                if i == 0:
                    prec_1 += 1/1

                if i <= 2:
                    prec_3 += 1/3

                if i <= 4:
                    prec_5 += 1/5


    # 召回率
    recall_1/= dic_length

    recall_2/= dic_length

    recall_10/= dic_length

    # MRR
    mrr /= dic_length

    # 平均排名
    frank/= dic_length

    # 准确率
    prec_1/=dic_length

    prec_3/=dic_length

    prec_5/=dic_length

    # ACC
    acc /= dic_length

    # NDCG
    ndcg = dcg / float(idcg)

    return recall_1, recall_2, mrr,  frank,prec_1,  acc, ndcg

def count_parameters():

    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_params = 1
        for dim in shape:
            variable_params *= int(dim)
        total_params += variable_params

    return total_params


