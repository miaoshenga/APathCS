# -*- coding:UTF-8 -*-
import os
import time
import logging

# 数据处理
from data_helper import *
# 额外处理
from model_adds import *

# 模型选择
from model_APathCS import SiameseCSNN


# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)


class NNConfig(object):
    def __init__(self, embeddings=None):

        #  序列长度
        self.query_length = 12
        self.code_length  = 480
        #  迭代次数
        self.num_epochs = 200
        #  批次大小
        self.batch_size = 128
        #  评测的批次
        self.eval_batch = 128

        #  隐层大小
        self.hidden_size = 256
        #  丢失率
        self.keep_prob = 0.25

        #  词嵌入矩阵
        self.embeddings = np.array(embeddings).astype(np.float32)
        self.embedding_size = 300

        #  优化器的选择
        self.optimizer = 'adam'
        #  正则化
        self.l2_lambda = 0.0001

        #  学习率
        self.learning_rate = 0.1
        #  间距值
        self.margin = 0.05

        self.save_path='./'

        self.best_path='./'


def evaluate(sess, model, corpus, config):
    """
    using corpus to evaluate the session and model’s MAP MRR
    """
    iterator = Iterator(corpus)

    count = 0
    total_qids =  []
    total_cids =  []
    total_preds = []
    total_labels =[]
    total_loss = 0


    for batch_x in iterator.next(config.eval_batch, shuffle=True):
        # 查询id, 样例, 查询长度, 样例长度
        batch_qids, batch_q, batch_cids, batch_c, batch_qmask, batch_cmask, labels = zip(*batch_x)

        batch_q = np.asarray(batch_q)
        batch_c = np.asarray(batch_c)

        batch_U = np.random.random(size=(batch_q.shape[0],50))
        batch_U = np.asarray(batch_U)

        #距离
        q_cp_cosine, loss = sess.run([model.q_cpos_cosine, model.total_loss],
                                     feed_dict ={model.code_pos:batch_c,
                                                model.query:batch_q,
                                                model.code_neg:batch_c,
                                                model.midd_u:batch_U})

                                                # model.dropout_keep_prob: 0.25})

        total_loss += loss

        count += 1

        total_qids.append(batch_qids)
        total_cids.append(batch_cids)
        total_preds.append(q_cp_cosine)
        total_labels.append(labels)


    total_qids   = np.concatenate(total_qids, axis=0)
    total_cids   = np.concatenate(total_cids, axis=0)
    total_preds  = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels,axis=0)

    # 评价指标
    recall_1, recall_10, mrr, frank, prec_1,  acc, ndcg= eval_metric(total_qids, total_cids, total_preds, total_labels)
    # 平均损失
    ave_loss = total_loss/count


    return ave_loss, recall_1, recall_10, mrr, frank, prec_1, acc, ndcg



def train(train_corpus, valid_corpus, test_corpus, config):

    iterator = Iterator(train_corpus)

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    if not os.path.exists(config.best_path):
        os.makedirs(config.best_path)

    with tf.Session() as sess:
        # 训练的主程序模块
        print('#######################开始训练和评价#######################')
        logger.info("开始模型训练！！！!")

        # 训练开始的时间
        start_time = time.time()

        model = SiameseCSNN(config)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        ckpt = tf.train.get_checkpoint_state(config.save_path)

        print('#######################配置TensorBoard#######################')

        summary_writer = tf.summary.FileWriter(config.save_path, graph=sess.graph)

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('..........重新加载模型的参数..........')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('..........创建新建模型的参数..........')
            sess.run(tf.global_variables_initializer())

        # 计算模型的参数个数
        total_parameters = count_parameters()
        print('统计全部的参数个数:%d'%total_parameters)


        current_step = 0




        for epoch in range(config.num_epochs):
            # 开始迭代
            print("- Epoch {}/{} -".format(epoch+1, config.num_epochs))

            count = 0

            for batch_x in iterator.next(config.batch_size, shuffle=True):
                # 正列, 查询,  负例, 正例长度, 查询长度，负例长度
                batch_c_pos, batch_q, batch_c_neg, batch_c_pos_mask,batch_qmask,batch_c_neg_mask = zip(*batch_x)

                batch_c_pos = np.asarray(batch_c_pos)
                batch_q  = np.asarray(batch_q)
                batch_c_neg = np.asarray(batch_c_neg)
                batch_U = np.random.random(size=(batch_q.shape[0],50))
                batch_U = np.asarray(batch_U)

                _, loss, summary= sess.run([model.train_op, model.total_loss, model.summary_op],
                                                      feed_dict={model.code_pos:batch_c_pos,
                                                                 model.query:batch_q,
                                                                 model.code_neg:batch_c_neg,# 填补参数
                                                                 model.midd_u: batch_U})

                count += 1
                current_step += 1

                if count % 500 == 0:
                    print('[epoch {}, batch {}], loss:{}'.format(epoch, count, loss))

                summary_writer.add_summary(summary, current_step)

            if valid_corpus is not None:


                valid_loss, valid_recall_1, valid_recall_10, valid_mrr,  valid_frank,  valid_prec_1, valid_acc, valid_ndcg= evaluate(sess, model, valid_corpus, config)
                print("epoch: %d valid Loss: %.4f \nvalid recall@1: %.4f valid recall@10: %.4f valid mrr: %.4f  valid ndcg: %.4f"% (epoch, valid_loss, valid_recall_1,valid_recall_10,valid_mrr, valid_ndcg))

                logger.info("Epoch {} Mean Loss is: {:.5f}".format(epoch, np.mean(valid_loss)))
                logger.info("Model Eval......")

                logger.info("Epoch {} Eval Mrr is {:.5f}".format(epoch, valid_mrr))
                logger.info("Epoch {} Eval SR@1 is {:.5f}".format(epoch, valid_recall_1))
                logger.info("Epoch {} Eval SR@10 is {:.5f}".format(epoch, valid_recall_10))
                logger.info("Epoch {} Eval NDCG is {:.5f}".format(epoch, valid_ndcg))

                #实时模型的文件
                checkpoint_path =os.path.join(config.save_path, 'mrr_{:.4f}_{}.ckpt'.format(valid_mrr, current_step))

                # 保存的地址
                saver.save(sess, checkpoint_path, global_step=epoch)


        # 训练结束的时间
        end_time=time.time()

        print('训练稳定后程序运行的时间：%s 秒'%(end_time-start_time))



def main():

    # 按照PCI_BUS_ID顺序从0开始排列GPU设备
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # 使用第1, 2块GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 不同数据类型  origin  cleaned  nlqfpro
    data_type  = 'nlqfpro'
    print('data_type',data_type)

    # 创建一个handler，用于写入日志文件
    timestamp = str(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
    modelstamp = 'model-%s' %data_type + timestamp

    # 加载记录文件
    fh = logging.FileHandler('./log/' + modelstamp + '.txt')
    fh.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)

    # 序列长度
    max_q_length = 30   #查询
    max_c_length = 300  #代码

    # 读取训练路径
    pair_path = './corpus/pair'
    # 读取其他路径
    data_path =  './corpus/data'

    # 手工设置路径
    save_path = "./ckpt/%s/checkpoint"%data_type
    best_path = "./ckpt/%s/bestval"%data_type

    vocab_path = './vocab'

    # 词典文件
    vocab = os.path.join(vocab_path, '%s_clean_vocab.txt' % data_type)

    # 结构化词向量的文件路径
    embed_path = './struc2vec'

    # 词向量文件
    struc2vec_file = os.path.join(embed_path,'%s_struc2vec.pkl'%data_type)
    # 词向量编码 (?,300)
    struc2vecs = load_embedding(struc2vec_file)

    print('struc2vecs词向量维度',np.shape(struc2vecs))

    # 网络参数
    config = NNConfig(embeddings=struc2vecs)

    config.query_length = max_q_length
    config.code_length = max_c_length

    config.save_path = save_path
    config.best_path = best_path

    # 读取训练数据
    train_file = os.path.join(pair_path,'%s_train_triplets.txt'%data_type)
    # 读取验证数据
    valid_file = os.path.join(data_path,'%s_parse_valid.txt'%data_type)
    # 读取测试数据
    test_file  = os.path.join(data_path,'%s_parse_test.txt'%data_type)

    # [q_id, query, c_id, code, q_mask, c_mask, label]
    train_transform = transform_train(train_file,vocab)
    # 转换ID
    valid_transform = transform(valid_file,vocab)
    test_transform  = transform(test_file,vocab)

    # padding处理
    train_corpus = load_train_data(train_transform, max_q_length, max_c_length)

    # padding处理
    valid_corpus = load_data(valid_transform, max_q_length, max_c_length, keep_ids=True)
    test_corpus  = load_data(test_transform, max_q_length, max_c_length, keep_ids=True)


    train(train_corpus, valid_corpus, test_corpus, config)


if __name__ == '__main__':
    main()
