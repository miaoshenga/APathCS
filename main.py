
import os
import torch
import pickle
import tqdm
import time
from apex import amp
import math
import argparse
import configs
import numpy as np
from models.model_APathCS import Embedder
# from models.model_CARLCS-CNN import Embedder
# from models.model_TCS import Embedder
# from models.model_SANCS import Embedder
# from models.model_PSCS import Embedder
from models import *
from preprocess import  SiamessDataset
from torch.utils.data import DataLoader


# 随机种子
import random
random.seed(42)

# 记录日志
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


# 可用显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1,2,3'



class SearchEngine:
    def __init__(self, model, args, configs=None):
        # 命令参数
        self.args = args
        # 训练参数
        self.configs = configs

        # 数据路径
        self.dataset= os.path.join(self.args.data_path,self.args.data_name)
        # 加载数据
        self.data_params = self.configs.get('data_params', dict())
        # 训练的参数
        self.train_params = self.configs.get('train_params', dict())
        # 模型参数
        self.model_params = self.configs.get('model_params',dict())
        # 运行GPU
        self.device= torch.device(f"cuda:{self.args.with_cuda}" if torch.cuda.is_available() else "cpu")
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.train_params['learning_rate'],
                                          betas=(self.train_params['adam_beta1'], self.train_params['adam_beta2']),
                                          weight_decay=self.train_params['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lr_lambda=lambda epoch: self.train_params['decay_ratio'] **
                                                                                   self.train_params['epoch'])



    def valid(self, args,valid_iter, model, pool_size, K, sim_measure):
        """
        simple validation in a code pool.
        @param: poolsize - size of the code pool, if -1, load the whole test set
        """

        def Recall(real, predict, results):
            sum = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index <= results:
                    sum = sum + 1
            return sum / float(len(real)),index

        def REC1(real, predict, results):
            rat, index = Recall(real, predict, results)
            r = 0
            if index == 0:
                r = rat
            return r

        def ACC(real, predict):
            sum = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1:
                    sum = sum + 1
            return sum / float(len(real))

        def MAP(real, predict):
            sum = 0.0
            for id, val in enumerate(real):
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1: sum = sum + (id + 1) / float(index + 1)
            return sum / float(len(real))

        def SR(real, predict):
            sum = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1:
                    sum += 1.0
            return sum / float(len(real)), index

        def SR1(real, predict):
            rat, index = SR(real, predict)
            r = 0
            if index == 0:
                r = rat
            return r

        def SR10(real, predict):
            rat, index = SR(real, predict)
            r = 0
            if index >= 0 and index <= 9:
                r = rat
            return r

        def MRR(real, predict):
            sum = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1: sum = sum + 1.0 / float(index + 1)
            return sum / float(len(real))

        def NDCG(real, predict):
            dcg = 0.0
            idcg = IDCG(len(real))
            for i, predictItem in enumerate(predict):
                if predictItem in real:
                    itemRelevance = 1
                    rank = i + 1
                    dcg += (math.pow(2, itemRelevance) - 1.0) * (
                            math.log(2) / math.log(rank + 1))
            return dcg / float(idcg)

        def IDCG(n):
            idcg = 0
            itemRelevance = 1
            for i in range(n): idcg += (math.pow(2, itemRelevance) - 1.0) * (
                    math.log(2) / math.log(i + 2))
            return idcg

        model.eval()

        device = next(model.parameters()).device


        sr1,sr10,rec1,accs, mrrs, maps, ndcgs = [], [], [], [],[],[],[]
        code_reprs, desc_reprs = [], []

        for data in tqdm.tqdm(valid_iter):

            with torch.no_grad():
                for key, value in data.items():
                    data[key] = value.to(device)
                #print('nl',data['nl'].shape)
                nl_emb, code_vec = model.forward(       data['nl'],
                                                        data["paths"],
                                                        data["start_tokens"],
                                                        data["end_tokens"])
                # tensor变numpy
                code_joint = code_vec.data.cpu().numpy().astype(np.float32)
                desc_joint = nl_emb.data.cpu().numpy().astype(np.float32)

                # 添加到列表组
                code_reprs.append(code_joint)
                desc_reprs.append(desc_joint)

        code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)

        for i in tqdm.tqdm(range(0, len(valid_iter), pool_size)):
            code_pool, desc_pool = code_reprs[i:i + pool_size], desc_reprs[i:i + pool_size]
            for j in range(desc_pool.shape[0]):
                # 向量 [1 x dim]
                desc_vec = np.expand_dims(desc_pool[j], axis=0)

                sims = vecsimilar(code_pool, desc_vec, sim_measure)
                n_results = K
                negsims = np.negative(sims)

                predict_origin = np.argsort(negsims)
                predict_origin = [int(k) for k in predict_origin]

                predict = np.argsort(negsims)
                predict = predict[:n_results]
                predict = [int(k) for k in predict]
                real = [j]
                sr1.append(SR1(real, predict))
                sr10.append(SR10(real, predict))
                rec1.append(REC1(real, predict_origin, n_results))
                accs.append(ACC(real, predict))
                mrrs.append(MRR(real, predict))
                maps.append(MAP(real, predict))
                ndcgs.append(NDCG(real, predict))

        return np.mean(sr1),np.mean(sr10),np.mean(rec1),np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)

    def train(self, model,args,nl_vocab,path_vocab):

        if torch.cuda.device_count() > 1:
            #  GPU并行处理
            #print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)

        model.to(self.device)

        # 训练集
        corpus_train = SiamessDataset(args,
              '/{}/path_data/github/java/processed/descr_train.txt'.format(args.data_dir),
              nl_vocab, args.nl_seq_len, path_vocab, args.path_len,
              '/{}/path_data/github/java/code_path/train'.format(args.data_dir),args.k, is_train=True)

        train_iter = DataLoader(dataset=corpus_train, batch_size=self.train_params['batch_size'],drop_last=True,)

        print('train_iter_len', len(train_iter))

        # 验证集
        valid_set = SiamessDataset(args,
              '/{}/path_data/github/processed/descr_test.txt'.format(args.data_dir),
              nl_vocab, args.nl_seq_len, path_vocab,args.path_len,
             '/{}/path_data/github/code_path/test'.format(args.data_dir),args.k, is_train=False)

        valid_iter = DataLoader(dataset=valid_set, batch_size=self.train_params['batch_size'],drop_last=True,)

        print('valid_iter_len', len(valid_iter))
        lossf = RankLoss(args)
        logger.info("开始模型训练！！！!")

        # 训练开始的时间
        start_time = time.time()

        # 存放模型的路径
        def save_model(model, ckpt_path, ckpt_name):
            if not os.path.exists(ckpt_path):
                os.mkdir(ckpt_path)
            ckpt_file = os.path.join(ckpt_path, ckpt_name)
            torch.save(model.state_dict(), ckpt_file)

        best_mrr_val = 0.0
        best_rec_val = 0.0
        torch.backends.cudnn.enabled = False
        for epoch in range(self.train_params['epoch']):
            # 打印第几轮训练
            logger.info('[ Epoch ' + str(epoch) + ' ]')
            # 这轮的起始时间
            start_time = time.time()
            data_iter = tqdm.tqdm(enumerate(train_iter),
                                  total=len(train_iter),
                                  bar_format="{l_bar}{r_bar}")

            train_loss = []
            for index, batch in enumerate(train_iter):

                for i, (data, trg_data) in data_iter:

                    for key, value in data.items():
                        data[key] = value.to(self.device)

                    for key, value in trg_data.items():
                        trg_data[key] = value.to(self.device)


                    # if use_parallel:
                    nl_vec, code_vec = model.module.forward(
                            data['nl'],
                            data["paths"],
                            data["start_tokens"],
                            data["end_tokens"])

                    trg_nl_vec, trg_code_vec = model.module.forward(
                            trg_data['nl'],
                            data["paths"],
                            data["start_tokens"],
                            data["end_tokens"])

                    loss = lossf(nl_vec, trg_nl_vec, code_vec, trg_code_vec)

                    #  前一步损失清零
                    self.optimizer.zero_grad()

                    if self.train_params['fp16']:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            #  反向传播
                            scaled_loss.mean().backward()
                            if self.train_params['grad_norm'] is not None:
                                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer),
                                                               self.train_params['grad_norm'])
                    else:
                        # 反向传播
                        loss.mean().backward()  # 多卡loss进行平均
                        if self.train_params['grad_norm'] is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_params['grad_norm'])

                    # 优化器
                    self.optimizer.step()
                    # 拿出损失
                    train_loss.append(loss.mean().item())
                    if (index + 1) % self.args.log_every == 0:
                        logger.info('log for  {}'.format(index + 1))
                        elapsed = time.time() - start_time
                        info = 'Itr:{}, Step_time:{:.8f}, Loss={:.5f}'.format((index + 1), elapsed / (index + 1),
                                                                              np.mean(train_loss))
                        logger.info(info)

                    if (index + 1) % self.args.eval_every == 0:
                        logger.info('eval for  {}'.format(index + 1))
                        # 各项评价指标
                        every_sr1,every_sr10,every_rec,every_acc, every_mrr, every_map, every_ndcg = self.valid(args,valid_iter, model.module, args.pool_size,
                                                                                            args.topK, self.model_params['sim_measure'])
                        result = 'SR1={:.5f}, SR10={:.5f}, Recall={:.5f}, Accurate={:.5f}, Mrr={:.5f}, Map={:.5f}, NDCG={:.5f}'.format(every_sr1,
                                                                                                              every_sr10,
                                                                                                              every_rec,
                                                                                                              every_acc,
                                                                                                              every_mrr,
                                                                                                              every_map,
                                                                                                              every_ndcg)


                        logger.info(result)

            logger.info("Epoch {} Mean Loss is: {:.5f}".format(epoch, np.mean(train_loss)))

            logger.info("Model Eval......")

            epoch_sr1,epoch_sr10,epoch_rec1,epoch_acc, epoch_mrr, epoch_map, epoch_ndcg = self.valid(args,valid_iter, model.module, args.pool_size, args.topK,
                                                                                self.model_params['sim_measure'])

            results = 'SR@1={:.5f}, SR@10={:.5f},Recall={:.5f}, Accurate={:.5f}, Mrr={:.5f}, Map={:.5f}, NDCG={:.5f},epoch={:.5f}'.format(epoch_sr1,
                                                                                                  epoch_sr10,
                                                                                                  epoch_rec1,
                                                                                                  epoch_acc,
                                                                                                  epoch_mrr,
                                                                                                  epoch_map,
                                                                                                  epoch_ndcg,
                                                                                                   epoch
                                                                                                   )

            with open('/.../APathCS/output/APathCS_epoch_results.txt', 'a') as f:
                f.write(results + '\n')

            logger.info("Epoch {} Mrr is {:.5f}".format(epoch, epoch_mrr))
            logger.info("Epoch {} SR@1 is {:.5f}".format(epoch, epoch_rec1))
            logger.info("Epoch {} SR@10 is {:.5f}".format(epoch, epoch_sr10))
            logger.info("Epoch {} NDCG is {:.5f}".format(epoch, epoch_ndcg))

            self.scheduler.step()

            if epoch_mrr > best_mrr_val or epoch_rec1 > best_rec_val:
                # 数字慢慢变大
                best_mrr_val = epoch_mrr
                best_rec_val = epoch_rec1

                # 时刻保存模型
                self.ckpt_name = 'model_best.pt'.format(best_mrr_val)
                self.ckpt_path = os.path.join(self.args.data_path, self.args.model_path)

                save_model(model, self.ckpt_path, self.ckpt_name)

        # 训练结束的时间
        end_time = time.time()

        print('训练稳定后程序运行的时间：%s 秒' % (end_time - start_time))

#
def vecsimilar(np_vec1, np_vec2, sim_measure='cos'):
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


def parse_args():
    ####################################模型训练的配置#######################################################
    parser = argparse.ArgumentParser("Train and Test Code Search Model")
    # 数据读取的路径
    parser.add_argument("--data_path", type=str, default='/.../APathCS/', help="data location")
    # 数据读取的类型
    parser.add_argument("--data_name", type=str, default="alldata/", help="dataset name")
    # 模型保存路径
    parser.add_argument("--model_path", type=str, default="model/", help="model path")

    # 网络的模型名称
    parser.add_argument("--model_id", type=str,default="APathCS", help="model name")
    # 网络运行的模式
    parser.add_argument("--model_type", type=str,  default='train', help= "The `train` mode, trains the model in train set")
    # 打印日志的轮数
    parser.add_argument('--log_every', type=int, default=5000, help='interval to log results')
    # 模型评价的轮数
    parser.add_argument('--eval_every', type=int, default=10000, help='interval to valid')
    # 使用gpu编号
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    # 其他的参数
    parser.add_argument("--verbose", action="store_true", default=True, help="be verbose")
    # 数据读取的路径
    parser.add_argument("--data_dir", type=str, default="/.../APathCS", help="data location")

    parser.add_argument("--k",type=int, default=40, help="k paths at a time")
    parser.add_argument("-rdp", "--rnn_dropout", type=float, default=0.25)
    parser.add_argument("--pool_size", type=int, default=100)
    parser.add_argument("--topK", type=int, default=10)
    parser.add_argument("-pl", "--path_len", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("-nls", "--nl_seq_len", type=int, default=12,
                        help="maximum sequence len of natural language")
    parser.add_argument("-es", "--emb_size", type=int, default=64)
    parser.add_argument("--max_norm", type=float, default=1.0,
                        help="learning rate of adam")
    parser.add_argument("-m", "--margin", type=float, default=0.3,help="margin")
    parser.add_argument("-dp", "--dropout", type=float, default=0.25)
    parser.add_argument("--proto", choices=["config_APathCS"], default="config_APathCS",
                        help="Prototype config to use for config")
    parser.add_argument('-cuda', "--with_cuda", type=int, default=2, help="training with CUDA: true, or false")

    return parser.parse_args()



def main():

    # 按照PCI_BUS_ID顺序从0开始排列GPU设备
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # 可用显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3 '

    #  模型训练的配置
    args = parse_args()

    # 创建一个handler，用于写入日志文件
    timestamp = str(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
    modelstamp = 'model-%s-' % args.model_id + timestamp

    # 加载记录文件
    fh = logging.FileHandler('/.../APathCS/log/' + modelstamp + '.txt')
    fh.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)

    f = open('/{}/path_data/github/processed/descr_vocab.pickle'.format(args.data_dir), 'rb')
    nl_vocab = pickle.load(f)
    f.close()
    f = open('/{}/path_data/github/processed/path_vocab.pickle'.format(args.data_dir), 'rb')
    path_vocab = pickle.load(f)
    f.close()

    print('cuda:', args.with_cuda)
    device = torch.device(f"cuda:{args.with_cuda}" if torch.cuda.is_available() else "cpu")

    #模型构架的形式
    config = getattr(configs, 'config_APathCS')()
    model = Embedder(config, len(nl_vocab), len(path_vocab), device)
    #运行模型PSCS时用下方法
    # model = Embedder(args, len(nl_vocab), len(path_vocab), device)

    Search = SearchEngine(model, args, configs=config)

    if args.model_type == 'train':
        Search.train(model,args,nl_vocab,path_vocab)

    if args.model_type == 'eval':
        Search.eval(model,args,nl_vocab,path_vocab)


if __name__ == '__main__':
    main()


