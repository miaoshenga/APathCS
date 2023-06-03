import tensorflow as tf
# -*- coding:UTF-8 -*-
import numpy as np


# todo 参数配置




class SiameseCSNN(object):
    def __init__(self, config):
        #  序列长度
        self.query_len = config.query_length
        self.code_len = config.code_length

        #  词向量矩阵
        self.embeddings = config.embeddings
        self.embedding_size = config.embedding_size

        #  优化器
        self.optimizer = config.optimizer
        #  正则化
        self.l2_lambda = config.l2_lambda

        #  学习率
        self.learning_rate = config.learning_rate
        #  间隔阈值
        self.margin = config.margin

        # self-attention
        self.num_blocks = 6
        #  注意力头数
        self.num_heads = 6

        self.placeholder_init()

        # 正负样本距离
        self.q_cpos_cosine, self.q_cneg_cosine = self.build(self.embeddings)
        # 损失和精确度
        self.total_loss = self.add_loss_op(self.q_cpos_cosine, self.q_cneg_cosine, self.l2_lambda)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)


    def placeholder_init(self):
        self.code_pos = tf.placeholder(tf.int32, [None, self.code_len], name='code_point')

        self.query = tf.placeholder(tf.int32, [None, self.query_len], name='query_point')

        self.code_neg = tf.placeholder(tf.int32, [None, self.code_len], name='code_point')

        self.midd_u = tf.placeholder(tf.int32, [None, 50], name='midd_u')

        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.batch_size, self.list_size = tf.shape(self.code_pos)[0], tf.shape(self.code_neg)[1]



    def scal_attention(self, scope, Q, K, V,U,dropout_rate):
        with tf.variable_scope("scalatt" + scope):
            # 点积
            outputs = tf.matmul(Q,U)
            # print('outputs',outputs.shape)
            # print('K',K.shape)
            # print('K.tr',tf.transpose(K, [0, 2, 1]).shape)

            outputs = tf.matmul(outputs, tf.transpose(K, [0, 2, 1]))
            # key masking
            # 对填充的部分进行一个mask，这些位置的attention score变为极小，embedding是padding操作的，
            # 填充的部分其embedding都是0，加起来也是0，我们就会填充一个很小的数。
            key_masks = tf.sign(tf.abs(tf.reduce_sum(K, axis=-1)))
            # print('key_masks',key_masks.shape)
            key_masks = tf.expand_dims(key_masks, 1)
            # print('key_masks', key_masks.shape)
            key_masks = tf.tile(key_masks, [1, tf.shape(Q)[1], 1])

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

            # softmax操作
            outputs = tf.nn.softmax(outputs)

            # query masking
            query_masks = tf.sign(tf.reduce_sum(tf.abs(Q), axis=-1))
            query_masks = tf.expand_dims(query_masks, -1)  #
            query_masks = tf.tile(query_masks, [1, 1, tf.shape(K)[1]])

            outputs *= query_masks
            # dropout操作
            outputs = tf.nn.dropout(outputs, dropout_rate)
            outputs = tf.matmul(outputs, V)

        return outputs

    def normalize(self, inputs, epsilon=1e-8):

        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
        outputs = gamma * normalized + beta

        return outputs

    def multi_attention(self, scope, queries, keys, values,u, num_units=None, num_heads=6, dropout_rate=0.25):

        num_units = queries.get_shape().as_list()[-1] if num_units is None else num_units
        # print('num_units', num_units)
        with tf.variable_scope("multiatt" + scope):
            #  线性变换
            #  维度(b,length,num_units)
            Q = tf.layers.dense(queries, num_units, use_bias=True)
            #  维度(b,length,num_units)
            K = tf.layers.dense(keys, num_units, use_bias=True)
            #  维度(b,length,num_units)
            V = tf.layers.dense(values, num_units, use_bias=True)
            U = tf.layers.dense(u, num_units, use_bias=True)
            # print('U', U.shape)

            # 分解和拼接
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (n*b, length, embedding_size/n)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (n*b, length, embedding_size/n)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (n*b, length, embedding_size/n)
            U_ = tf.concat(tf.split(U, num_heads, axis=2), axis=0)  # (n*b, length, embedding_size/n)

            # Attention
            outputs = self.scal_attention(scope, Q_, K_, V_,U_, dropout_rate)
            # 恢复拼接
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
            # 残差链接
            # outputs += queries
            # # 正则化
            # outputs = self.normalize(outputs)

        return outputs



    def joint_enc(self,queries, keys, values, u,num_units=None, num_heads=6, dropout_rate=0.25):

        num_units = queries.get_shape().as_list()[-1] if num_units is None else num_units
        # queries (b, length,embedding_size)
        #  线性变换
        #  维度(b,length,num_units)
        Q = tf.layers.dense(queries, num_units, use_bias=True)
        #  维度(b,length,num_units)
        K = tf.layers.dense(keys, num_units, use_bias=True)
        #  维度(b,length,num_units)
        V = tf.layers.dense(values, num_units, use_bias=True)
        U = tf.layers.dense(u, num_units, use_bias=True)

        # 分解和拼接
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (n*b, length, embedding_size/n)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (n*b, length, embedding_size/n)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (n*b, length, embedding_size/n)
        U_ = tf.concat(tf.split(U, num_heads, axis=2), axis=0)  # (n*b, length, embedding_size/n)

        # Attention
        outputs = tf.matmul(Q_, U_)
        outputs = tf.matmul(outputs, tf.transpose(K_, [0, 2, 1]))

        # key masking
        # 对填充的部分进行一个mask，这些位置的attention score变为极小，embedding是padding操作的，
        # 填充的部分其embedding都是0，加起来也是0，我们就会填充一个很小的数。
        key_masks = tf.sign(tf.abs(tf.reduce_sum(K_, axis=-1)))
        key_masks = tf.expand_dims(key_masks, 1)
        key_masks = tf.tile(key_masks, [1, tf.shape(Q_)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # softmax操作
        outputs = tf.nn.softmax(outputs)

        # query masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(Q_), axis=-1))
        query_masks = tf.expand_dims(query_masks, -1)  #
        query_masks = tf.tile(query_masks, [1, 1, tf.shape(K_)[1]])
        outputs *= query_masks
        # # dropout操作
        outputs = tf.nn.dropout(outputs, dropout_rate)


        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V_)

        # 恢复拼接
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # 残差链接
        # outputs += queries
        # # 正则化
        # outputs = self.normalize(outputs)
        return outputs

    def jointAttention(self, ff_q, ff_c_pos,ff_c_neg,U):
        #tf.reset_default_graph()
        code_repr = self.joint_enc(ff_q, ff_c_pos,ff_c_pos,U)
        desc_pos_repr = self.joint_enc(ff_c_pos, ff_q,ff_q,U)
        desc_neg_repr = self.joint_enc(ff_c_neg, ff_q,ff_q,U)

        code_repr = tf.squeeze(tf.reduce_mean(code_repr, axis=1, keep_dims=True), axis=1)
        desc_pos_repr = tf.squeeze(tf.reduce_mean(desc_pos_repr, axis=1, keep_dims=True), axis=1)
        desc_neg_repr = tf.squeeze(tf.reduce_mean(desc_neg_repr, axis=1, keep_dims=True), axis=1)

        return code_repr, desc_pos_repr, desc_neg_repr

    def feed_forward(self, scope, inputs, num_units):

        with tf.variable_scope("forward" + scope):
            # 全连接full connection输出
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, num_units[1])
            # 残差链接
            outputs += inputs
            # 正则化
            outputs = self.normalize(outputs)

        return outputs

    def build(self, embeddings):

        self.Embedding = tf.Variable(tf.to_float(embeddings), trainable=False, name='Embedding')

        # 维度(?, 300, 300)
        c_pos_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.code_pos), 0.25)
        # 维度(?, 30, 300)
        q_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.query), 0.25)
        # 维度(?, 300, 300)
        c_neg_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.code_neg), 0.25)
        # 维度(?, 50, 300)
        c_midd_u = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.midd_u), 0.25)

        # Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # self-attention 自注意力机制
                with tf.variable_scope('multihead_attention') as scope0:
                    #(batch,300,300)
                    att_c_pos = self.multi_attention("code", queries=c_pos_embed,
                                                     keys=c_pos_embed,
                                                     values=c_pos_embed,
                                                     u = c_midd_u,
                                                     num_heads=self.num_heads,
                                                     dropout_rate=0.25

                                                     )

                    #(batch,30,300)
                    att_q = self.multi_attention("query", queries=q_embed,
                                                 keys=q_embed,
                                                 values=q_embed,
                                                 u=c_midd_u,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=0.25
                                                 )

                    scope0.reuse_variables()
                    # (batch,300,300)
                    att_c_neg = self.multi_attention("code", queries=c_neg_embed,
                                                     keys=c_neg_embed,
                                                     values=c_neg_embed,
                                                     u=c_midd_u,
                                                     num_heads=self.num_heads,

                                                     dropout_rate=0.25
                                                     )

                #  前项传播
                with tf.variable_scope('feed_forward') as scope1:
                    # 维度 (?,300,300)
                    ff_c_pos = self.feed_forward("code", att_c_pos,  num_units=[7 * self.embedding_size, self.embedding_size])
                    # 维度 (?,30,300)
                    ff_q = self.feed_forward("query", att_q, num_units=[7 * self.embedding_size, self.embedding_size])
                    # 维度 (?,300,300)
                    scope1.reuse_variables()
                    ff_c_neg = self.feed_forward("code", att_c_neg,num_units=[7 * self.embedding_size, self.embedding_size])

        with tf.variable_scope('attentive_pooling') :

             #交互注意力
             #（batch,300）,(batch,300),(batch,300)
            code_repr, desc_pos_repr, desc_neg_repr = self.jointAttention(ff_q, ff_c_pos,ff_c_neg,c_midd_u)


        # todo 2：COS式
        q_pos_cosine = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(code_repr, dim=1), tf.nn.l2_normalize(desc_pos_repr, dim=1)), axis=1)

        q_neg_cosine = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(code_repr, dim=1), tf.nn.l2_normalize(desc_neg_repr, dim=1)), axis=1)

        return q_pos_cosine, q_neg_cosine

    def margin_loss(self, pos_sim, neg_sim):
        original_loss = self.margin - pos_sim + neg_sim
        l = tf.maximum(tf.zeros_like(original_loss), original_loss)
        loss = tf.reduce_sum(l)
        return loss, l

    def add_loss_op(self, p_sim, n_sim,l2_lambda=0.0001):
        """
        损失节点
        """
        loss, l = self.margin_loss(p_sim, n_sim)

        tv = tf.trainable_variables()
        l2_loss = l2_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

        pairwise_loss = loss + l2_loss
        tf.summary.scalar('pairwise_loss', pairwise_loss)
        self.summary_op = tf.summary.merge_all()
        return pairwise_loss

    def add_train_op(self, loss):
        """
        训练节点
        """
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            opt = tf.train.AdamOptimizer(self.learning_rate)
            train_op = opt.minimize(loss, self.global_step)

            return train_op
