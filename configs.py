
def config_APathCS():

    configs = {
    #################################数据的参数#################################
    'data_params':
        {
        #  训练数据
        'train_name':'train.name.h5',
        'train_apiseq':'train.apiseq.h5',
        'train_tokens':'train.tokens.h5',
        'train_desc':'train.desc.h5',

        #  验证数据
        'valid_name':'test.name.h5',
        'valid_apiseq':'test.apiseq.h5',
        'valid_tokens':'test.tokens.h5',
        'valid_desc':'test.desc.h5',

        # 数据参数
        'nlseq_len': 12,
        'name_len': 6,   # name的长度
        'seq_len': 12, # api的长度 #SANCS=30
        'tokens_len': 480, # token的长度
        'desc_len': 30,  # 描述的长度
        'vocab_size': 10000, # 词典的大小

        # 字典数据
        'vocab_name': 'vocab.name.pkl',
        'vocab_api': 'vocab.apiseq.pkl',
        'vocab_tokens': 'vocab.tokens.pkl',
        'vocab_desc': 'vocab.desc.pkl',
        },

    #################################训练的参数#################################
    'train_params':
        {
        'batch_size': 64, #训练批次大小64
        'valid_size': 50, # 验证批次大小50
        'epoch': 200,  # 训练多少次
        'learning_rate': 1e-1, # 学习率
        'adam_epsilon': 1e-8,
        'adam_beta1': 0.9,
        'adam_beta2':0.999,
        'weight_decay': 0,
        'decay_ratio':0.95,
        'grad_norm': 5.0,
        'fp16': False,
        },

    ###################################模型的参数#################################
    'model_params':
        {
        'model_dim':128,
        'attn_dim': 128,
        'num_heads': 8,
        'num_layers': 2,
        'd_ff': 100,
        'max_norm':1.0,
        'ffn_dim': 256,
        'margin': 0.3,
        'sim_measure': 'cos' #相似性度量 gesd, cos, aesd
        }
    }

    return configs
