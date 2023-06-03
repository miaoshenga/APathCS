import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
import numpy as np

from torch import einsum


class MultiHeadAttention(nn.Module):
    def __init__(self,configs,pos_ffn,device,dropout = 0.25):
        super(MultiHeadAttention, self).__init__()

        self.pos_ffn = pos_ffn
        # 模型的参数
        self.model_params = configs.get('model_params', dict())

        self.model_dim = self.model_params['model_dim']
        self.n_heads = self.model_params['num_heads']
        self.dim_per_head =self.model_dim // self.n_heads
        self.device = device


        self.W_Q = nn.Linear(self.model_dim, self.dim_per_head * self.n_heads,bias=False)
        self.W_K = nn.Linear(self.model_dim, self.dim_per_head * self.n_heads,bias=False)
        self.W_V = nn.Linear(self.model_dim, self.dim_per_head * self.n_heads,bias=False)
        self.linear = nn.Linear(self.n_heads * self.dim_per_head, self.model_dim)
        self.layer_norm = nn.LayerNorm(self.model_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self,Q,K,V):
        dim_per_head = self.dim_per_head
        residual,batch_size = Q,Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)

        U = torch.randn(self.model_params['model_dim'], self.model_params['model_dim']).view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        U = U.to(self.device)


        #scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(dim_per_head)
        scores = torch.matmul(q_s, U.transpose(-1, -2))

        scores =  torch.matmul(scores.transpose(-1, -2),k_s)/ np.sqrt(dim_per_head)

        attn = nn.Softmax(dim=-1)(scores)  #softmax函数按横行来做

        context = torch.matmul(attn, v_s.transpose(-1, -2))

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)

        output = self.linear(context)

        context = self.dropout(output)


        output = self.layer_norm(context+residual)
        output = self.pos_ffn(output)

        return output

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,configs):
        self.model_params = configs.get('model_params', dict())
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=self.model_params['model_dim'], out_channels=self.model_params['d_ff'], kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.model_params['d_ff'], out_channels=self.model_params['model_dim'], kernel_size=1)
        self.layer_norm = nn.LayerNorm(self.model_params['model_dim'])

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class Embedder(nn.Module):
    def __init__(self,configs, nl_vocab_len, path_vocab_len, device,dropout=0.25):
        super(Embedder, self).__init__()

        # 训练参数
        self.configs = configs
        # 加载的数据
        self.data_params = self.configs.get('data_params', dict())
        # 训练的参数
        self.train_params = self.configs.get('train_params', dict())
        # 模型的参数
        self.model_params = self.configs.get('model_params', dict())
        #print('self.model_params',self.model_params['model_dim'])

        self.device = device

        self.nl_emb = nn.Embedding(nl_vocab_len, self.model_params['model_dim'],padding_idx=0).to(self.device)

        self.path_emb = nn.Embedding(path_vocab_len, self.model_params['model_dim'], padding_idx=0).to(self.device)

        self.linear = nn.Linear(self.model_params['model_dim'], self.model_params['model_dim']).to(self.device)

        self.layer_norm = nn.LayerNorm(self.model_params['model_dim'], eps=1e-6)

        self.pos_ffn = PoswiseFeedForwardNet(self.configs)

        self.self_att = MultiHeadAttention(self.configs,self.pos_ffn ,self.device).to(self.device)


        self.dropout = nn.Dropout(dropout)

    def forward(self, nl, path, start_token, end_token):

        # 查询嵌入
        nl_emb = self.nl_emb(nl)
        nl_emb = self.dropout(nl_emb)

        desc_conv1 = nn.Conv1d(in_channels=self.data_params['nlseq_len'], out_channels=self.model_params['d_ff'],kernel_size=1).to(self.device)
        desc_conv2 = nn.Conv1d(in_channels=self.data_params['nlseq_len'], out_channels=self.model_params['d_ff'],kernel_size=1).to(self.device)
        desc_conv3 = nn.Conv1d(in_channels=self.data_params['nlseq_len'], out_channels=self.model_params['d_ff'],kernel_size=1).to(self.device)
        desc_conv1_out = desc_conv1(nl_emb)
        desc_conv2_out = desc_conv2(nl_emb)
        desc_conv3_out = desc_conv3(nl_emb)
        desc_conv1_dropout = self.dropout(desc_conv1_out)
        desc_conv2_dropout = self.dropout(desc_conv2_out)
        desc_conv3_dropout = self.dropout(desc_conv3_out)


        merged_desc = torch.cat((desc_conv1_dropout, desc_conv2_dropout, desc_conv3_dropout), dim=1)


        # 代码三部分注意力 [?,480,128]
        #start_token_emb = self.nl_emb(start_token.view(-1, 480),)
        start_token_emb = self.nl_emb(start_token.view(-1, 480),)
        start_token_emb = self.dropout(start_token_emb)

        stokens_conv1 = nn.Conv1d(in_channels=self.data_params['tokens_len'], out_channels=self.model_params['d_ff'],
                               kernel_size=1).to(self.device)
        stokens_conv2 = nn.Conv1d(in_channels=self.data_params['tokens_len'], out_channels=self.model_params['d_ff'],
                               kernel_size=1).to(self.device)
        stokens_conv3 = nn.Conv1d(in_channels=self.data_params['tokens_len'], out_channels=self.model_params['d_ff'],
                               kernel_size=1).to(self.device)
        stokens_conv1_out = stokens_conv1(start_token_emb)
        stokens_conv2_out = stokens_conv2(start_token_emb)
        stokens_conv3_out = stokens_conv3(start_token_emb)


        stokens_conv1_dropout = self.dropout(stokens_conv1_out)
        stokens_conv2_dropout = self.dropout(stokens_conv2_out)
        stokens_conv3_dropout = self.dropout(stokens_conv3_out)
        merged_stokens = torch.cat((stokens_conv1_dropout, stokens_conv2_dropout, stokens_conv3_dropout), dim=1)


        #[?,480,128]
        end_token_emb = self.nl_emb(end_token.view(-1, 480),)
        end_token_emb = self.dropout(end_token_emb)
        etokens_conv1 = nn.Conv1d(in_channels=self.data_params['tokens_len'], out_channels=self.model_params['d_ff'],
                                  kernel_size=1).to(self.device)
        etokens_conv2 = nn.Conv1d(in_channels=self.data_params['tokens_len'], out_channels=self.model_params['d_ff'],
                                  kernel_size=1).to(self.device)
        etokens_conv3 = nn.Conv1d(in_channels=self.data_params['tokens_len'], out_channels=self.model_params['d_ff'],
                                  kernel_size=1).to(self.device)
        etokens_conv1_out = etokens_conv1(end_token_emb)
        etokens_conv2_out = etokens_conv2(end_token_emb)
        etokens_conv3_out = etokens_conv3(end_token_emb)

        etokens_conv1_dropout = self.dropout(etokens_conv1_out)
        etokens_conv2_dropout = self.dropout(etokens_conv2_out)
        etokens_conv3_dropout = self.dropout(etokens_conv3_out)
        merged_etokens = torch.cat((etokens_conv1_dropout, etokens_conv2_dropout, etokens_conv3_dropout), dim=1)


        #[?,480,128]
        path_emb = self.path_emb(path.view(-1, 480),)
        path_emb = self.dropout(path_emb)

        path_conv1 = nn.Conv1d(in_channels=self.data_params['tokens_len'], out_channels=self.model_params['d_ff'],
                                  kernel_size=1).to(self.device)
        path_conv2 = nn.Conv1d(in_channels=self.data_params['tokens_len'], out_channels=self.model_params['d_ff'],
                                  kernel_size=1).to(self.device)
        path_conv3 = nn.Conv1d(in_channels=self.data_params['tokens_len'], out_channels=self.model_params['d_ff'],
                                  kernel_size=1).to(self.device)
        path_conv1_out = path_conv1(path_emb)
        path_conv2_out = path_conv2(path_emb)
        path_conv3_out = path_conv3(path_emb)

        path_conv1_dropout = self.dropout(path_conv1_out)
        path_conv2_dropout = self.dropout(path_conv2_out)
        path_conv3_dropout = self.dropout(path_conv3_out)
        merged_path = torch.cat((path_conv1_dropout, path_conv2_dropout, path_conv3_dropout), dim=1)

        #代码表示 [?,480,128]
        merged_code = torch.cat((merged_stokens, merged_etokens, merged_path), dim=1)

        # print('merged_desc',merged_desc.shape)
        # print('merged_code',merged_code.shape)

        #中间矩阵U [128,128]
        U = torch.randn(self.model_params['model_dim'],self.model_params['model_dim'])
        U = U.to(self.device)

        #查询与代码交互  [?,12,480]                      [?,12,128] [128,128] [?,480,128]
        desc_code = torch.tanh(torch.matmul(torch.matmul(merged_desc, U),merged_code.transpose(1, 2)))
        #print('desc_code', desc_code.shape)
        #最大池化
        li_max_pool = nn.MaxPool1d(900)
        #[?,12,1]
        delta_desc = torch.softmax(li_max_pool(desc_code),dim=1)
        #print('delta_desc', delta_desc.shape)

        desc_code = desc_code.transpose(1, 2)
        co_max_pool = nn.MaxPool1d(300)
        delta_code = co_max_pool(desc_code)
        delta_code = delta_code.transpose(1, 2)
        #[?,1,480]
        delta_code = torch.softmax(delta_code,dim=2)
        #print('delta_code', delta_code.shape)


        #与初始的相乘
        #[?,128]                     [?,12,128]  [?,12,1]
        feat_desc = torch.matmul(merged_desc.transpose(1, 2),delta_desc).squeeze(2)
        # print('feat_desc1', feat_desc.shape)
        #[?,128]                [?,1,480]  [?,480,128]
        feat_code = torch.matmul(delta_code,merged_code).squeeze(1)
        # print('feat_code1', feat_code.shape)

        feat_desc = self.linear(feat_desc)
        feat_code = self.linear(feat_code)

        feat_desc = self.layer_norm(feat_desc)
        feat_code = self.layer_norm(feat_code)
        # print('feat_code2', feat_code.shape)
        # print('feat_desc2', feat_desc.shape)


        return feat_desc, feat_code


