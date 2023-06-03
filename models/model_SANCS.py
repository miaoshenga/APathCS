import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self,configs,pos_ffn,dropout = 0.25):
        super(MultiHeadAttention, self).__init__()

        self.pos_ffn = pos_ffn
        # 模型的参数
        self.model_params = configs.get('model_params', dict())

        self.model_dim = self.model_params['model_dim']
        self.n_heads = self.model_params['num_heads']
        self.dim_per_head =self.model_dim // self.n_heads
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

        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(dim_per_head)

        attn = nn.Softmax(dim=-1)(scores)  # softmax函数按横行来做
        context = torch.matmul(attn, v_s)

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
        self.dropout = nn.Dropout(0.25)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        context = self.dropout(output)
        return self.layer_norm(context + residual)


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

        self.device = device

        self.nl_emb = nn.Embedding(nl_vocab_len, self.model_params['model_dim'],padding_idx=0).to(self.device)

        self.path_emb = nn.Embedding(path_vocab_len, self.model_params['model_dim'], padding_idx=0).to(self.device)

        self.linear = nn.Linear(self.model_params['model_dim'], self.model_params['model_dim']).to(self.device)

        self.layer_norm = nn.LayerNorm(self.model_params['model_dim'], eps=1e-6)

        self.pos_ffn = PoswiseFeedForwardNet(self.configs)

        self.self_att = MultiHeadAttention(self.configs,self.pos_ffn ).to(self.device)

        self.dropout = nn.Dropout(dropout)

    def joint_encoding(self, repr1, repr2):
        batch_size = repr1.size(0)
        code_repr = self.self_att(repr1, repr2,repr2)
        desc_pos_repr = self.self_att(repr2, repr1,repr1)
        code_repr = torch.mean(self.linear(code_repr), dim=1)
        desc_pos_repr = torch.mean(self.linear(desc_pos_repr), dim=1)
        return desc_pos_repr,code_repr

    def forward(self, nl, path, start_token, end_token):

        # 查询嵌入
        desc_repr = self.nl_emb(nl)
        # 代码三部分注意力 [?,480,128]
        start_token_emb = self.nl_emb(start_token.view(-1, 480),)
        #[?,480,128]
        end_token_emb = self.nl_emb(end_token.view(-1, 480))
        #[?,480,128]
        path_emb = self.path_emb(path.view(-1, 480))
        code_repr = torch.cat((start_token_emb, end_token_emb, path_emb), dim=1)
        desc_pos_repr, code_repr = self.joint_encoding(code_repr, desc_repr)
        return desc_pos_repr, code_repr


