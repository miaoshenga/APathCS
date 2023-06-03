import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,configs,pos_ffn,device,dropout = 0.25):
        super(MultiHeadAttention, self).__init__()
        self.pos_ffn = pos_ffn
        # 模型的参数
        self.model_params = configs.get('model_params', dict())
        self.data_params = configs.get('data_params', dict())

        self.model_dim = self.model_params['model_dim']
        self.n_heads = self.model_params['num_heads']
        self.seq_len = self.data_params['seq_len']
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
        U = torch.randn(self.model_dim, batch_size*12).view(batch_size, self.n_heads, -1, dim_per_head)
        U = U.to(self.device)
        scores = torch.matmul(q_s, U.transpose(-1, -2))
        scores =  torch.matmul(scores.transpose(-1, -2),k_s)

        attn = nn.Softmax(dim=-1)(scores)  #softmax函数按横行来做
        context = torch.matmul(attn, v_s.transpose(-1, -2))

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)

        output = self.linear(context)

        context = self.dropout(output)

        output = self.layer_norm(context)

        # output = self.pos_ffn(output)

        return output
    def jointAttention(self,Q,K,V):
        dim_per_head = self.dim_per_head
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, dim_per_head).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, dim_per_head).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        U = torch.randn(self.model_dim,batch_size*16) .view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        U = U.to(self.device)
        scores = torch.matmul(q_s, U.transpose(-1, -2))
        scores = torch.matmul(scores.transpose(-1, -2), k_s)

        attn = nn.Softmax(dim=-1)(scores)  # softmax函数按横行来做

        context = torch.matmul(attn, v_s.transpose(-1, -2))
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)

        output = self.linear(context)

        context = self.dropout(output)
        return context

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,configs):
        self.model_params = configs.get('model_params', dict())
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=self.model_params['model_dim'], out_channels=self.model_params['d_ff'], kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.model_params['d_ff'], out_channels=self.model_params['model_dim'], kernel_size=1)
        self.layer_norm = nn.LayerNorm(self.model_params['model_dim'])

    def forward(self, inputs):
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output)

class Embedder(nn.Module):
    def __init__(self,configs, nl_vocab_len, path_vocab_len, device,dropout=0.2):
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

        self.self_att = MultiHeadAttention(self.configs,self.pos_ffn ,self.device).to(self.device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, nl, path, start_token, end_token):

        # 查询嵌入
        nl_emb = self.nl_emb(nl)
        nl_emb = self.dropout(nl_emb)
        # 查询多头注意力[?,12,128]
        nl_outputs = self.self_att(nl_emb, nl_emb, nl_emb)
        # 代码三部分注意力 [?,480,128]
        start_token_emb = self.nl_emb(start_token.view(-1, 480),)
        start_token_emb = self.dropout(start_token_emb)
        s_output = self.self_att(start_token_emb, start_token_emb, start_token_emb)

        #[?,480,128]
        end_token_emb = self.nl_emb(end_token.view(-1, 480))
        end_token_emb = self.dropout(end_token_emb)
        e_output = self.self_att(end_token_emb, end_token_emb, end_token_emb)

        #[?,480,128]
        path_emb = self.path_emb(path.view(-1, 480))
        path_emb = self.dropout(path_emb)
        p_output = self.self_att(path_emb, path_emb, path_emb)

        #代码表示 [?,480,128]
        code_output = self.self_att(s_output, e_output, p_output)

        #交互
        feat_desc = self.self_att.jointAttention(nl_outputs,nl_outputs,code_output)
        feat_code = self.self_att.jointAttention(code_output, code_output, nl_outputs)
        feat_code = torch.mean(self.linear(feat_code), dim=1)
        feat_desc = torch.mean(self.linear(feat_desc), dim=1)


        feat_desc = self.linear(feat_desc)
        feat_code = self.linear(feat_code)

        feat_desc = self.layer_norm(feat_desc)
        feat_code = self.layer_norm(feat_code)

        return feat_desc, feat_code


