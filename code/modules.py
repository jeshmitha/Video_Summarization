import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m
import numpy as np

device = torch.device('cuda:0')

class FrameScoring(torch.nn.Module):
      def __init__(self, input_dim):
        super().__init__()

        # Linear and Softmax
        self.frame_scoring = nn.Sequential(
            nn.Linear(input_dim,1),
            nn.Sigmoid()
        )

      def forward(self, x):
        #scores generation
        scores = self.frame_scoring(x)
        return scores

class EncoderBlock(torch.nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout_p):
        super().__init__()

        self.K=nn.Linear(input_dim, input_dim)
        self.Q=nn.Linear(input_dim, input_dim)
        self.V=nn.Linear(input_dim, input_dim)
        # Attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim= input_dim, num_heads=num_heads, dropout= dropout_p, batch_first=True)

    def forward(self, x):

        x=x.unsqueeze(0).to(device)

        K = self.K(x)  
        Q = self.Q(x)  
        V = self.V(x)

        attn_output, attn_output_weights = self.self_attn(Q, K, V, need_weights=True)
        attn_output_weights=attn_output_weights.squeeze(0)
        # graph refining
        attn_output_weights = attn_output_weights.cpu().detach().numpy()
        sorted_attn_output_weights=np.argsort(attn_output_weights)
        nf=attn_output_weights.shape[0]
        num_conn = m.ceil(0.15*nf)
        sorted_attn_output_weights=sorted_attn_output_weights[:, nf-num_conn:]

        graph=np.zeros((nf, nf))
        for i in range(nf):
            graph[i][sorted_attn_output_weights[i]] = 1
        
        x=attn_output

        return x.squeeze(0), torch.from_numpy(graph).to(device)

class GraphAttentionLayer(torch.nn.Module):
  def __init__(self, in_features, out_features, n_heads, dropout, leaky_relu_negative_slope, is_concat = True):
    super().__init__()
    self.is_concat = is_concat
    self.n_heads = n_heads
    if is_concat:
      assert out_features % n_heads == 0
      self.n_hidden = out_features // n_heads
    else:
      self.n_hidden = out_features

    self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
    self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
    self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
    self.softmax = nn.Softmax(dim=1)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, h, adj_mat):

    n_nodes = h.shape[0]

    g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
    g_repeat = g.repeat(n_nodes, 1, 1)
    g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
    g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
    g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
    e = self.activation(self.attn(g_concat))
    e = e.squeeze(-1)
    e = e.masked_fill(adj_mat == 0, float('-inf'))
    a = self.softmax(e)
    a = self.dropout(a)

    attn_res = torch.einsum('ijh,jhf->ihf', a, g)

    if self.is_concat:
      return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
    else:
      return attn_res.mean(dim=1)

class Generator(torch.nn.Module):
  def __init__(self, in_features, out_features, n_heads, dropout, leaky_relu_negative_slope):
        super().__init__()

        self.n_heads=n_heads

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionLayer(in_features, out_features, n_heads, dropout, leaky_relu_negative_slope, is_concat=True)
       
        # Activation function after first graph attention layer
        self.activation = nn.ELU()

        self.output = GraphAttentionLayer(out_features, out_features, n_heads, dropout, leaky_relu_negative_slope, is_concat=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
  
  def forward(self, x, adj_mat):

        x=torch.squeeze(x,0)
        adj_mat=torch.squeeze(adj_mat,0)

        n_nodes = x.shape[0]
    
        adj_mat=adj_mat.view(n_nodes,n_nodes,1)
        adj_mat=adj_mat.repeat(1,1,self.n_heads)

        # Apply dropout to the input
        x = self.dropout(x)
        x = self.layer1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output(x, adj_mat)

class Critic(torch.nn.Module):
    def __init__(self, input_dim, leaky_relu_negative_slope):
        super(Critic, self).__init__()

        self.cLSTM = nn.LSTM(input_dim, input_dim, num_layers=1)
        self.model = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.LeakyReLU(leaky_relu_negative_slope, inplace=True),
            nn.Linear(int(input_dim/2), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        sl=x.shape[0]
        f=x.shape[1]
        x=x.view(sl, 1, f)
        output, (h_n, c_n) = self.cLSTM(x)
        h = h_n[-1].squeeze()
        critic_output = self.model(h)
        return h, critic_output