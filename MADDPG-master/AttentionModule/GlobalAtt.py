# 接受ei，ej和动作ai作为输入，输出eiself和eiother其中eiself对应Q，其他两个对应   K，V，
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from Local import split_heads, combine_heads


# ei的形状是(batch, seq_len, 2), 如现在所定义的就是(batch, 7, 2)
# 其中接收ej，aj的应该是多项的拼接，e1, e2, e3,形状是(batch, seq_len*(agent_num-1), 2),然后拼上联合动作(batch, (agent_num-1), 2)
# 最后的形状是(batch, 8*(n-1) , 2)

class GlobalAttention(nn.Module):
    def __init__(self, input_dim, embed_dim=128, attention_heads=4):
        super(GlobalAttention, self).__init__()
        self.head_num = attention_heads
        self.head_dim = int(embed_dim / attention_heads)
        assert self.head_dim == embed_dim / attention_heads, "head_dim is not an integer"

        # 前编码层，将输入x映射到隐含空间,变成batch,7,128
        self.embedding_layer = nn.Linear(input_dim, embed_dim)  # (2,128)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.layer_norm = nn.LayerNorm(embed_dim)
        #   -----用于y
        self.embedding_layer2 = nn.Linear(input_dim, embed_dim)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        # -------------------------------
        self.fc_q = nn.Linear(self.head_dim, self.head_dim)
        self.fc_k = nn.Linear(self.head_dim, self.head_dim)
        self.fc_v = nn.Linear(self.head_dim, self.head_dim)

        # -------------------------------

    def forward(self, x, y):
        # 这里输入的分别是当前agent i 的表达，观察和动作的拼接后的表达。x为(batch_size, 8, 2)
        # 以及agent j（其他的）的表达, y为(batch_size, 8*(n-1), 2)

        # 第一步：前编码阶段，这其实就是整个pre-encoding层
        ei = self.embedding_layer(x)  # [batch_size, seq_len, embed_dim]
        ei = self.leaky_relu(ei)
        ei = self.layer_norm(ei)  # (batch, 8, 128)
        ei_mul = split_heads(ei, self.head_num)  # (batch_size, num_heads, seq_len, head_dim)
        # 这里head_dim为128/4 = 32，所以维度为(batch, 4, 8, 32)
        ej = self.embedding_layer2(y)
        ej = self.leaky_relu(ej)
        ej = self.layer_norm2(ej)
        ej = split_heads(ej, self.head_num)

        Q = self.fc_q(ei_mul)       # (batch, 4, 8, 32)
        K = self.fc_k(ej)           # (batch, 4, 8*(n-1), 32)
        V = self.fc_v(ej)           # (batch, 4, 8*(n-1), 32)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, 4, 8, 8*(n-1))
        attention_probs = F.softmax(attention_scores, dim=-1)  # (batch_size, 4, 8, 8*(n-1))
        vi = torch.matmul(attention_probs, V)  # (batch_size, 4, 8*(n-1), 32)

        vi = combine_heads(vi, self.head_num)
        # 返回的形状都是(batch, 8, 128)
        return ei, vi
