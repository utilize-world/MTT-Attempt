# Local-Attention，仅有Q，V，

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAttention(nn.Module):
    def __init__(self, input_dim, embed_dim=128, attention_heads=4):
        super(LocalAttention, self).__init__()
        self.head_num = attention_heads
        # 前编码层，将输入映射到隐含空间
        self.embedding_layer = nn.Linear(input_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.layer_norm = nn.LayerNorm(embed_dim)
        # -------------------------------
        # W,两个全连接层 for Q
        #self.W1 = nn.Linear(32, 128)    # ?怎么可能是32,128，应该是32,32否则后面怎么连
        self.W1 = nn.Linear(32, 32)
        self.W_leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W2 = nn.Linear(32, 7)  # 之所以为7，是因为原数据中的2维数据有7组，最后的维度为[batch size, head, sequence_len, seq_len]
        # W2还要接softmax层，才能生成注意权重alpha   # 这里有可能进行改变，不一定是7
        # -------------------------------
        # pre-encoding layer, for v
        self.v_embedding_layer = nn.Linear(input_dim, embed_dim)
        self.v_leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.v_layer_norm = nn.LayerNorm(embed_dim)
        #  --------------------------------
        # 最后把(batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, feature)
        # 也就是(ba, 7, 128) -> (ba, 7 ,2)
        self.output = nn.Linear(128, 2)

        # 多头注意力层，4个注意力头
        # self.multihead_attn = nn.MultiheadAttention(embed_dim, attention_heads)



    def forward(self, x, y):
        # 输入x的形状是 [batch_size, num_features, 2]，需要调整为 [batch_size, seq_len, embed_dim]
        # batch_size, seq_len, _ = x.size()
        # x = x.view(batch_size, -1)

        # 第一步：前编码阶段，这其实就是整个pre-encoding层
        q = self.embedding_layer(x)  # [batch_size, seq_len, embed_dim]
        q = self.leaky_relu(q)
        q = self.layer_norm(q)
        q = split_heads(q, self.head_num)   # (batch_size, num_heads, seq_len, head_dim)
        # 这里head_dim为128/4 = 32，所以维度为(batch, 4, 7, 32)
        q = self.W1(q)  # (batch, 4, 7, 32)
        q = self.W_leaky_relu(q)
        q = self.W2(q)  # (batch, 4, 7, 7)
        attn_weights = F.softmax(q, dim=-1) # 注意力权重

        v = self.v_embedding_layer(y)
        v = self.leaky_relu(v)
        v = self.v_layer_norm(v)
        v = split_heads(v, self.head_num) # (batch_size, num_heads, seq_len, head_dim)

        attn_output = torch.matmul(attn_weights, v)
        # 形状 (batch_size, num_heads, seq_len, head_dim)
        # ------------------------最后拼起多头，得到最终的attn_output
        attn_output = combine_heads(attn_output, self.head_num)
        # (batch_size, seq_len, head_dim * num_head)
        attn_output = self.output(attn_output)


        return attn_weights, attn_output
    # attention weights * v = e
    # 这里的output就相当于e， (ba, 7, 2)


def split_heads(tensor, num_heads):
    batch_size, seq_len, feature_dim = tensor.size()
    head_dim = feature_dim // num_heads
    output = tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    return output  # 形状 (batch_size, num_heads, seq_len, feature_dim)


def combine_heads(tensor, num_heads):
    batch_size, num_heads, seq_len, head_dim = tensor.size()
    feature_dim = num_heads * head_dim
    output = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim)
    return output  # 形状 : (batch_size, seq_len, feature_dim)


# # 使用示例：
# input_dim = 2 + 2 + 2 * n + 2 * m  # 输入向量的维度（例如位置、速度等）
# model = LocalAttention(input_dim)
# sample_input = torch.randn(1024, 1 + 1 + n + m, 2)  # batch_size 为1024
# output = model(sample_input)
# print(output.shape)  # 输出的形状应为 [1024, seq_len, 7]