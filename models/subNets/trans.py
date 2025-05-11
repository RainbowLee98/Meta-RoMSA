import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_fw(nn.Linear):
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast,
                           self.bias.fast)  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

# metana 中的Conv2d_fw
class Conv2d_fw(nn.Conv2d): # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if bias:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out

class Conv3d_fw(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv3d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if bias:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv3d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv3d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv3d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv3d_fw, self).forward(x)

        return out


class ScaledDotProductAttention(nn.Module):
    # 实现了自注意力机制中的 ** 缩放点积注意力 **，这是    Transformer    架构中的核心部分，用于在序列数据中捕捉全局依赖关系
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = Linear_fw(d_model, h * d_k)
        self.fc_k = Linear_fw(d_model, h * d_k)
        self.fc_v = Linear_fw(d_model, h * d_v)
        self.fc_o = Linear_fw(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        # 使用permute换维度
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        # attention_weights:注意力权重的乘法是为了对注意力机制进行额外控制或调整。这种控制可能来自于外部的先验信息或者额外的特征。
        # attention_mask：掩码的使用主要是为了忽略特定的时间步（tokens）。比如在处理不同长度的输入序列时，较短序列的填充部分（padding tokens）需要被掩盖，以防这些填充部分对注意力计算造成干扰。
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            att = att.masked_fill(~attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class seqEncoder(nn.Module):
    def __init__(self, input_dim, embd_size=128, head=2, in_channels=1, kernel_heights=5, dropout=0.0) -> None:
        super().__init__()

        self.conv = Conv2d_fw(in_channels, embd_size, (kernel_heights, input_dim), padding=((kernel_heights-1)//2, 0))
        self.self_attn = ScaledDotProductAttention(d_model=embd_size, d_k=embd_size, d_v=embd_size, h=head, dropout=dropout)

    def forward(self, x, mask=None):
        ''' x: modality sequences. [batch_size, embd_size]'''
        x = torch.unsqueeze(x, dim=1)
        b, l, d = x.size()
        x = x.view(b, 1, l, d)
        hidden = F.relu(self.conv(x).squeeze(3)).transpose(1,2)
        attn_hidden = self.self_attn(hidden, hidden, hidden, attention_mask=mask)
        attn_hidden = torch.squeeze(attn_hidden, dim=1)
        return attn_hidden

class meta_seqEncoder(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, kernel_heights=5, dropout=0.0) -> None:
        super().__init__()

        self.conv = Conv2d_fw(in_channels, embd_size, (kernel_heights, input_dim), padding=((kernel_heights-1)//2, 0))
        self.self_attn = ScaledDotProductAttention(d_model=embd_size, d_k=embd_size, d_v=embd_size, h=2, dropout=dropout)

    def embd_maxpool(self, r_out):

        in_feat = r_out.transpose(1,2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)


    def forward(self, x, mask=None):
        ''' x: modality sequences. [batch_size, seq_len, embd_size]'''

        b, l, d = x.size()
        x = x.view(b, 1, l, d)
        hidden = F.relu(self.conv(x).squeeze(3)).transpose(1,2)
        attn_hidden = self.self_attn(hidden, hidden, hidden, attention_mask=mask)
        return self.embd_maxpool(attn_hidden)

# class seqEncoder(nn.Module):
#     def __init__(self, input_dim, embd_size=128, head=2, in_channels=1, kernel_heights=5, dropout=0.0) -> None:
#         super().__init__()
#         # - `self.conv`: 使用二维卷积将输入序列映射到一个高维嵌入空间。
#         # - `self.self_attn`: 应用自注意力机制来捕获序列中元素之间的依赖关系。
#         # input_dim 三维都不固定，需要先获取才能得到
#         # self.conv = Conv3d_fw(in_channels, embd_size, (kernel_heights, input_dim), padding=((kernel_heights-1)//2, 0))
#         # self.conv = nn.Conv3d(in_channels, embd_size, (kernel_heights, kernel_heights, input_dim),
#         #                       padding=(0, (kernel_heights - 1) // 2, 0))
#         self.conv = Conv3d_fw(in_channels, embd_size, (kernel_heights, kernel_heights, input_dim),
#                               padding=(0, (kernel_heights - 1) // 2, 0))
#         self.self_attn = ScaledDotProductAttention(d_model=embd_size, d_k=embd_size, d_v=embd_size, h=head,
#                                                    dropout=dropout)
#         self.conv1d = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=1, padding=0)
#
#     def forward(self, x, target, mask=None):
#         ''' x: modality sequences. [batch_size, embd_size]'''
#         # mosi 三模态数据集
#         # Batch ，Times 帧 ，D 维度
#         # 三维卷积需要5d输入(batch, num_channels, depth, height, width)，需要填充2维
#         x = torch.unsqueeze(x, dim=1)
#         b, c, t, d = x.size()
#         x = x.view(b, c, t, 1, d)
#         hidden = F.relu(self.conv(x).squeeze(4))
#         B, C, D, X = hidden.shape
#         hidden = hidden.view(B, C, -1)
#         hidden = hidden.transpose(1, 2)
#         attn_hidden = self.self_attn(hidden, hidden, hidden, attention_mask=mask)
#
#         # 最终输出的注意力加权后的隐藏状态
#
#         # 需要自适应池化将翻译结果形状与原始结果形状靠拢
#         adaptive_pool = nn.AdaptiveAvgPool1d(target)
#         pool_hidden = adaptive_pool(attn_hidden.permute(0, 2, 1))
#         pool_hidden = pool_hidden.permute(0, 2, 1)
#         return pool_hidden

class seqEncoder_metana(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, kernel_heights=5, dropout=0.0) -> None:
        super().__init__()

        self.conv = Conv2d_fw(in_channels, embd_size, (kernel_heights, input_dim), padding=((kernel_heights-1)//2, 0))
        self.self_attn = ScaledDotProductAttention(d_model=embd_size, d_k=embd_size, d_v=embd_size, h=2, dropout=dropout)

    def embd_maxpool(self, r_out):

        in_feat = r_out.transpose(1,2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)
# # 参考——metana的seqencoder
# class seqEncoder(nn.Module):
#     def __init__(self, input_dim, embd_size=128, in_channels=1, kernel_heights=5, dropout=0.0) -> None:
#         super().__init__()
#
#         self.conv = Conv2d_fw(in_channels, embd_size, (kernel_heights, input_dim), padding=((kernel_heights-1)//2, 0))
#         self.self_attn = ScaledDotProductAttention(d_model=embd_size, d_k=embd_size, d_v=embd_size, h=2, dropout=dropout)
#
#     def embd_maxpool(self, r_out):
#
#         in_feat = r_out.transpose(1,2)
#         embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
#         return embd.squeeze(-1)
#
#
#     def forward(self, x, mask=None):
#         ''' x: modality sequences. [batch_size, seq_len, embd_size]'''
#
#         b, l, d = x.size()
#         x = x.view(b, 1, l, d)
#         hidden = F.relu(self.conv(x).squeeze(3)).transpose(1,2)
#         attn_hidden = self.self_attn(hidden, hidden, hidden, attention_mask=mask)
#         return self.embd_maxpool(attn_hidden)


# if __name__ == '__main__':
#     rd = np.random.RandomState(111)
#     matrix = rd.random((32,38,768))
#     print(matrix.shape)
#     # 挪到tensor，并且调整格式为float
#     matrix = torch.tensor(matrix)
#     matrix = matrix.to(torch.float32)
#     # 三维卷积需要5d输入(batch, num_channels, depth, height, width)，需要填充2维num_channels和height
#     # 都用1填充，depth * height * width = 时间步（帧） * 1 * 特征维度
#     x = torch.unsqueeze(matrix,dim = 1)
#     print(x.shape)
#     b, c, t, d = x.size()
#     # (batch, 1, depth, 1, width)
#     x = x.view(b, c, t, 1, d)
#     print(x.shape)
#     # self, input_dim, embd_size = 128, head = 2, in_channels = 1, kernel_heights = 5, dropout = 0.0
#     # padding 0*2 2*2 0*2 使卷积核可以应用（滑动）
#     conv =  nn.Conv3d(1, 128, (3,3, 768), padding=(0,(5 - 1) // 2,0))
#     hidden = conv(x)
#     print(hidden.shape)
#     # 卷积后的输出是 (B, C, D, H, W)B：批大小  C：通道数（通常作为每个位置的特征维度） D, H, W：深度、高度、宽度
#     # 去除被input_dim维度卷积完的原特征维度d，此时特征维度已经变成了1，因此无用
#     hidden = hidden.squeeze(4)
#     print(hidden.shape)
#     # 将后两维度展平以适应后续的注意力
#     B, C, D, X = hidden.shape
#     hidden = hidden.view(B, C, -1)
#     # 进行一个 relu激活，将所有负值置零，引入非线性，使得模型能捕捉更复杂的特征
#     hidden = F.relu(hidden)
#     # 因为点积注意力需要输入是(batch_size, sequence_length, embedding_dim)，转置一下
#     hidden = hidden.transpose(1,2)
#     print(hidden.shape)
#     # 点积注意力
#     self_attn = ScaledDotProductAttention(d_model=128, d_k=128, d_v=128, h=2, dropout=0.0)
#     attn_hidden = self_attn(hidden, hidden ,hidden,attention_mask = None)
#     print(attn_hidden.shape)
#     # 此时保留中间的维度，nq
#     # return
