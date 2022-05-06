import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

# https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/layer_norm.py
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
# https://github.com/openspeech-team/openspeech
class DotProductAttention(nn.Module):
    def __init__(self, dim, scale: bool):
        super(DotProductAttention, self).__init__()
        if scale:
            self.sqrt_dim = np.sqrt(dim)
        else:
            self.sqrt_dim = 1
            
    def forward(self, query, key, value, mask: Optional=None):
        score = torch.matmul(query, key.transpose(2, 3)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask, 1e-4)
        attn = F.softmax(score, -1)
        context = torch.matmul(attn, value)
        return context, attn
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        assert dim % num_heads == 0
        super(MultiHeadAttention, self).__init__()
        
        self.d_head = int(dim/num_heads)
        self.num_heads = num_heads
        
        self.query_proj = nn.Linear(dim, self.d_head * num_heads)
        self.key_proj = nn.Linear(dim, self.d_head * num_heads)
        self.value_proj = nn.Linear(dim, self.d_head * num_heads)
        self.scaled_dot_attn = DotProductAttention(dim, scale=True)
        
    def forward(self, query, key, value, mask: Optional=None):
        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key = self.query_proj(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value = self.query_proj(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_head)
        
        return context, attn
    
class BertBlock(nn.Module):
    def __init__(self, dim = 512, d_ff = 2048, num_heads = 4, dropout_p = 0.3):
        super(BertBlock, self).__init__()
        self.layer_norm = LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            self.layer_norm,
            nn.Dropout(dropout_p),
            nn.Linear(dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, dim),
            self.layer_norm,
            nn.Dropout(dropout_p),
        )
        
    def forward(self, inputs, mask: Optional=None):
        normalized = self.layer_norm(inputs)
        attend, _ = self.attention(normalized, normalized, normalized, mask)
        attend += normalized
        outputs = self.fc(attend)
        return outputs
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int=512, max_len: int=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, length: int):
        return self.pe[:, :length]
    
def get_attn_pad_mask(inputs, input_lengths, expand_length):
    def get_transformer_non_pad_mask(inputs, input_lengths):
        batch_size = inputs.size(0)
        if len(inputs.size()) == 2:
            non_pad_mask = inputs.new_ones(inputs.size())
        elif len(inputs.size()) == 3:
            non_pad_mask = inputs.new_ones(inputs.size()[:-1])
        else:
            raise ValueError(f"Unsupported input shape {inputs.size()}")
            
        for i in range(batch_size):
            non_pad_mask[i, input_lengths[i]:] = 0
        return non_pad_msak
    non_pad_mask = get_transformer_non_pad_mask(inputs, input_lengths)
    pad_mask = non_pad_mask.lt(1)
    attn_pad_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_pad_mask

def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
        
    return subsequent_mask

    