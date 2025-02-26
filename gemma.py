import torch
import torch.nn as nn

class GemmaConfig:
    vocab_size = 256
    embed_dim = 32

    num_heads = 8
    num_kv = 2
    head_dim = 4
    num_layers = 12
    intermediate_dim = 8

    eps = 0.001

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.eps
        self.param = nn.Parameter(torch.zeros(config.embed_dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
        x = x * (1.0 + self.param)
        return x
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv = config.num_kv
        self.head_dim = config.head_dim
        assert self.num_heads * self.head_dim == config.embed_dim
        self.Q = nn.Linear(config.embed_dim, config.embed_dim)
        self.K = nn.Linear(config.embed_dim, self.num_kv * self.head_dim)
        self.V = nn.Linear(config.embed_dim, self.num_kv * self.head_dim)
        self.linear = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x, attention_mask=0):
        batch_size = x.shape[0]
        time_steps = x.shape[1]

        queries = self.Q(x)
        keys = self.K(x)
        values = self.V(x)

        queries = queries.view(batch_size, self.num_heads, time_steps, self.head_dim)
        keys = keys.view(batch_size, self.num_kv, time_steps, self.head_dim)
        values = values.view(batch_size, self.num_kv, time_steps, self.head_dim)
        # todo: apply rotary positional encoding
        keys = keys.repeat(1, self.num_heads // self.num_kv, 1, 1)
        values = values.repeat(1, self.num_heads // self.num_kv, 1, 1)
        scores = queries @ keys.transpose(2, 3) / self.head_dim ** -0.5
        scores = scores + attention_mask
        scores = nn.functional.softmax(scores, -1)
        scores = nn.functional.dropout(scores, training = self.training)
        output = scores @ values 
        output = output.transpose(1, 2)
        output = output.flatten(2)
        output = self.linear(output)
        return output
        
        