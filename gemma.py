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