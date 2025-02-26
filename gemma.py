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

