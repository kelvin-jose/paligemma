import torch 
import torch.nn as nn

class SigLIPConfig:
    image_size = 224
    num_channels = 3
    patch_size = 16

    hidden_size = 768
    intermediate_size = 3072
    num_layer = 12
    num_heads = 12

