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


class ImageProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.conv_layer = nn.Conv2d(in_channels = config.num_channels, 
                                    out_channels = config.hidden_size, 
                                    kernel_size = config.patch_size,
                                    stride = config.patch_size,
                                    padding = "valid")
        self.position_encoding = nn.Embedding(num_embeddings = (config.image_size // config.patch_size) ** 2,
                                              embedding_dim = config.hidden_size)
        self.register_buffer("position_ids", torch.arange((config.image_size // config.patch_size) ** 2).view(1, -1),
                             persistent = False)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.conv_layer(batch)
        features = features.view(batch.shape[0], -1, self.conv_layer.out_channels)
        output = features + self.position_encoding(self.position_ids)
        return output
        

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.Q = nn.Linear(self.hidden_size, self.hidden_size)
        self.K = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.scale_factor = (self.hidden_size // self.num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        batch_size = x.shape[0]
        timesteps = x.shape[1]
        queries = self.Q(x)
        keys = self.K(x)
        values = self.V(x)

        queries = queries.view(batch_size, self.num_heads, timesteps, self.hidden_size // self.num_heads)
        keys = keys.view(batch_size, self.num_heads, timesteps, self.hidden_size // self.num_heads)
        values = values.view(batch_size, self.num_heads, timesteps, self.hidden_size // self.num_heads)

        attention = torch.matmul(queries, keys.transpose(2, 3)) * self.scale_factor
        scores = nn.functional.softmax(attention, -1)
        scores = nn.functional.dropout(scores, training = self.training)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2)
        output = output.flatten(2)
        output = self.linear(output)
        return output