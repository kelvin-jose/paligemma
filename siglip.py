import torch 
import torch.nn as nn

class SigLIPConfig:
    image_size = 224
    num_channels = 3
    patch_size = 16

    hidden_size = 768
    intermediate_size = 3072
    num_layers = 12
    num_heads = 8


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
        features = features.reshape(batch.shape[0], -1, self.conv_layer.out_channels)
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

    def forward(self, batch: torch.Tensor) -> torch.FloatTensor:
        batch_size = batch.shape[0]
        timesteps = batch.shape[1]
        queries = self.Q(batch)
        keys = self.K(batch)
        values = self.V(batch)

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

class SigLIPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.mha = MultiHeadedAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        out = self.layer_norm1(batch)
        out = self.mha(out)
        out = out + batch
        residual = out
        out = self.layer_norm2(out)
        out = self.ffn1(out)
        out = nn.functional.gelu(out, approximate='tanh')
        out = self.ffn2(out)
        out = out + residual
        return out
        
class SigLIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([SigLIPBlock(config) for i in range(config.num_layers)])

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        out = batch
        for block in self.blocks:
            out = block(out)
        return out
        
class SigLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_processor = ImageProcessor(config)
        self.encoder = SigLIPEncoder(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        out = self.img_processor(batch)
        out = self.encoder(out)
        out = self.layer_norm(out)
        return out
    