import torch
import torch.nn as nn

class GemmaConfig:
    vocab_size = 256
    embed_dim = 768
    max_sequence_len = 2048
    pad_token_id = 0
    sep_token_id = 108
    eos_token_id = 1
    image_token_id = 256000
    vocab_size = 257153

    num_heads = 8
    num_kv = 2
    head_dim = 96
    num_layers = 12
    intermediate_dim = 2048

    eps = 0.00001

    theta = 10000

    ignore_index = -100

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.eps
        self.param = nn.Parameter(torch.zeros(config.embed_dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
        x = x * (1.0 + self.param)
        return x

class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        theta = config.theta ** (-torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        self.register_buffer("theta", theta, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        batch_size, num_heads, time_steps, head_dim = x.shape
        pos_emb = position_ids.float().unsqueeze(-1) * self.theta.view(1, 1, -1)
    
        cos, sin = pos_emb.cos(), pos_emb.sin()

        cos = cos.unsqueeze(1).expand(-1, num_heads, -1, -1)
        sin = sin.unsqueeze(1).expand(-1, num_heads, -1, -1)

        x1, x2 = x[..., ::2], x[..., 1::2]
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos
        rot = torch.stack([rot_x1, rot_x2], dim=-1).reshape(batch_size, num_heads, time_steps, head_dim)
        return rot

    
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
        self.rope = RoPE(config)

    def forward(self, x, attention_mask):
        batch_size = x.shape[0]
        time_steps = x.shape[1]

        queries = self.Q(x)
        keys = self.K(x)
        values = self.V(x)

        queries = queries.view(batch_size, self.num_heads, time_steps, self.head_dim)
        keys = keys.view(batch_size, self.num_kv, time_steps, self.head_dim)
        values = values.view(batch_size, self.num_kv, time_steps, self.head_dim)
        position_ids = torch.arange(time_steps).expand(batch_size, -1)
        keys = self.rope(keys, position_ids)
        values = self.rope(values, position_ids)
        keys = keys.repeat(1, self.num_heads // self.num_kv, 1, 1)
        values = values.repeat(1, self.num_heads // self.num_kv, 1, 1)
        scores = queries @ keys.transpose(2, 3) / self.head_dim ** -0.5
        mask = attention_mask.unsqueeze(1) == 0
        scores = scores.masked_fill(mask, 1e-9)
        scores = nn.functional.softmax(scores, -1)
        scores = nn.functional.dropout(scores, training = self.training)
        output = scores @ values 
        output = output.transpose(1, 2)
        output = output.flatten(2)
        output = self.linear(output)
        return output
        
class GemmaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rms_norm1 = RMSNorm(config)
        self.mha = MultiHeadedAttention(config)
        self.rms_norm2 = RMSNorm(config)
        self.ffn1 = nn.Linear(config.embed_dim, config.intermediate_dim)
        self.ffn2 = nn.Linear(config.intermediate_dim, config.embed_dim)

    def forward(self, x: torch.FloatTensor, attention_mask: torch.IntTensor) -> torch.FloatTensor:
        out = self.rms_norm1(x)
        out = self.mha(out, attention_mask)
        out = out + x
        residual = out
        out = self.rms_norm2(out)
        out = self.ffn1(out)
        out = nn.functional.gelu(out, approximate='tanh')
        out = self.ffn2(out)
        out = out + residual
        return out
    
class GemmaDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([GemmaBlock(config) for i in range(config.num_layers)])

    def forward(self, x: torch.FloatTensor, attention_mask: torch.IntTensor) -> torch.FloatTensor:
        out = x
        for layer in self.layers:
            out = layer(out, attention_mask)
        return out
        
class Gemma(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.decoder = GemmaDecoder(config)
        self.linear = nn.Linear(config.embed_dim, config.vocab_size)

    def get_embeddings(self, indices):
        return self.embeddings(indices)

    def forward(self, batch: torch.FloatTensor, attention_mask: torch.IntTensor) -> torch.FloatTensor:
        out = self.decoder(batch, attention_mask)
        logits = self.linear(out)
        probs = nn.functional.softmax(logits, -1)
        return probs