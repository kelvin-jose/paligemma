import gemma
import siglip
from processor import PaliGemmaProcessor

import torch
import torch.nn as nn
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

class PaliGemma(nn.Module):
    def __init__(self, vision_config, language_config):
        super().__init__()
        self.vision_tower = siglip.SigLIP(siglip.SigLIPConfig)