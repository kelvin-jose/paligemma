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
        self.langauge_tower = gemma.Gemma(gemma.GemmaConfig)
        self.input_processor = PaliGemmaProcessor(tokenizer, vision_config.image_size, (siglip.SigLIPConfig.image_size // siglip.SigLIPConfig.patch_size)**2, language_config.max_sequence_len)

    def forward(self, images, prefix, suffix):
        output = self.input_processor(images, prefix, suffix)
        img_tensor = output['image_tensors']
        input_ids = output['input_ids']
        image_features = self.vision_tower(img_tensor)