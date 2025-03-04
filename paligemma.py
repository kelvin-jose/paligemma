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

    def generate_attention_mask(self, batch):
        b, n = batch.shape 
        attention_masks = torch.zeros((b, n, n), dtype=torch.int32)
        
        for i in range(b):
            seq = batch[i]

            sep_index = (seq == gemma.GemmaConfig.sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_index) == 0:
                continue  
            sep_index = sep_index.item()
            attention_masks[i, :sep_index + 1, :sep_index + 1] = 1
            
            for j in range(sep_index + 1, n):
                attention_masks[i, j, :j] = 1 

            pad_indices = (seq == gemma.GemmaConfig.pad_token_id).nonzero(as_tuple=True)[0]
            for pad_idx in pad_indices:
                attention_masks[i, pad_idx, :] = 0  
                attention_masks[i, :, pad_idx] = 0
        
        return attention_masks
    
    def forward(self, images, prefix, suffix):
        output = self.input_processor(images, prefix, suffix)
        img_tensor = output['image_tensors']
        input_ids = output['input_ids']
        image_features = self.vision_tower(img_tensor)
        embeds = self.langauge_tower.get_embeddings(input_ids)
        img_last_idx = image_features.shape[1]
        embeds[:, :img_last_idx, :] = image_features
        attention_mask = self.generate_attention_mask(input_ids)
        output = self.langauge_tower(embeds, attention_mask)

