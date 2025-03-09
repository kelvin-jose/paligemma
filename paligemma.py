import gemma
import siglip

import torch
import torch.nn as nn

class PaliGemma(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = siglip.SigLIP(siglip.SigLIPConfig)
        self.langauge_tower = gemma.Gemma(gemma.GemmaConfig)
    
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
    
    def generate_labels(self, input_ids):
        b, n = input_ids.shape
        labels = torch.full_like(input_ids, gemma.GemmaConfig.ignore_index)

        for i in range(b):
            row = input_ids[i]
            suffix_idx = (row == gemma.GemmaConfig.sep_token_id).nonzero(as_tuple=True)[0]
            if not len(suffix_idx):
                continue
            for j in range(len(suffix_idx)):
                start_idx = suffix_idx[j] + 1
                end_idx = (row[start_idx:] == gemma.GemmaConfig.eos_token_id).nonzero(as_tuple=True)[0]
                if not len(end_idx):
                    continue
                end_idx = start_idx + end_idx[0]

                if start_idx < end_idx:
                    labels[i, start_idx: end_idx - 1] = row[start_idx + 1: end_idx]
                    labels[i, end_idx - 1] = gemma.GemmaConfig.eos_token_id
        return labels
    
    def forward(self, image_tensors, input_ids):
        image_features = self.vision_tower(image_tensors)
        embeds = self.langauge_tower.get_embeddings(input_ids)
        img_last_idx = image_features.shape[1]
        embeds[:, :img_last_idx, :] = image_features
        attention_mask = self.generate_attention_mask(input_ids)
        output = self.langauge_tower(embeds, attention_mask)
        labels = self.generate_labels(input_ids)
        return output, labels

    
