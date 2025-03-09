import re
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from gemma import GemmaConfig
from siglip import SigLIPConfig
from paligemma import PaliGemma
from processor import PaliGemmaProcessor

from transformers import AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("MathLLMs/MathVision")
test = dataset['test']
train = dataset['testmini']

images = []
prefix = []
suffix = []
for row in test:
    p = re.sub(r"<image\d+>", "", row['question']).replace("\n", "")
    s = row['answer']
    i = row['decoded_image']
    prefix.append(p)
    suffix.append(s)
    images.append(i)

class TrainConfig:
    batch_size = 1
    device = 'cpu'
    epochs = 200
    learning_rate = 1e-2
    weight_decay = 1e-5

    log_interval = 10
    log_tensorboard = True

class Dataset:
    def __init__(self, images, prefix, suffix):
        assert len(images) == len(prefix) == len(suffix)
        self.images = images
        self.prefix = prefix
        self.suffix = suffix

    def get_batch(self, batch_size):
        for i in range(0, len(self.images), batch_size):
            yield {
                "images": self.images[i:i + batch_size],
                "prefix": self.prefix[i:i + batch_size],
                "suffix": self.suffix[i:i + batch_size]
            }

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

class Trainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.input_processor = PaliGemmaProcessor(tokenizer, 
                                                  SigLIPConfig.image_size, 
                                                  (SigLIPConfig.image_size // SigLIPConfig.patch_size)**2, 
                                                  GemmaConfig.max_sequence_len)
    def init_weights_xavier(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)    

    def train(self):
        self.model.to(self.config.device)
        self.model.apply(self.init_weights_xavier)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[x] trainable parameters: {num_params:,}")
        optimizer = optim.AdamW(self.model.parameters(), self.config.learning_rate, weight_decay=self.config.weight_decay)

        self.model.train()

        if self.config.log_tensorboard:
            writer = SummaryWriter("runs/paligemma/train")
            
        for epoch in range(self.config.epochs):
            for idx, batch in enumerate(self.dataset.get_batch(self.config.batch_size)):
                optimizer.zero_grad()
                output = self.input_processor(batch['images'], batch['prefix'], batch['suffix'])
                output, labels = self.model(output['image_tensors'], output['input_ids'])
                output = output.permute(0, 2, 1)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                if (epoch + 1) * (idx + 1) % self.config.log_interval == 0:
                    if self.config.log_tensorboard:
                        writer.add_scalar("Loss/Train", loss.item(), (epoch + 1) * (idx + 1))
                    else:
                        print(f'[x] epoch: {epoch} | step: {idx} | loss: {loss.item()}')
                optimizer.step()
        if self.config.log_tensorboard:
            writer.close()

pg = PaliGemma()
tconfig = TrainConfig()
dataset = Dataset(images[:4], prefix[:4], suffix[:4])
trainer = Trainer(pg, dataset, tconfig)
trainer.train()