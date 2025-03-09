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