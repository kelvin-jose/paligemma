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