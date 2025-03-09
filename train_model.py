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