import numpy as np
from PIL import Image

class PaliGemmaProcessor:
    image_token = '<image>'
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def __init__(self, tokenizer, image_size):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.set_tokens()
    
    def set_tokens(self):
        special_token = {"additional_special_tokens": [self.image_token]}
        extra_tokens = [f"loc{i:04d}" for i in range(1024)]
        extra_tokens += [f"seg{i:03d}" for i in range(128)]
        self.tokenizer.add_special_tokens(special_token)
        self.tokenizer.add_tokens(extra_tokens)
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False

    def preprocess_image(self, images):
        for image in images:
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image) * (1/255.).astype(np.float32)
            image = (image - self.mean) / self.std
            image = image.transpose(2, 0, 1)
        return images

    def __call__(self, images, texts):
        images = self.preprocess_image(images)