import torch
import numpy as np

class PaliGemmaProcessor:
    image_token = '<image>'
    mean = np.array([0.5, 0.5, 0.5], dtype = np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype = np.float32)

    def __init__(self, tokenizer, image_size, image_seq_len, max_seq_len):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.image_size = image_size
        self.image_seq_len = image_seq_len
        self.max_seq_len = max_seq_len
        self.set_tokens()
    
    def set_tokens(self):
        print(f"[x] current vocab size: {self.tokenizer.vocab_size}")
        special_token = {"additional_special_tokens": [self.image_token]}
        extra_tokens = [f"loc{i:04d}" for i in range(1024)]
        extra_tokens += [f"seg{i:03d}" for i in range(128)]
        self.tokenizer.add_special_tokens(special_token)
        self.tokenizer.add_tokens(extra_tokens)
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False
        print(f"[x] current vocab size: {len(self.tokenizer)}")

    def preprocess_image(self, images):
        updated_images = []
        for image in images:
            image = image.resize((self.image_size, self.image_size))
            image = (np.array(image) * (1/255.)).astype(np.float32)
            image = (image - self.mean) / self.std
            image = image.transpose(2, 0, 1)
            updated_images.append(image)
        return updated_images

    def prepare_input_strings(self, prefix, suffix):
        strings = []
        for p, s in zip(prefix, suffix):
            strings.append(f"{self.image_token * self.image_seq_len}{self.tokenizer.bos_token}{p}\n{s}{self.tokenizer.eos_token}")
        return strings

    def __call__(self, images, prefix, suffix):
        images = self.preprocess_image(images)
        images = np.stack(images, axis = 0)
        image_tensors = torch.tensor(images)
        input_strings = self.prepare_input_strings(suffix, prefix)
        input_tokens = self.tokenizer(input_strings,
                                      return_tensors = "pt",
                                      padding = "max_length",
                                      max_length = self.max_seq_len)
        return {
            "image_tensors": image_tensors,
            **input_tokens
        }