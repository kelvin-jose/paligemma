class PaliGemmaProcessor:
    def __init__(self, tokenizer):
        self.image_token = '<image>'
        self.tokenizer = tokenizer
        self.set_tokens()
    
    def set_tokens(self):
        special_token = {"additional_special_tokens": [self.image_token]}
        extra_tokens = [f"loc{i:04d}" for i in range(1024)]
        extra_tokens += [f"seg{i:03d}" for i in range(128)]
        self.tokenizer.add_special_tokens(special_token)
        self.tokenizer.add_tokens(extra_tokens)
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False
