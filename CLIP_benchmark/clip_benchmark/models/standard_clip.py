import clip 
import torch

def load_standard_clip(model_name: str = "RN50", pretrained: str = "True", cache_dir: str = None, device="cpu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = clip.load(model_name)
    model = model.to(device)
    tokenizer = clip.tokenize
    return model, transform, tokenizer
