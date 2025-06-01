import os
import sys

# Compute the absolute path 3 levels above current script
current_dir = os.path.dirname(os.path.abspath(__file__))
bcos_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, bcos_root)

import clip
import torch
import torchvision.transforms as transforms

import bcos.data.transforms as custom_transforms
from bcos.experiments.utils import Experiment

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def load_bcos_clip(model_name: str = "RN50", pretrained: str = "True", cache_dir: str = None, device="cpu"):   
    exp_path = "experiments/ImageNet/clip_bcosification/"
    exp_name = exp_path + model_name 
    # Loading Bcosifyed CLIP model
    exp = Experiment(exp_name)
    model = exp.load_trained_model()
    ## If the model_name contains attnUnpool, then add cosine_power to the model
    if "attnUnpool" in model_name:
        model.cosine_power = int(pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    common_trans = [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    _convert_image_to_rgb,
                    transforms.ToTensor(),
                    custom_transforms.AddInverse()
                ]
    transform = transform=transforms.Compose(common_trans)
    tokenizer = clip.tokenize
    return model, transform, tokenizer
