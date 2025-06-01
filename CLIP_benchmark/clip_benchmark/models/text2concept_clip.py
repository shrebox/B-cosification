import os
import sys

# Compute the absolute path 3 levels above current script
current_dir = os.path.dirname(os.path.abspath(__file__))
bcos_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, bcos_root)

import clip
import torch
import torchvision.transforms as transforms

from torch import Tensor

import bcos.data.transforms as custom_transforms
from .Text2Concept.TextToConcept import TextToConcept

# Adding AddInverse transform for B-cos models
class AddInverse(torch.nn.Module):
    """To a [B, C, H, W] input add the inverse channels of the given one to it.
    Results in a [B, 2C, H, W] output. Single image [C, H, W] is also accepted.

    Args:
        dim (int): where to add channels to. Default: -3
    """

    def __init__(self, dim: int = -3):
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor: Tensor) -> Tensor:
        return torch.cat([in_tensor, 1 - in_tensor], dim=self.dim)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def load_t2c_clip(model_name: str = "RN50", pretrained: str = "True", cache_dir: str = None, device="cpu"):
    model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
    encoder = torch.nn.Sequential(*list(model.children())[:-2]) # Remove the logit_layer and the last fc layer
    model.forward_features = lambda x : encoder(x)
    model.get_normalizer = AddInverse()
    model.has_normalizer = True
    text_to_concept = TextToConcept(model, 'bcos_resnet50')
    text_to_concept.load_linear_aligner('CLIP_benchmark/clip_benchmark/models/Text2Concept/pretrained_aligners/imagenet_bcos_resnet50_aligner_trainsetfull_noBiasLinearAligner.pth')
    model_aligned = text_to_concept.get_aligned_features_encoder()
    common_trans = [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    _convert_image_to_rgb,
                    transforms.ToTensor(),
                    custom_transforms.AddInverse()
                ]
    transform = transform=transforms.Compose(common_trans)
    tokenizer = clip.tokenize
    return model_aligned, transform, tokenizer
