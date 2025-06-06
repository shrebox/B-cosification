from typing import Union
import torch
from .open_clip import load_open_clip
from .japanese_clip import load_japanese_clip
from .standard_clip import load_standard_clip
from .bcos_clip import load_bcos_clip
from .text2concept_clip import load_t2c_clip
from .bcos_clip_cc3m import load_bcos_clip_cc3m

# loading function must return (model, transform, tokenizer)
TYPE2FUNC = {
    "open_clip": load_open_clip,
    "ja_clip": load_japanese_clip,
    "standard_clip": load_standard_clip,
    "bcos_clip": load_bcos_clip,
    "text2concept_clip": load_t2c_clip,
    "bcos_clip_cc3m": load_bcos_clip_cc3m,
}
MODEL_TYPES = list(TYPE2FUNC.keys())


def load_clip(
        model_type: str,
        model_name: str,
        pretrained: str,
        cache_dir: str,
        device: Union[str, torch.device] = "cuda"
):
    assert model_type in MODEL_TYPES, f"model_type={model_type} is invalid!"
    load_func = TYPE2FUNC[model_type]
    return load_func(model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, device=device)
