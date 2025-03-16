from torch import nn

from bcosify import BcosifyNetwork
from CLIP import clip

__all__ = ["get_model"]

def get_model(model_config) -> nn.Module:
    assert model_config.get("is_bcos", False), "Should be true!"

    model, _ = clip.load("RN50", pretrained=True)
    model.float() # Float 32 conversion
    print("Loaded CLIP model")
    
    model = BcosifyNetwork(model.visual, model_config, add_channels=True, logit_layer=False)

    # Making all the bias parameters None
    print("Removing bias parameters (making None)")
    for mod in model.modules():
        if hasattr(mod, "bias") and mod.bias is not None:
            mod.bias = None
        if hasattr(mod, "positional_embedding") and mod.positional_embedding is not None:
            mod.positional_embedding = None

    return model
    