import torch

from bcosify_vit import BcosifyNetwork

__all__ = ["get_model"]

def get_model(model_config):
    # extract args
    arch_name = model_config["name"]
    args = model_config["args"]

    arch_name_load = "standard_"+arch_name # For loading the pre-trained standard model
    model = torch.hub.load("B-cos/B-cos-v2", arch_name_load, pretrained=True)
    print("Loaded model from hub")

    logit_layer = model_config.get("logit_layer", False) 
    model = BcosifyNetwork(model, model_config, add_channels=True, logit_layer=logit_layer)

    # Making all the bias parameters None
    if not model_config.get('bcosify_args', {}).get('use_bias', False):
        print("Removing bias parameters (making None)")
        for mod in model.modules():
            if hasattr(mod, "bias") and mod.bias is not None:
                mod.bias = None

    # GAP Re-order 
    gap_reorder = args.get("gap_reorder", False)
    if gap_reorder:
        model.model.gap_reorder = gap_reorder

    return model

