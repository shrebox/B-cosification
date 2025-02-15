from torch import nn
from torchvision.models.densenet import DenseNet121_Weights, _load_state_dict
from torchvision.models.resnet import (
    BasicBlock,
    Bottleneck,
    ResNet18_Weights,
    ResNet50_Weights,
)

from bcos.models.standard_models import DenseNetBcos, ResNetBcos
from bcosify import BcosifyNetwork

__all__ = ["get_model"]

def get_torch_model_modified(arch_name: str, model_config):
    if arch_name=='resnet18':
        tv_model = ResNetBcos(BasicBlock, [2, 2, 2, 2])
        weight_type = model_config["weights"]
        if weight_type:
            weights = ResNet18_Weights.verify(model_config["weights"])
            tv_model.load_state_dict(weights.get_state_dict(progress=False))
        return tv_model
    if arch_name=='resnet50':
        tv_model = ResNetBcos(Bottleneck, [3, 4, 6, 3])
        weight_type = model_config["weights"]
        if weight_type:
            weights = ResNet50_Weights.verify(model_config["weights"])
            tv_model.load_state_dict(weights.get_state_dict(progress=False))
        return tv_model
    if arch_name=='densenet121':
        tv_model = DenseNetBcos(32, (6, 12, 24, 16), 64)
        weight_type = model_config["weights"]
        if weight_type:
            weights = DenseNet121_Weights.verify(model_config["weights"])
            _load_state_dict(model=tv_model, weights=weights, progress=False)
        return tv_model

def get_model(model_config) -> nn.Module:
    assert model_config.get("is_bcos", False), "Should be true!"
    # extract args
    arch_name = model_config["name"]

    model = BcosifyNetwork(get_torch_model_modified(arch_name, model_config), model_config, add_channels=True, logit_layer=True) 

    # For standard changes
    standard_changes = model_config.get("standard_changes", None)
    for k,v in standard_changes.items():
        print("Changing maxpool to avgpool")
        exec(f'model.model.{k} = v')
    
    # Making all the bias parameters None
    print("Removing bias parameters (making None)")
    for mod in model.modules():
        if hasattr(mod, "bias") and mod.bias is not None:
            mod.bias = None

    return model