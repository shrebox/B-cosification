# Loading a pretrained ResNet18 and converting it to a B-COS ResNet18
import math
import warnings

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from bcos.common import BcosUtilMixin
from bcos.modules import LogitLayer, norms
from bcos.modules.bcosifyconv2d import BcosifyConv2d
from bcos.modules.bcosifylinear import BcosifyLinear

IMAGENET_MEAN_ADDINVERSE = (0.485, 0.456, 0.406, 0.515, 0.544, 0.594)
IMAGENET_STD_ADDINVERSE = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)

CLIP_MEAN_ADDINVERSE = (0.48145466, 0.4578275, 0.40821073, 0.51854534, 0.5421725, 0.59178927)
CLIP_MEAN_ZERO = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
CLIP_STD_ADDINVERSE = (0.26862954, 0.26130258, 0.27577711, 0.26862954, 0.26130258, 0.27577711)

import numpy as np

from bcos.modules.common import DetachableModule


class MyGELU(DetachableModule):
    def forward(self, x):
        gate = 0.5 * (1 + torch.erf(x/np.sqrt(2)))
        if self.detach:
            gate = gate.detach()
        return gate * x
    
# Add Inverse in present in the DataLoader for correct viz
# We simply normalize the 6 channels which is equivalent to BcosifyNormalization
class BcosifyNormLayer(BcosUtilMixin, nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.bcosifynormalize = transforms.Normalize(mean=IMAGENET_MEAN_ADDINVERSE,std=IMAGENET_STD_ADDINVERSE) # Bcosify normalization as 0th layer

    def forward(self, x):
        return self.model(self.bcosifynormalize(x))

class BcosifyNetwork(BcosUtilMixin, nn.Module):
    def __init__(self, model, model_config, add_channels=True, logit_layer=False):
        super().__init__()
        self.model = model
        self.model_config = model_config
        
        # Setting logit layer
        self.logit_layer = None
        if logit_layer:
            self.logit_bias = model_config.get("logit_bias", -math.log(1000 - 1))
            self.logit_temperature = model_config.get("logit_temperature", None)
            self.logit_layer = LogitLayer(logit_temperature=self.logit_temperature, logit_bias=self.logit_bias,)
        
        # Setting clip_kd 
        self.clip_kd = model_config['bcosify_args'].get('clip_kd', None)
        self.bfy_mean_zero = model_config.get('bfy_mean_zero', False)
        self.linearprobe_clip = model_config['bcosify_args'].get('linearprobe_clip', False)
        
        # Bcosify normalization as 0th layer
        # For CLIP cases: KD with zero mean
        if self.clip_kd and self.bfy_mean_zero: 
            self.bcosifynormalize = transforms.Normalize(mean=CLIP_MEAN_ZERO,std=CLIP_STD_ADDINVERSE) 
        # For CLIP cases: Linear probing and KD with non-zero mean
        elif (self.clip_kd or self.linearprobe_clip) and not self.bfy_mean_zero: 
            self.bcosifynormalize = transforms.Normalize(mean=CLIP_MEAN_ADDINVERSE,std=CLIP_STD_ADDINVERSE)
        # For standard cases
        else:
            self.bcosifynormalize = transforms.Normalize(mean=IMAGENET_MEAN_ADDINVERSE,std=IMAGENET_STD_ADDINVERSE) 
        
        # Add channels to the first convolutional layer to allow for 6 channel inputs
        if add_channels:
            BcosifyNetwork.add_channels(self.model)
        BcosifyNetwork.bcosify(self.model, self.model_config)

    def forward(self, x):
        if self.logit_layer:
            return self.logit_layer(self.model(self.bcosifynormalize(x)))
        return self.model(self.bcosifynormalize(x))
    
    @classmethod
    def add_channels(cls, model):
        found_linear_layer = False
        for name, module in model.named_modules():
            if not found_linear_layer and name == 'to_patch_embedding.conv_stem.0':
                module.in_channels = 6
                module.weight.data = torch.cat((module.weight.data, -module.weight.data), dim=1) / 2
                found_linear_layer = True
                break

            if not found_linear_layer and name == 'to_patch_embedding.linear':
                # Double the input features
                module.in_features *= 2  # from 768 to 1536

                # Original weight shape: [out_features, 768]
                W = module.weight.data  # Shape: [out_features, 768]

                # Reshape weights to process in groups of 3
                W_reshaped = W.view(module.out_features, -1, 3)  # Shape: [out_features, 256, 3]

                # Compute positive half: divide by 2
                W_pos = W_reshaped / 2  # Shape: [out_features, 256, 3]

                # Compute negative half: negative of positive half
                W_neg = -W_pos  # Shape: [out_featoures, 256, 3]

                # Concatenate positive and negative halves along the last dimension
                W_new = torch.cat([W_pos, W_neg], dim=2)  # Shape: [out_features, 256, 6]

                # Flatten the weights back to 2D
                W_new_flat = W_new.view(module.out_features, module.in_features)  # Shape: [out_features, 1536]

                # Update the weight data
                module.weight.data = W_new_flat

                # Set the flag to True to avoid modifying any other layers
                found_linear_layer = True
                break
        if not found_linear_layer:
            warnings.warn("No linear layer was found. " +
                          "Bcosification might thus not work as intended."
                          )
    @classmethod
    def bcosify(cls, model, model_config):
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                cls.bcosify(module, model_config)

            act_layer = model_config.get("act_layer", True)    
            if isinstance(module, nn.Conv2d):
                # replace Conv2d with BcosConv2d
                setattr(model, n, BcosifyConv2d.from_standard_module(module, model_config))
            elif isinstance(module, nn.Linear):
                # replace Linear with BcosLinear
                if n!= 'to_qkv': # Only modify c_proj (output layer) for the clip_kd
                    setattr(model, n, BcosifyLinear.from_standard_module(module, model_config))
            elif isinstance(module, nn.GELU):
                # replace GELU with DetachableGELU
                if act_layer:
                    setattr(model, n, MyGELU())
                else:
                    setattr(model, n, nn.Identity())
            elif isinstance(module, nn.LayerNorm):
                # replace LayerNorm with BcosLayerNorm
                setattr(model, n, norms.DetachableLayerNorm.from_standard_module(module, model_config))
            elif isinstance(module, nn.GroupNorm):
                # replace LayerNorm with BcosGroupNorm
                setattr(model, n, norms.DetachableGroupNorm2d.from_standard_module(module, model_config))
            else:
                # rest of the modules are not replaced
                pass