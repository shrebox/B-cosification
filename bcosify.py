import math
import warnings

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from CLIP.clip.model import AttentionPool2d

from bcos.common import BcosUtilMixin
from bcos.modules import BcosAttentionPool2d, BcosSequential, LogitLayer
from bcos.modules.bcosifyconv2d import BcosifyConv2d
from bcos.modules.bcosifylinear import BcosifyLinear
from bcos.modules.norms.uncentered_norms import BatchNormUncentered2d

IMAGENET_MEAN_ADDINVERSE = (0.485, 0.456, 0.406, 0.515, 0.544, 0.594)
IMAGENET_STD_ADDINVERSE = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)

CLIP_MEAN_ADDINVERSE = (0.48145466, 0.4578275, 0.40821073, 0.51854534, 0.5421725, 0.59178927)
CLIP_MEAN_ZERO = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
CLIP_STD_ADDINVERSE = (0.26862954, 0.26130258, 0.27577711, 0.26862954, 0.26130258, 0.27577711)

class BcosifyNetwork(BcosUtilMixin, nn.Module):
    def __init__(self, model, model_config, add_channels=True, logit_layer=False):
        super().__init__()
        self.model = model
        self.model_config = model_config
        
        # Setting logit layer
        self.logit_layer = None
        if logit_layer:
            self.logit_layer = LogitLayer(logit_temperature=None, logit_bias=-math.log(1000 - 1),)
        
        # Setting clip_kd 
        self.clip_kd = model_config['bcosify_args'].get('clip_kd', None)
        self.bfy_mean_zero = model_config.get('bfy_mean_zero', False)
        self.linearprobe_clip = model_config['bcosify_args'].get('linearprobe_clip', False)
        # Bcosify normalization as 0th layer
        if self.clip_kd and self.bfy_mean_zero:
            self.bcosifynormalize = transforms.Normalize(mean=CLIP_MEAN_ZERO,std=CLIP_STD_ADDINVERSE) 
        elif (self.clip_kd or self.linearprobe_clip) and not self.bfy_mean_zero:
            self.bcosifynormalize = transforms.Normalize(mean=CLIP_MEAN_ADDINVERSE,std=CLIP_STD_ADDINVERSE)
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
        found_conv_layer = False
        for module in model.modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if module.in_channels == 3:
                if found_conv_layer:
                    warnings.warn("Found multiple layers with 3 input channels. " +
                                  "Bcosification might thus not work as intended.")
                found_conv_layer = True
                module.in_channels = 6
                module.weight.data = torch.cat((module.weight.data, -module.weight.data), dim=1) / 2
        if not found_conv_layer:
            warnings.warn("No conv layer with 3 input channels was found. " +
                          "However, 'add_channels' was set to True." +
                          "Bcosification might thus not work as intended."
                          )
                          
    @classmethod
    def bcosify(cls, model, model_config):
        bcosify_args = model_config.get("bcosify_args", None)
        clip_kd = bcosify_args.get("clip_kd", False) if bcosify_args is not None else False
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                if clip_kd and n == 'attnpool' and isinstance(module, AttentionPool2d): 
                    setattr(model, n, BcosAttentionPool2d.from_standard_module(model, module, model_config))
                    cls.bcosify(model.attnpool, model_config) # For Bcosifying the linear layers inside the BcosAttentionPool2d
                else:
                    # compound module, go inside it
                    cls.bcosify(module, model_config)
            
            norm_layer = model_config['bcosify_args'].get('norm_layer', 'BnUncV2')
            gap = model_config['bcosify_args'].get('gap', True) # Global Average Pooling reorder works with conv1x1 for the last linear layer
            last_layer_name = model_config.get('last_layer_name', 'NoLastLayerName')
            if isinstance(module, nn.Conv2d):
                # replace Conv2d with BcosConv2d
                setattr(model, n, BcosifyConv2d.from_standard_module(module, model_config))
            elif isinstance(module, nn.Linear) and (n!=last_layer_name or clip_kd or (not gap)): # For standard models as the modified model changes the forward and the last fc linear to conv1x1    
                # replace Linear with BcosLinear
                if n!= 'k_proj' and n!= 'v_proj' and n!= 'q_proj': # Only modify c_proj (output layer) for the clip_kd
                    setattr(model, n, BcosifyLinear.from_standard_module(module, model_config))
            elif isinstance(module, nn.Linear) and n==last_layer_name and gap:
                # replace Linear with BcosConv2d (conv1x1) for the last layer
                setattr(model, n, BcosifyConv2d.from_standard_module_linear(module, model_config))
                print('Last Linear Layer Bcosified (Conv1x1) with GAP')
            elif isinstance(module, nn.Sequential):
                # replace Sequential with BcosSequential
                setattr(model, n, BcosSequential.from_standard_module(module))
            elif isinstance(module, nn.BatchNorm2d) and (norm_layer == 'BnUnc2d' or norm_layer == 'BnUncV2'):
                ## Add the norms
                setattr(model, n, BatchNormUncentered2d.from_standard_module(module, model_config))
            else:
                # rest of the modules are not replaced
                pass
            if isinstance(module, nn.ReLU):
                # For ablation with removing ReLU activation with identity
                act_layer = model_config['bcosify_args'].get('act_layer', True)
                if not act_layer:
                    setattr(model, n, nn.Identity())