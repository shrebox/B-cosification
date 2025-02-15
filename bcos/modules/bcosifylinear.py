"""
Contains a Linear layer which uses the B-cos transform.

NOTE: In case you're wondering why the convolution models do not use
`BcosLinear`, it's because maintaining two versions of essentially
the same thing would be very error-prone during development and testing!
"""
from typing import Union

import torch.linalg as LA
import torch.nn as nn
from torch import Tensor

from bcos.modules.bcoslinear import BcosLinear


class BcosifyLinear(BcosLinear):
    def __init__(self,
                 *args, 
                 clamping: bool = False,
                 b_loss: bool = False, 
                 **kwargs):
        super(BcosifyLinear, self).__init__(*args, **kwargs)
                
        self.clamping = clamping
        self.b_loss = b_loss

        self.linear = nn.Linear(
            in_features=self.in_features,
                out_features=self.out_features * self.max_out,
                bias=self.bias,
                device=self.device,
                dtype=self.dtype,
        )

    # Following property is added as the clip converts the input's dtype to conv1's weight's dtype at line 146 here:
    # https://github.com/openai/CLIP/blob/main/clip/model.py#L146
    @property
    def weight(self) -> Tensor:
        return self.linear.weight
    
    def forward(self, in_tensor: Tensor) -> Tensor:
        """
        Forward pass.
        Args:
            in_tensor: Input tensor. Expected shape: (*, H_in)

        Returns:
            B-cos Linear output on the input tensor.
            Shape: (*, H_out)
        """

        if self.clamping:
            b = self.b.clamp(1 + 1e-6)
        
        # Using the b=-1 with weight decay (loss)
        if self.b_loss:
            b = self.b + 2

        # Simple linear layer
        out = self.linear(in_tensor)

        # MaxOut computation
        if self.max_out > 1:
            M = self.max_out
            O = self.out_features  # noqa: E741
            out = out.unflatten(dim=-1, sizes=(O, M))
            out = out.max(dim=-1, keepdim=False).values

        # if B=1, no further calculation necessary
        if self.b == 1 and self.b_loss == False:
            return out

        # Calculating the norm of input vectors ||x||
        norm = LA.vector_norm(in_tensor, dim=-1, keepdim=True) + 1e-12

        # Calculate the dynamic scale (|cos|^(B-1))
        # Note that cos = (x·ŵ)/||x||
        maybe_detached_out = out
        if self.detach:
            maybe_detached_out = out.detach()
            norm = norm.detach()

        if self.b == 2 and self.b_loss == False:
            dynamic_scaling = maybe_detached_out.abs() / norm
        else:
            abs_cos = (maybe_detached_out / norm).abs() + 1e-6
            if self.clamping or self.b_loss:
                dynamic_scaling = abs_cos.pow(b - 1)
            else:
                dynamic_scaling = abs_cos.pow(self.b - 1)

        # put everything together
        out = dynamic_scaling * out  # |cos|^(B-1) (ŵ·x)
        return out
    
    def extra_repr(self) -> str:
        # rest in self.linear
        s = "B={b}"

        if self.max_out > 1:
            s += ", max_out={max_out}"

        # final comma as self.linear is shown in next line
        s += ","
        additional_entries = dict(b=self.b.data.item()) if isinstance(self.b, nn.Parameter) else {}
        return s.format(**self.__dict__, **additional_entries)
    
    @classmethod
    def from_standard_module(cls, mod, model_config):
        """
        Create a BcosLinear from a standard nn.Linear module.
        """
        clamping = model_config['bcosify_args'].get("clamping", False)
        b_loss = model_config['bcosify_args'].get("learn_b", False)
        b = model_config['bcos_args'].get("b", 1)
        new_mod = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
            max_out=1,
            clamping=clamping,
            b_loss=b_loss,
            b=b,
        )
        weights = model_config.get("weights", None)
        if weights is not None:
            new_mod.linear.weight.data = mod.weight.data
            if mod.bias is not None:
                new_mod.linear.bias = nn.Parameter(mod.bias.data)
        return new_mod
