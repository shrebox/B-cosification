import torch.nn as nn
from torch import Tensor

from bcos.modules.bcosconv2d import BcosConv2d


class BcosifyConv2d(BcosConv2d):
    def __init__(self,
                 *args, 
                 clamping: bool = False,
                 b_loss: bool = False, 
                 **kwargs):
        super(BcosifyConv2d, self).__init__(*args, **kwargs)
                
        self.clamping = clamping
        self.b_loss = b_loss

        linear = nn.Conv2d 
        self.linear = linear(
            in_channels=self.in_channels,
                out_channels=self.out_channels * self.max_out,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
                padding_mode=self.padding_mode,
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
        Forward pass implementation.
        Args:
            in_tensor: Input tensor. Expected shape: (B, C, H, W)

        Returns:
            BcosConv2d output on the input tensor.
        """
        return self.forward_impl(in_tensor)

    def forward_impl(self, in_tensor: Tensor) -> Tensor:
        """
        Forward pass.
        Args:
            in_tensor: Input tensor. Expected shape: (B, C, H, W)

        Returns:
            BcosConv2d output on the input tensor.
        """
        # For clamping
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
            O = self.out_channels  # noqa: E741
            out = out.unflatten(dim=1, sizes=(O, M))
            out = out.max(dim=2, keepdim=False).values

        # if B=1, no further calculation necessary
        if self.b == 1 and self.b_loss == False:
            return out

        # Calculating the norm of input patches: ||x||
        norm = self.calc_patch_norms(in_tensor)

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
        Create a BcosConv2d from a standard Conv2d module.
        Args:
            mod: Standard Conv2d module.
        Returns:
            BcosConv2d module.
        """
        clamping = model_config['bcosify_args'].get("clamping", False)
        b_loss = model_config['bcosify_args'].get("learn_b", False) 
        b = model_config['bcos_args'].get("b", 1)

        new_mod = cls(
            in_channels=mod.in_channels,
            out_channels=mod.out_channels,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            clamping=clamping,
            b_loss=b_loss,
            b = b,
        )
        weights = model_config.get("weights", None)
        if weights is not None:
            new_mod.linear.weight.data = mod.weight.data
            if mod.bias is not None:
                new_mod.linear.bias = nn.Parameter(mod.bias.data)
        return new_mod

    # same method as from_standard_module specifcally for last layer where i replace a Linear layer with BcosConv2d
    @classmethod
    def from_standard_module_linear(cls, mod, model_config):
        """
        Create a BcosConv2d from a standard Linear module.
        Args:
            mod: Standard Linear module.
        Returns:
            BcosConv2d module.
        """
        clamping = model_config['bcosify_args'].get("clamping", False)
        b_loss = model_config['bcosify_args'].get("learn_b", False)
        b = model_config['bcos_args'].get("b", 1)
        new_mod = cls(
            in_channels=mod.in_features,
            out_channels=mod.out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=mod.bias is not None,
            padding_mode="zeros",
            clamping=clamping,
            b_loss=b_loss,
            b = b,
        )
        weights = model_config.get("weights", None)
        if weights is not None:
            new_mod.linear.weight.data = mod.weight.data.view_as(new_mod.linear.weight.data)
            if mod.bias is not None:
                new_mod.linear.bias = nn.Parameter(mod.bias.data)
        return new_mod
    




