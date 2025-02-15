import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DetachableModule

__all__ = ["BcosAttentionPool2d"]

# https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L58C8-L58C22
class BcosAttentionPool2d(DetachableModule):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, attn_unpool: bool = False):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        if not attn_unpool:
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.attn_unpool = attn_unpool

    def forward(self, x):
        if self.attn_unpool:
            x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC, C=embed_dim
            x = self.v_proj(x)
            x = self.c_proj(x)
            # Typically, output is N x D'
            # New output is (HW) x N x D'
            norm = x.norm(dim=-1, keepdim=True)
            if self.detach:
                norm = norm.detach()
            return x / norm
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        q = x[:1]
        k = x
        if self.detach:
            q = q.detach()
            k = k.detach()
        x, _ = F.multi_head_attention_forward(
            query=q, key=k, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=None,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

    @classmethod
    def from_standard_module(cls, model, module, model_config):
        spacial_dim = model.input_resolution // 32
        embed_dim = model.conv1.out_channels * 64
        num_heads = module.num_heads
        output_dim = model.output_dim
        attn_unpool = model_config.get("attn_unpool", False)
        new_module = cls(spacial_dim, embed_dim, num_heads, output_dim, attn_unpool)
        
        # Copying the parameter values
        # ! For float32 setting, can directly copy the parameters
        weights = model_config.get("weights", None)
        if weights is not None:
            for name, param in module.named_parameters():
                if attn_unpool and ('k_proj' not in name) and ('q_proj' not in name):
                    exec(f'new_module.{name}.data = param.data')
        return new_module
    
    