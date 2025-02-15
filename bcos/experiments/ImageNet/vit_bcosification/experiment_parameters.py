"""
Configs for ViTs, both B-cos and non-B-cos (standard).

Paper: https://arxiv.org/abs/2205.01580
"""
import math  # noqa

import numpy as np
import torch
from torch import nn

from bcos.data.presets import (
    ImageNetClassificationPresetEval,
    ImageNetClassificationPresetTrain,
)
from bcos.experiments.utils import (
    configs_cli,
    create_configs_with_different_seeds,
    update_config,
)

# from bcos.experiments.utils import configs_cli, update_config
from bcos.modules import DetachableGNLayerNorm2d, norms
from bcos.modules.common import DetachableModule
from bcos.modules.losses import (
    BinaryCrossEntropyLoss,
    UniformOffLabelsBCEWithLogitsLoss,
)
from bcos.optim import LRSchedulerFactory, OptimizerFactory

__all__ = ["CONFIGS"]

NUM_CLASSES = 1_000

DEFAULT_BATCH_SIZE = 128  # per GPU! * 8 = 2048 effective
DEFAULT_NUM_EPOCHS = 90
DEFAULT_LR = 1e-3
DEFAULT_CROP_SIZE = 224


DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
    warmup_method="linear",
    warmup_steps=10_000,
    interval="step",
    warmup_decay=0.01,
)

DEFAULT_LR_SCHEDULE_180 = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=180,
    warmup_method="linear",
    warmup_steps=10_000,
    interval="step",
    warmup_decay=0.01,
)

LONG_WARM_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
    warmup_method="linear",
    warmup_steps=50_000,
    interval="step",
    warmup_decay=0.01,
)

LONG_WARM_SCHEDULE_180 = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=180,
    warmup_method="linear",
    warmup_steps=50_000,
    interval="step",
    warmup_decay=0.01,
)

DEFAULTS = dict(
    data=dict(
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=16,
        num_classes=NUM_CLASSES,
        mixup_alpha=0.2,
    ),
    model=dict(
        args=dict(
            num_classes=NUM_CLASSES,
        ),
    ),
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(
        max_epochs=DEFAULT_NUM_EPOCHS,
    ),
    use_agc=True,
)

class MyGELU(DetachableModule):

    def forward(self, x):
        gate = 0.5 * (1 + torch.erf(x/np.sqrt(2)))
        if self.detach:
            gate = gate.detach()
        return gate * x

# helper
def update_default(new_config):
    return update_config(DEFAULTS, new_config)


def is_big_model(model_name: str) -> bool:
    return "_l_" in model_name or "simple_vit_b" in model_name


SIMPLE_VIT_ARCHS = [
    "simple_vit_ti_patch16_224",
    "simple_vit_s_patch16_224",
    "simple_vit_b_patch16_224",
    "simple_vit_l_patch16_224",
    "vitc_s_patch1_14",
    "vitc_ti_patch1_14",
    "vitc_b_patch1_14",
    "vitc_l_patch1_14",
]

bcos = {
    f"bcos_{name}": update_default(
        dict(
            data=dict(
                batch_size=DEFAULT_BATCH_SIZE
                if not is_big_model(name)
                else DEFAULT_BATCH_SIZE // 2,
                train_transform=ImageNetClassificationPresetTrain(
                    crop_size=DEFAULT_CROP_SIZE,
                    auto_augment_policy="ra",
                    ra_magnitude=10,
                    is_bcos=True,
                ),
                test_transform=ImageNetClassificationPresetEval(
                    crop_size=DEFAULT_CROP_SIZE,
                    is_bcos=True,
                ),
                num_workers=10,
            ),
            model=dict(
                is_bcos=True,
                name=name,
                args=dict(
                    # linear_layer and conv2d_layer set by model.py
                    norm_layer=norms.NoBias(norms.DetachableLayerNorm),
                    act_layer=nn.Identity,
                    channels=6,
                    norm2d_layer=norms.NoBias(DetachableGNLayerNorm2d),
                ),
                bcos_args=dict(
                    b=2,
                    max_out=1,
                ),
                logit_bias=math.log(1 / (NUM_CLASSES - 1)),
            ),
            criterion=UniformOffLabelsBCEWithLogitsLoss(),
            lr_scheduler=DEFAULT_LR_SCHEDULE
            if not is_big_model(name)
            else LONG_WARM_SCHEDULE,
            test_criterion=BinaryCrossEntropyLoss(),
            optimizer=OptimizerFactory(
                "Adam",
                lr=DEFAULT_LR,
            ),
        )
    )
    for name in SIMPLE_VIT_ARCHS
}

# Following config needs to be edited such way so that the B-cos model is loaded with pre-trained weights weights
bcosify = {
    f"bcosifyv2_{name}"+\
    f"{underscore+weight if weight=='random' else ''}"+\
    f"{underscore+str(lr) if lr==1e-3 or lr==1e-2 or lr==1e-5 else ''}"+\
    f"{underscore+lrwarmup if lrwarmup=='lrWarmup' else ''}"+\
    f"{underscore+useBias if useBias=='useBias' else ''}"+\
    f"{underscore+gelu if gelu=='noGelu' else ''}"+\
    f"{underscore+gapReorder if gapReorder=='gapReorder' else ''}": update_config(
        old_config,
        dict(
            model=dict(
                weights=f"pretrained" if weight == "pretrained" else None,
                args=dict(
                    gap_reorder=True if gapReorder=='gapReorder' else False,
                    # act_layer=nn.Identity if gelu=='identity' else MyGELU,
                ),  
                bcosify_args=dict(
                    fix_b = True,
                    use_bias = True if useBias=='useBias' else False,
                    ),
                logit_layer = True,
                act_layer=True if gelu=='gelu' else False,
            ),
            # Skipping warmup as Conv case
            lr_scheduler=LRSchedulerFactory(
                name="cosineannealinglr",
                epochs=DEFAULT_NUM_EPOCHS,
                warmup_method="linear" if lrwarmup=='lrWarmup' else "constant",
                warmup_steps=10_000 if (lrwarmup=='lrWarmup' and not is_big_model(name)) else 50_000 if (lrwarmup=='lrWarmup' and is_big_model(name)) else None,
                interval="step",
                warmup_decay=0.01,
            ),
            # Using LR as 1e-4 as Conv case
            optimizer=OptimizerFactory(
                "Adam",
                lr=0.0001 if lr==1e-4 else 0.001 if lr==1e-3 else 0.00001 if lr==1e-5 else 0.01,
            ),
        )
    )
    for name, old_config in bcos.items()
    for underscore in ['_']
    for weight in ['pretrained','random']
    for lrwarmup in ['lrWarmup','noLrWarmup']
    for lr in [1e-2, 1e-3, 1e-4, 1e-5]
    for gelu in ['gelu', 'noGelu']
    for useBias in ['useBias', 'noBias']
    for gapReorder in ['gapReorder','noGapReorder']
}
# -------------------------------------------------------------------------

CONFIGS = dict()
CONFIGS.update(bcos)
CONFIGS.update(bcosify)
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[5, 420, 1337]))

if __name__ == "__main__":
    configs_cli(CONFIGS)
