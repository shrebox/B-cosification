import math  # noqa

from bcos.data.presets import (
    CLIPBcosImageNetClassificationPresetEval,
    CLIPBcosImageNetClassificationPresetTrain,
)
from bcos.experiments.utils import (
    configs_cli,
    create_configs_with_different_seeds,
    update_config,
)
from bcos.modules import norms
from bcos.modules.losses import SigLipLoss
from bcos.optim import LRSchedulerFactory, OptimizerFactory

__all__ = ["CONFIGS"]

NUM_CLASSES = 1_000
NUM_TRAIN_EXAMPLES: int = 1_281_167
NUM_EVAL_EXAMPLES: int = 50_000

# These are mainly based on the recipes from
# https://github.com/pytorch/vision/blob/93723b481d1f6e/references/classification/README.md
DEFAULT_BATCH_SIZE = 64  # per GPU! * 4 = 256 effective
DEFAULT_NUM_EPOCHS = 90 
DEFAULT_LR =1e-4
DEFAULT_CROP_SIZE = 224

DEFAULT_NORM_LAYER = norms.NoBias(norms.BatchNormUncentered2d)  # bnu-linear
DEFAULT_OPTIMIZER = OptimizerFactory(name="Adam", lr=DEFAULT_LR, bcosify=True, b_opt = False)
DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
)

DEFAULTS = dict(
    data=dict(
        train_transform=CLIPBcosImageNetClassificationPresetTrain(
            crop_size=DEFAULT_CROP_SIZE,
        ),
        test_transform=CLIPBcosImageNetClassificationPresetEval(
            crop_size=DEFAULT_CROP_SIZE,
        ),
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=16,
        num_classes=NUM_CLASSES,
    ),
    model=dict(
        is_bcos=True,
        # "name": None,
        args=dict(
            num_classes=NUM_CLASSES,
            norm_layer=DEFAULT_NORM_LAYER,
            logit_bias=-math.log(NUM_CLASSES - 1),
        ),
        bcos_args=dict(
            b=2,
            max_out=1,
        ),
    ),
    criterion=SigLipLoss(),
    test_criterion=SigLipLoss(),
    optimizer=DEFAULT_OPTIMIZER,
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(
        max_epochs=DEFAULT_NUM_EPOCHS,
    ),
    use_agc=True,
)

# helper
def update_default(new_config, **kwargs):
    # kwargs is a dict
    return update_config(DEFAULTS, new_config)

RESNET_DEPTHS = [50]
resnets_clip = {
    f"resnet_{depth}_clip"+\
    f"_b2_noBias"+\
    f"_randomResizedCrop"+\
    f"{underscore+schDLR if schDLR=='cyclicLR' else ''}"+\
    f"_sigLip_ImageNet_bcosification": update_default(
        dict(
            clip_kd = True, # This is accesed in config (not model_config)
            data=dict(
                train_transform=CLIPBcosImageNetClassificationPresetTrain(crop_size=DEFAULT_CROP_SIZE),
                test_transform=CLIPBcosImageNetClassificationPresetEval(crop_size=DEFAULT_CROP_SIZE),  
            ),
            model=dict(
                name=f"resnet{depth}clip", 
                bcosify_args = dict(
                    clip_kd = True, # Enable CLIP related functions
                    fix_b = True, # Fix b to 2
                    norm_layer = "BnUncV2", # Modified BatchNorm
                    schDLR = schDLR, # Learning rate scheduler
                    use_bias = False, # No bias
                ),
            ),
        )
    )
    for depth in RESNET_DEPTHS
    for schDLR in ['cosineAnnealingLR', 'cyclicLR']
    for underscore in ['_']  
}
# -------------------------------------------------------------------------

CONFIGS = dict()
CONFIGS.update(resnets_clip)
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[420, 1337]))

if __name__ == "__main__":
    configs_cli(CONFIGS)
