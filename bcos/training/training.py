import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins import environments as pl_env_plugins
from pytorch_lightning.utilities import rank_zero_info

from bcos.experiments.utils import Experiment, CHECKPOINT_LAST_FILENAME
from bcos.training.bcosify_trainer import BcosifyTrainer
from bcos.training.trainer import setup_loggers, ClassificationLitModel, setup_callbacks, \
    put_trainer_args_into_trainer_config

def run_training(args):
    """
    Instantiates everything and runs the training.
    """
    base_directory = args.base_directory
    dataset = args.dataset
    base_network = args.base_network
    experiment_name = args.experiment_name
    save_dir = Path(base_directory, dataset, base_network, experiment_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    # set up loggers early so that WB starts capturing output asap
    loggers = setup_loggers(args)

    # get config
    exp = Experiment(dataset, base_network, experiment_name)
    config = exp.config.copy()

    # get and set seed
    seed = exp.config.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    
    trainer = ClassificationLitModel # For automatic optimization
    bcosify_args = exp.config["model"].get("bcosify_args", None)
    if bcosify_args is not None and bcosify_args.get("manual_optim", False):
        trainer =  BcosifyTrainer

    # init model
    model = trainer(
        dataset,
        base_network,
        experiment_name
    )
    rank_zero_info(f"Model: {repr(model.model)}")

    # jit the internal model if specified
    if args.jit:
        model.model = torch.jit.script(model.model)
        rank_zero_info("Jitted the model!")

    # init datamodule
    datamodule = model.experiment.get_datamodule(
        cache_dataset=getattr(args, "cache_dataset", None),
    )

    # callbacks
    callbacks = setup_callbacks(args, config)

    # init trainer
    trainer_config = config["trainer"]
    put_trainer_args_into_trainer_config(args, trainer_config)

    # plugin for slurm
    if "SLURM_JOB_ID" in os.environ:  # we're on slurm
        # let submitit handle requeuing
        trainer_config["plugins"] = [
            pl_env_plugins.SLURMEnvironment(auto_requeue=False)
        ]

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
        **trainer_config,
    )

    # decide whether to resume
    ckpt_path = None
    if args.resume:
        ckpt_path = save_dir / CHECKPOINT_LAST_FILENAME
        ckpt_path = ckpt_path if ckpt_path.exists() else None

    # start training
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
