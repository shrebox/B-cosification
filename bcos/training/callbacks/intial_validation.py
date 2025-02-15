import pytorch_lightning.callbacks as pl_callbacks

class InitialValidationCallback(pl_callbacks.Callback):
    def on_train_start(self, trainer, pl_module):
        # Run a validation loop at the start of training
        trainer.validate(pl_module, dataloaders=trainer.val_dataloaders)
