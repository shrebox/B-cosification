import copy
import io
import math
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import torch
import torchvision.transforms.functional as transformsF
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

from bcos.common import gradient_to_image
import bcos.data.transforms as custom_transforms


def to_numpy(tensor: "Union[torch.Tensor, np.ndarray]") -> "np.ndarray":
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()


# class that creates a cpu copy of the model
class ModelCopy(torch.nn.Module):
    def __init__(self, model: "torch.nn.Module", use_cpu: bool = True):
        super().__init__()
        self.device = (
            torch.device("cpu") if use_cpu else next(model.parameters()).device
        )
        self.model = copy.deepcopy(model).to(self.device)
        self.original_model = model  # ref to original model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def explanation_mode(self, *args, **kwargs):
        return self.model.explanation_mode(*args, **kwargs)

    def update(self):
        """Update the copy with the original model's parameters."""
        with torch.no_grad():
            # get original model's state dict and put tensors on given device
            state_dict = {
                k: v.to(self.device, non_blocking=True, copy=True)
                for k, v in self.original_model.state_dict().items()
            }
            self.model.load_state_dict(state_dict)

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False


# explanation logger
class ExplanationsLogger(pl_callbacks.Callback):
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        idx: Optional[torch.Tensor] = None,
        max_imgs: int = 32,
        clip_kd: bool = False,
    ):
        self.log_every_n_epochs = log_every_n_epochs
        self.max_imgs = max_imgs
        self.idx = idx

        self.imgs = None
        self.lbls = None

        self.has_bcos_model = False

        self.model: "Optional[ModelCopy]" = None

        self.clip_kd = clip_kd

    @rank_zero_only
    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        if stage != "fit":
            return

        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            dataset = trainer.datamodule.eval_dataset
        else:
            dataset = trainer.val_dataloaders[0].dataset

        # if not bcos specified in pl_module
        # *and* if it neither inherits from BcosModelBase; do not do expl logging
        if not getattr(pl_module, "is_bcos", False):
            rank_zero_warn(
                "Explanation logger is active but PLModule doesn't specify a B-cos model! "
                "Explanation logging will be skipped."
            )
            return

        # we have a bcos model so we now we can safely do explanation log
        self.has_bcos_model = True

        max_imgs = self.max_imgs
        idx = self.idx
        if idx is None:
            g = torch.Generator().manual_seed(42)
            idx = torch.randint(high=len(dataset), size=(max_imgs,), generator=g)
        self.idx = idx[:max_imgs]

        imgs = []
        lbls = []
        for i in idx:
            img, lbl = dataset[i]
            imgs.append(img)
            lbls.append(lbl)
        self.imgs = torch.stack(imgs)
        self.lbls = torch.tensor(lbls)

        if self.has_bcos_model and not hasattr(pl_module.model, "explanation_mode"):
            raise RuntimeError(
                "PLModule specified a bcos model. "
                "However, model doesn't contain .explanation_mode()!"
            )
        self.model = ModelCopy(pl_module.model, use_cpu=True)

    @rank_zero_only
    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self.has_bcos_model:
            return  # no point in logging

        if trainer.sanity_checking:
            return

        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # not initialized yet
        if self.model is None or self.imgs is None or self.lbls is None:
            return

        wandb_logger = None
        tensorboard_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                wandb_logger = logger
            elif isinstance(logger, pl_loggers.TensorBoardLogger):
                tensorboard_logger = logger

        self.log_explanations(wandb_logger, tensorboard_logger)
        # my_numers = self.evaluate_explanations()
        # trainer.my_loc_metric1(my_numbers)
    def log_explanations(
        self,
        wandb_logger: "Optional[pl_loggers.WandbLogger]" = None,
        tensorboard_logger: "Optional[pl_loggers.TensorBoardLogger]" = None,
        postfix="",
    ):
        if wandb_logger is None and tensorboard_logger is None:
            return

        results = self.get_results()
        explanations = self.plot_explanations(results)
        expl_pil_img = self.fig_to_pil(explanations)

        namespace = f"explanations{postfix}"
        if wandb_logger:
            wandb_logger.log_image(f"Explanations/{namespace}", [expl_pil_img])

        if tensorboard_logger:
            writer = tensorboard_logger.experiment
            if hasattr(writer, "add_figure"):
                writer.add_figure(f"{namespace}/Explanations", explanations)
            else:
                writer.add_image(
                    f"{namespace}/Explanations", transformsF.to_tensor(expl_pil_img)
                )

        # figures are retained, hence manual clean up
        plt.close(explanations)

    def get_results(self):
        # prepare things
        imgs = self.imgs
        lbls = self.lbls
        if self.clip_kd:
            transform_bcos = custom_transforms.AddInverse()
            imgs = transform_bcos(imgs)
        assert self.model is not None
        self.model.update()
        self.model.eval()
        self.model.freeze()  # just to be sure
        device = self.model.device

        # result dict
        results = []  # (img, lbl, pred, w)
        with torch.enable_grad(), self.model.explanation_mode():
            # calculate expls
            for img, lbl in zip(imgs, lbls):
                assert not img.requires_grad and img.grad is None
                # prep data
                img = img[None].to(device).requires_grad_()
                lbl = lbl.item()

                # get predictions
                out = self.model(img)
                pred = out.max(1)
                pred.values.backward(inputs=[img])
                pred = pred.indices.item()

                # dynamic weights
                grad = img.grad

                # add to results
                results.append(
                    (
                        img.detach().cpu()[0],
                        lbl,
                        pred,
                        grad.detach().cpu()[0],
                    )
                )

        return results

    def plot_explanations(self, results):
        N = len(results) * 3  # Adjust for three rows per result
        H, W = self.get_grid_size(N)  # Ensure this function returns correct grid dimensions

        # fig = plt.figure(figsize=(W*2, H*2), dpi=200) # For plotting bigger figures but takes more space; can always plot larger figures
        fig = plt.figure(figsize=(W, H), dpi=200)
        grid = ImageGrid(
            fig,
            111,  # similar to subplot(111)
            nrows_ncols=(H, W),  # Ensure this matches the expected grid size
            axes_pad=0.0,
        )

        for i, (img, lbl, pred, weight) in enumerate(results):
            x = i // W
            y = i - W * x
            img_ax = grid[3 * x * W + y]
            expl_ax = grid[(3 * x + 1) * W + y]
            contrib_ax = grid[(3 * x + 2) * W + y]  # New axis for contribution map

            expl, contribs = gradient_to_image(img, weight, smooth=0, return_contribs=True)  # Get contribution map
            contribs = contribs.squeeze()

            # Determine the range of values for the colormap
            cutoff = np.percentile(np.abs(contribs), 99.5)
            contribs = np.clip(contribs, -cutoff, cutoff)
            vrange = np.max(np.abs(contribs.flatten()))

            # Create a custom colormap that maps zero contributions to white
            cmap = plt.cm.bwr
            # cmap.set_under('white', 1.0)
            # cmap.set_over('white', 1.0)

            img_ax.imshow(to_numpy(img[:3].permute(1, 2, 0)))
            expl_ax.imshow(to_numpy(expl))
            contrib_ax.imshow(contribs, cmap=cmap, vmin=-vrange, vmax=vrange)

            # Style for img_ax, expl_ax, and contrib_ax
            for ax in (img_ax, expl_ax, contrib_ax):
                ax.set_xticks([])
                ax.set_yticks([])

            # Style for expl_ax and contrib_ax
            for ax in (expl_ax, contrib_ax):
                for spine in ax.spines.values():
                    spine.set_edgecolor("lime" if lbl == pred else "orange")
                    # spine.set_linewidth(3)  # Set the width of the spine
                    spine.set_linestyle((0, (5, 3)) if lbl == pred else (0, (2, 3)))

        return fig

    @staticmethod
    def get_grid_size(n, ratio=2 / 3):
        # Adjust n to account for three rows per result
        adjusted_n = math.ceil(n / 3)

        # Compute grid dimensions
        x = int(math.sqrt(ratio * adjusted_n))
        if x % 2 == 1:
            x += 1
        y = int(math.ceil(adjusted_n / x))

        # Ensure enough space for all three components of each result
        if x * y < adjusted_n:
            y += 1

        # Since each result requires 3 rows, we multiply x by 3
        return 3 * x, y

    @staticmethod
    def fig_to_pil(fig):
        # from https://stackoverflow.com/a/61754995/10614892
        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        return img

    @staticmethod
    def convert_image(pil_img):
        # TODO: if needed change
        return pil_img
