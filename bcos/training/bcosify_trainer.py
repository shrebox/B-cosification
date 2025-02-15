import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from bcos.training.trainer import ClassificationLitModel

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x: x  # noqa: E731
from bcos.settings import IMAGENET_PATH

# Define the path to the ImageNet validation data folder
val_data_folder = IMAGENET_PATH+'/val'

def create_test_loader(transform, val_data_folder=val_data_folder):

    # Create the ImageFolder dataset for validation data
    val_dataset = ImageFolder(val_data_folder, transform=transform)

    # Create the validation data loader
    batch_size = 64  # Adjust according to your needs
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return test_loader

def evaluate(self, device, model, data_loader):
    # https://github.com/pytorch/vision/blob/657c0767c5ca5564c8b437ac44263994c8e0/references/classification/train.py#L61
    model.eval()

    total_samples = 0
    total_correct_top1 = 0
    total_correct_top5 = 0
    with torch.inference_mode():
        for image, target in tqdm(data_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(image)

            total_samples += image.shape[0]
            correct_top1, correct_top5 = check_correct(output, target, topk=(1, 5))
            total_correct_top1 += correct_top1.item()
            total_correct_top5 += correct_top5.item()

    acc1 = total_correct_top1 / total_samples
    acc5 = total_correct_top5 / total_samples
    print(
        f"Out of a total of {total_samples}, got {total_correct_top1=} and {total_correct_top5=}"
    )
    print()
    print("--------------------------------------------")
    print(f"Acc@1 {acc1:.3%} Acc@5 {acc5:.3%}")
    print("--------------------------------------------")
    print()
    model.train()
    return acc1, acc5


def check_correct(output, target, topk=(1,)):
    with torch.inference_mode():
        maxk = max(topk)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum()
            res.append(correct_k)
        return res
    
class BcosifyTrainer(ClassificationLitModel):

    def __init__(self, dataset, base_network, experiment_name):
        super().__init__(dataset, base_network, experiment_name, auto_optimization=False)

    def configure_optimizers(self):        
        optimizer = self.config["optimizer"].create(self.model)
        scheduler = self.config["lr_scheduler"].create(
            optimizer,
            # this is total as in "whole" training
            total_steps=self.trainer.estimated_stepping_batches,
        )
        if self.config["model"]["bcosify_args"].get("linear_b", None):
            num_gpus = self.trainer.num_devices
            print(f"Number of GPUs: {num_gpus}")
            
            b_lr = self.config["model"]["bcosify_args"].get("b_lr", 0.0001) 
            print(f"Old b_lr: {b_lr}")

            b_lr = self.config["model"]["bcosify_args"].get("b_lr", 0.0001) * num_gpus
            print(f"Updated b_lr: {b_lr}")

            b_optimizer = torch.optim.SGD((p for n, p in self.model.named_parameters() if n.endswith(".b")), lr=b_lr)

        if self.config["model"]["bcosify_args"].get("b_loss", None):
            b_weight_decay = self.config["model"]["b_optim_args"]["weight_decay"]
            b_momentum = self.config["model"]["b_optim_args"]["momentum"]
            b_lr = self.config["model"]["b_optim_args"]["lr"]
            b_optimizer = torch.optim.SGD((p for n, p in self.model.named_parameters() if n.endswith(".b")), lr=b_lr, momentum=b_momentum, weight_decay=b_weight_decay)

        return ({"optimizer":optimizer, "lr_scheduler":scheduler,},\
                {"optimizer":b_optimizer})
    
    def training_step(self, batch, batch_idx):
        ## https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        opt, opt_b = self.optimizers()

        images, labels = batch
        outputs = self(images)

        loss = self.criterion(outputs, labels)

        opt.zero_grad()
        opt_b.zero_grad()
        self.manual_backward(loss)
        opt.step()
        opt_b.step()

        sch = self.lr_schedulers()
        sch.step()

        # step every N batches
        if (batch_idx + 1) % 1000 == 0:
            print(f"Batch {batch_idx}")
            for name, param in self.model.named_parameters():
                if name.endswith('.b'):
                    print(f"Name: {name}, Parameter: {param.item()}")
                    self.log(f"b_{name}", param.item())

        with torch.no_grad():
            if labels.ndim == 2:
                # b/c of mixup/cutmix or sparse labels in general. See
                # https://github.com/pytorch/vision/blob/9851a69f6d294f5d672d973/references/classification/utils.py#L179
                labels = labels.argmax(dim=1)
            self.train_acc1(outputs, labels)
            self.train_acc5(outputs, labels)

            self.log("train_loss", loss)
            self.log(
                "train_acc1",
                self.train_acc1,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "train_acc5",
                self.train_acc5,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

            if self.ema is not None and batch_idx % self.ema_steps == 0:
                ema = self.ema
                ema.update_parameters(self.model)
                if self.trainer.current_epoch < self.lr_warmup_epochs:
                    ema.n_averaged.fill_(0)

        return loss
