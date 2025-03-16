import os
from pathlib import Path
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

try:
    import rich  # noqa: F401

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# START: >>>>>>>>>>>> Evaluation at 0th epoch <<<<<<<<<<<<<<<
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import bcos.data.transforms as custom_transforms
import bcos.training.callbacks as custom_callbacks
from bcos.experiments.utils import Experiment, sanitize_config
from bcos.settings import DATA_ROOT, IMAGENET_PATH, IMAGENET_RN50_ZEROSHOT_WEIGHTS_PATH
from bcos.training.agc import adaptive_clip_grad_
from bcos.training.ema import ExponentialMovingAverage
from bcos.training.hooks import Hook, forward_hook_fn

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x: x  # noqa: E731

import clip

# Define the path to the ImageNet validation data folder
val_data_folder = IMAGENET_PATH+'/val'

def get_imagenet_zeroshot_weights(model_name):
    return torch.load(IMAGENET_RN50_ZEROSHOT_WEIGHTS_PATH)

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

    with torch.no_grad():
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

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def clip_accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def clip_evaluate(self, model, loader, zeroshot_weights):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()
            
            # predict
            image_features = model(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = 100. * image_features @ zeroshot_weights
            
            attn_unpool = self.config['model'].get("attn_unpool", False)
            if attn_unpool:
                cos_power = self.config['model'].get("cos_power",1)
                logits = (logits) * (logits.abs().detach()**(cos_power-1))
                logits = logits.sum(0)

            # measure accuracy
            acc1, acc5 = clip_accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) 
    top5 = (top5 / n) 

    return top1, top5

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

def zeroshot_classifier(clip_model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = clip_model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def clip_zeroshot_evaluate(self, model):

    # Loading clip model
    if 'vit' in self.model_name.lower():
        clip_model, _ = clip.load(self.model_name)
    else:
        clip_model, _ = clip.load("RN50") 

    text_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
    batch_size = 64
    common_trans = [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                custom_transforms.AddInverse(),
            ]

    # -------------------- CIFAR-10 --------------------
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"{len(cifar10_classes)} classes, {len(text_templates)} templates")

    # Get the zeroshot weights
    zeroshot_weights = zeroshot_classifier(clip_model, cifar10_classes, text_templates).float()

    cifar10_testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False,
                                       download=True, transform=transforms.Compose(common_trans))
    cifar10_loader = torch.utils.data.DataLoader(cifar10_testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    top1, top5 = clip_evaluate(self, model, cifar10_loader, zeroshot_weights)

    print("CIFAR-10 accuracies:")
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
    print('--------------------------------------------\n')

    self.log("cifar10_zeroshot_acc1", top1, rank_zero_only=True, sync_dist=True)
    self.log("cifar10_zershot_acc5", top5, rank_zero_only=True, sync_dist=True)

    # -------------------- CIFAR-100 --------------------
    cifar100_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    print(f"{len(cifar100_classes)} classes, {len(text_templates)} templates")

    # Get the zeroshot weights
    zeroshot_weights = zeroshot_classifier(clip_model, cifar100_classes, text_templates).float()

    cifar100_testset = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False,
                                       download=True, transform=transforms.Compose(common_trans))
    cifar100_loader = torch.utils.data.DataLoader(cifar100_testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    top1, top5 = clip_evaluate(self, model, cifar100_loader, zeroshot_weights)

    print("CIFAR-100 accuracies:")
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
    print('--------------------------------------------\n')

    self.log("cifar100_zeroshot_acc1", top1, rank_zero_only=True, sync_dist=True)
    self.log("cifar100_zershot_acc5", top5, rank_zero_only=True, sync_dist=True)

    # -------------------- FashionMNIST --------------------
    FashionMNIST_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(f"{len(FashionMNIST_classes)} classes, {len(text_templates)} templates")

    # Get the zeroshot weights
    zeroshot_weights = zeroshot_classifier(clip_model, FashionMNIST_classes, text_templates).float()

    FashionMNIST_testset = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=False,
                                       download=True, transform=transforms.Compose(common_trans))
    FashionMNIST_loader = torch.utils.data.DataLoader(FashionMNIST_testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    top1, top5 = clip_evaluate(self, model, FashionMNIST_loader, zeroshot_weights)

    print("FashionMNIST accuracies:")
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
    print('--------------------------------------------\n')

    self.log("FashionMNIST_zeroshot_acc1", top1, rank_zero_only=True, sync_dist=True)
    self.log("FashionMNIST_zershot_acc5", top5, rank_zero_only=True, sync_dist=True)

    # -------------------- STL-10 --------------------
    stl10_classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

    print(f"{len(stl10_classes)} classes, {len(text_templates)} templates")

    # Get the zeroshot weights
    zeroshot_weights = zeroshot_classifier(clip_model, stl10_classes, text_templates).float()

    stl10_testset = torchvision.datasets.STL10(root=DATA_ROOT, split='test',
                                       download=True, transform=transforms.Compose(common_trans))
    stl10_loader = torch.utils.data.DataLoader(stl10_testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    top1, top5 = clip_evaluate(self, model, stl10_loader, zeroshot_weights)

    print("STL-10 accuracies:")
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
    print('--------------------------------------------\n')

    self.log("stl10_zeroshot_acc1", top1, rank_zero_only=True, sync_dist=True)
    self.log("stl10_zershot_acc5", top5, rank_zero_only=True, sync_dist=True)
    
    return

# END: >>>>>>>>>>>> Evaluation at 0th epochs <<<<<<<<<<<<<<<

def calculate_metrics(param):
    metrics = {
        'mae': torch.mean(torch.abs(param)),
        # 'rms': torch.sqrt(torch.mean(param ** 2)),
        # 'l1': torch.norm(param, p=1),
        # 'l2': torch.norm(param, p=2),
        # 'inf': torch.norm(param, p=float('inf'))
    }
    return metrics

class ClassificationLitModel(pl.LightningModule):
    def __init__(self, dataset, base_network, experiment_name, auto_optimization=True):
        super().__init__()

        self.automatic_optimization=auto_optimization

        self.experiment = Experiment(dataset, base_network, experiment_name)
        config = self.experiment.config
        model = self.experiment.get_model()

        self.save_hyperparameters()  # passed arguments
        self.save_hyperparameters(sanitize_config(config))  # the config as well

        self.experiment_name = experiment_name
        self.config = config
        self.model = model
        self.is_bcos = self.config["model"].get("is_bcos", False)
        self.criterion = self.config["criterion"]
        self.test_criterion = self.config["test_criterion"]
        self.clip_kd = self.config.get("clip_kd", False)
        self.model_name = self.config["model"]["name"]

        print("Experiment config name: ", experiment_name)
        for key, value in self.config.items():
            print(f"{key}: {value}")

        logit_scale = config['model'].get('logit_scale', None)
        logit_bias = config['model'].get('logit_bias', None)

        ## -------- Attention Unpooling Fine Tuning ---------
        # For loading the model without attnUnpool, load the weights in unpool model and fine tune it
        if config['model'].get('pool_mode', None) == 'FineTune':

            ## ------ prepare the experiment name for loading the pooled model weights ------
            cos_power = config['model']['cos_power']
            if cos_power != 1:
                replace_str = 'attnUnpool'+str(config['model']['cos_power'])+'FineTune'
            else:
                replace_str = 'attnUnpoolFineTune'
            # Logit bias
            if logit_bias is None:
                replace_str = replace_str + 'NoLogitBias'
            # Logit scale
            if logit_scale is None:
                replace_str = replace_str + 'NoLogitScale'
            replace_str = replace_str + '_'
            ## ------ prepare the experiment name for loading the pooled model weights ------
            
            print(f"replace_str: {replace_str}")
            print(f"experiment_name: {experiment_name}")
            load_exp_name = experiment_name.replace(replace_str, '')
            exp = Experiment(dataset, base_network, load_exp_name)
            model_data = exp.load_trained_model(
                        return_training_ckpt_if_possible=True,
                    )
            model_pooled = model_data['model']
            model.load_state_dict(model_pooled.state_dict(), strict=False)
            epoch = model_data['ckpt']['epoch']
            print(f"Loaded the pooled model weights for fine tuning from {load_exp_name} at epoch {epoch}")
        
        self.loss_type = str(type(self.test_criterion)) 
        if "SigLip" in self.loss_type:
            if logit_bias is not None:
                fixed_logit_bias = config['model'].get("fixed_logit_bias", True)
                logit_bias_value = config['model'].get("logit_bias_value", -10)
                if not fixed_logit_bias:
                    model.logit_bias = nn.Parameter(torch.ones([]) * logit_bias_value)
            if logit_scale is not None:
                fixed_logit_scale = config['model'].get("fixed_logit_scale", True)
                logit_scale_value = config['model'].get("logit_scale_value", 10)
                if not fixed_logit_scale:
                    model.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale_value))
        elif "distillClip" in self.loss_type or "clipLoss" in self.loss_type:
            if logit_scale is not None:
                fixed_logit_scale = config['model'].get("fixed_logit_scale", True)
                if not fixed_logit_scale:
                    model.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # B parameter (make just learnable or linearly interpolate using Hook)
        bcosify_args = self.config["model"].get("bcosify_args", None)
        if bcosify_args is not None and bcosify_args.get("fix_b", False) == False: 
            print("Making b as parameter")
            for mod in model.modules():
                
                # For adding batch_size to each module
                # Used in linear interpolation of b parameter
                mod.register_forward_hook(forward_hook_fn) 
                
                if hasattr(mod, "b"):
                    # This changes the b parameter in the Bcos layers from being a float to a nn.Parameter
                    # Note that the addition of 1e-6 is done to avoid the if self.b==1: return out 
                    # behaviour in the Bcos layers. This should probably be done differently, via a 
                    # b_as_parameter keyword in the layer definition.
                    b_at_start = bcosify_args.get("b_at_start", 1)
                    mod.b=nn.Parameter(torch.tensor(b_at_start + 1e-6).cuda(), requires_grad=True)
                    
                    if bcosify_args.get("fix_b", False):
                        mod.b.requires_grad = False
                    
                    # Adding the hook to manipulate the gradients
                    if bcosify_args.get("linear_b", False):
                        start_b_value = bcosify_args.get("b_at_start", 1)
                        end_b_value = bcosify_args.get("b_at_end", 2)
                        mod.b.register_hook(Hook(mod, start=start_b_value, end=end_b_value))
                
            print("Registered hook for b parameter in trainer.py")

        self.linearprobe_clip = bcosify_args.get('linearprobe_clip', False) if bcosify_args is not None else False

        clip_kd = self.config.get("clip_kd", False) 
        if clip_kd:
            # Loading standard_clip teacher model
            arch_name = self.config["model"]["name"]
            if "vit" in arch_name.lower():
                clip_model, _ = clip.load(arch_name)
            else:
                clip_model, _ = clip.load("RN50")
            self.clip_model = clip_model.visual
            self.clip_model.float()
            self.clip_model.eval()
            
            # Bcosifyed clip model
            self.model = model
            
            # Making data loaders for clip and bcosifyed model
            # Need to the common transforms again as the validation set is loaded from from scratch
            # Common transformations
            common_trans = [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                _convert_image_to_rgb,
                transforms.ToTensor(),
            ]

            # Specific transformations for clip and bcosifyed model
            clip_trans = common_trans + [transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]
            clip_bcos = common_trans + [custom_transforms.AddInverse()]

            self.clip_loader = create_test_loader(transform=transforms.Compose(clip_trans))
            self.bcos_loader = create_test_loader(transform=transforms.Compose(clip_bcos))                              

        num_classes = config["data"]["num_classes"]
        if not clip_kd:
            self.train_acc1 = torchmetrics.Accuracy(
                task="multiclass", top_k=1, num_classes=num_classes, compute_on_cpu=True
            )
            self.train_acc5 = torchmetrics.Accuracy(
                task="multiclass", top_k=5, num_classes=num_classes, compute_on_cpu=True
            )
            self.eval_acc1 = torchmetrics.Accuracy(
                task="multiclass", top_k=1, num_classes=num_classes, compute_on_cpu=True
            )
            self.eval_acc5 = torchmetrics.Accuracy(
                task="multiclass", top_k=5, num_classes=num_classes, compute_on_cpu=True
            )

            has_ema = self.config.get("ema", None) is not None
            self.eval_acc1_ema = self.eval_acc1.clone() if has_ema else None
            self.eval_acc5_ema = self.eval_acc5.clone() if has_ema else None

        self.ema = None  # will be set during setup(stage="fit")
        self.ema_steps = None

        self.lr_warmup_epochs = self.config["lr_scheduler"].warmup_epochs

        self.use_agc = self.config.get("use_agc", False)
        if self.use_agc:
            rank_zero_info("Adaptive Gradient Clipping is enabled!")

    def setup(self, stage: str) -> None:
        if stage != "fit":
            return

        ema_config = self.config.get("ema", None)
        if ema_config is None:
            return

        decay = ema_config["decay"]
        self.ema_steps = ema_config.get("steps", 32)
        rank_zero_info(f"Using EMA with {decay=} and steps={self.ema_steps}")

        
        # see https://github.com/pytorch/vision/blob/657c0767c5ca5564c8b437ac442/references/classification/train.py#L317
        adjust = (
            self.trainer.world_size
            * self.trainer.accumulate_grad_batches
            * self.config["data"]["batch_size"]
            * self.ema_steps
            / self.trainer.max_epochs
        )
        alpha = 1.0 - decay
        alpha = min(1.0, alpha * adjust)
        self.ema = ExponentialMovingAverage(self.model, decay=1.0 - alpha)
        self.ema.requires_grad_(False)

    def configure_optimizers(self):
        
        # Check if 'bcosify_args' is available and not None
        bcosify_args = self.config.get('model', {}).get('bcosify_args', None)
        
        # Retrieve configuration values
        # Bias
        use_bias = bcosify_args.get('use_bias', False) if bcosify_args is not None else False
        decay_bias = bcosify_args.get("decay_bias", False) if bcosify_args is not None else False
        bias_decay_value = decay_bias.get('weight_decay', 0) if decay_bias and use_bias else 0
        # B
        linear_b = bcosify_args.get("linear_b", False) if bcosify_args is not None else False
        learn_b = bcosify_args.get("learn_b", False) if bcosify_args is not None else False
        decay_b = bcosify_args.get("decay_b", False) if bcosify_args is not None else False
        b_decay_value = decay_b.get('weight_decay', 0) if decay_b and learn_b else 0

        # Initialize parameter groups
        parameters_with_bias = []
        parameters_with_b = []
        other_parameters = []
        params_with_bias_names = []

        # Categorize parameters
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                parameters_with_bias.append(param)
                params_with_bias_names.append(name)
            elif name.endswith('.b'):
                parameters_with_b.append(param)
            else:
                other_parameters.append(param)
        print(params_with_bias_names)
            
        # Define parameter groups for the optimizer based on the enabled settings
        param_groups = []
        if decay_bias and decay_b:
            param_groups = [
                {'params': parameters_with_bias, 'weight_decay': bias_decay_value},
                {'params': parameters_with_b, 'weight_decay': b_decay_value},
                {'params': other_parameters, 'weight_decay': 0}
            ]
        elif decay_bias:
            param_groups = [
                {'params': parameters_with_bias, 'weight_decay': bias_decay_value},
                {'params': other_parameters + parameters_with_b, 'weight_decay': 0}
            ]
        elif decay_b:
            param_groups = [
                {'params': parameters_with_b, 'weight_decay': b_decay_value},
                {'params': other_parameters + parameters_with_bias, 'weight_decay': 0}
            ]
        else:
            param_groups = [{'params': self.model.parameters(), 'weight_decay': 0}]
        
        # Create the optimizer with the defined parameter groups
        norm_type = bcosify_args.get("norm_layer", "Bn2d") if bcosify_args is not None else "Bn2d"
        clip_kd = self.config.get("clip_kd", False)
        if norm_type == 'Bn2d' or clip_kd:
            optimizer = self.config["optimizer"].create(self.model)
        else:
            optimizer = optim.AdamW(param_groups, lr=self.config['optimizer'].args['lr']) 

        # -------- Print parameter group information for debugging ---------
        print("Number of param groups: ", len(optimizer.param_groups))
        print("Number of bias parameters: ", len(parameters_with_bias))
        print("Number of .b parameters: ", len(parameters_with_b))
        print("Number of other parameters: ", len(other_parameters))
        print(optimizer)
        # Print the number of parameter groups
        print("Number of parameter groups:", len(optimizer.param_groups))
        # Print the number of parameters in each group
        for i, group in enumerate(optimizer.param_groups):
            print(f"Group {i+1} has {len(group['params'])} parameters")
        # -------- Print parameter group information for debugging ---------

        # Setup learning rate scheduler
        schD = bcosify_args.get("schDLR", "cosineannealinglr") if bcosify_args is not None else "cosineannealinglr"
        if schD == 'cyclicLR':
            print("Using COSINEANNEALINGWARMRESTARTS")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        else:
            scheduler = self.config["lr_scheduler"].create(
                optimizer,
                total_steps=self.trainer.estimated_stepping_batches,
            )
        print(scheduler)
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def forward(self, in_tensor):
        clip_kd = self.config.get("clip_kd", False)
        if clip_kd:
            transform_clip = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            transform_bcos = custom_transforms.AddInverse()
            out_clip = self.clip_model(transform_clip(in_tensor))
            out_bcos = self.model(transform_bcos(in_tensor))
            attn_unpool = self.config['model'].get("attn_unpool", False)
            loss_type = str(type(self.test_criterion)) 
            if attn_unpool and ("SigLip" not in loss_type):
                return out_clip, out_bcos.sum(0)
            return out_clip, out_bcos
        return self.model(in_tensor)

    def training_step(self, batch, batch_idx):
        clip_kd = self.config.get("clip_kd", False)
        images, labels = batch
        if clip_kd:
            output_clip, output_bcos = self(images)

            # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L267
            output_bcos = F.normalize(output_bcos, dim=-1)
            output_clip = F.normalize(output_clip, dim=-1)
            
            if "distillClipLoss" in self.loss_type or "clipLoss" in self.loss_type:
                # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L229
                logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp() 
                loss = self.test_criterion(output_bcos, output_clip, self.logit_scale.exp())
            elif "SigLipLoss" in self.loss_type:
                """
                Implementation for loss calculation: https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384#with-openclip
                https://github.com/mlfoundations/open_clip/blob/main/src/training/main.py#L220
                https://github.com/mlfoundations/open_clip/issues/746; 
                https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L327
                https://github.com/mlfoundations/open_clip/issues/712, 
                https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/factory.py#L152
                """
                fixed_logit_bias = self.config['model'].get("fixed_logit_bias", True)
                logit_bias_value = self.config['model'].get("logit_bias_value", -10)
                if fixed_logit_bias:
                    local_logit_bias = nn.Parameter(torch.ones([]) * logit_bias_value)
                
                fixed_logit_scale = self.config['model'].get("fixed_logit_scale", True)
                logit_scale_value = self.config['model'].get("logit_scale_value", 10)
                if fixed_logit_scale:
                    local_logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale_value))
                    
                attn_unpool = self.config['model'].get("attn_unpool", False)
                if attn_unpool:
                    cos_power = self.config['model'].get("cos_power", 1)
                else:
                    cos_power = 0
                if fixed_logit_bias and fixed_logit_scale:
                    loss = self.test_criterion(output_bcos, output_clip, local_logit_scale.exp(), local_logit_bias, output_dict=False, cos_scaling=cos_power)
                else:
                    loss = self.test_criterion(output_bcos, output_clip, self.model.logit_scale.exp(), self.model.logit_bias, output_dict=False, cos_scaling=cos_power)
                
                # Log the logit_scale and logit_bias values after every 1000 batches
                if (batch_idx + 1) % 1000 == 0:
                    print(f"Batch {batch_idx}")
                    if fixed_logit_bias and fixed_logit_scale:
                        print(f"logit_scale: {local_logit_scale.exp()}, logit_bias: {local_logit_bias}")
                        self.log("logit_scale", local_logit_scale.exp())
                        self.log("logit_bias", local_logit_bias)
                    else:
                        print(f"logit_scale: {self.model.logit_scale.exp()}, logit_bias: {self.model.logit_bias}")
                        self.log("logit_scale", self.model.logit_scale.exp())
                        self.log("logit_bias", self.model.logit_bias)
            else:
                loss = self.test_criterion(output_clip, output_bcos)
        else:
            outputs = self(images)
            loss = self.criterion(outputs, labels)

        # For learning b
        bcosify_args = self.config["model"].get("bcosify_args", None)
        if bcosify_args is not None and bcosify_args.get("fix_b", False) == False and (bcosify_args.get('learn_b', False) or bcosify_args.get('linear_b', False)):
            # step every N batches
            if (batch_idx + 1) % 1000 == 0:
                print(f"Batch {batch_idx}")
                for name, param in self.model.named_parameters():
                    if name.endswith('.b'):
                        print(f"Name: {name}, Parameter: {param.item()}")
                        self.log(f"b_{name}", param.item())

        # For bias decay
        decay_bias = self.config.get("decay_bias", False)
        if decay_bias:
            # step every N batches
            if (batch_idx + 1) % 200 == 0:
                print(f"Batch {batch_idx}")
                for name, param in self.model.named_parameters():
                    if 'bias' in name:
                        metrics = calculate_metrics(param) # Function defined above after evaluate
                        print(f"Name: {name}")
                        for metric_name, value in metrics.items():
                            print(f"{metric_name.upper()}: {value}")
                            self.log(f"{metric_name}_{name}", value)

        if not clip_kd: # Only log metrics for the clip mode
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
        else:
            with torch.no_grad():
                self.log("train_loss", loss)
        return loss

    def eval_step(self, batch, _batch_idx, val_or_test):
        
        images, labels = batch
        clip_kd = self.config.get("clip_kd", False)
        if clip_kd:
            output_clip, output_bcos = self(images)
            
            # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L267
            output_bcos = F.normalize(output_bcos, dim=-1)
            output_clip = F.normalize(output_clip, dim=-1)
            
            if "distillClipLoss" in self.loss_type  or "clipLoss" in self.loss_type:
                # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L229
                logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp() 
                loss = self.test_criterion(output_bcos, output_clip, self.logit_scale.exp())
            elif "SigLipLoss" in self.loss_type:
                """
                Implementation for loss calculation: https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384#with-openclip
                https://github.com/mlfoundations/open_clip/blob/main/src/training/main.py#L220
                https://github.com/mlfoundations/open_clip/issues/746; 
                https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L327
                https://github.com/mlfoundations/open_clip/issues/712, 
                https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/factory.py#L152
                """
                fixed_logit_bias = self.config['model'].get("fixed_logit_bias", True)
                logit_bias_value = self.config['model'].get("logit_bias_value", -10)
                if fixed_logit_bias:
                    local_logit_bias = nn.Parameter(torch.ones([]) * logit_bias_value)
                
                fixed_logit_scale = self.config['model'].get("fixed_logit_scale", True)
                logit_scale_value = self.config['model'].get("logit_scale_value", 10)
                if fixed_logit_scale:
                    local_logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale_value))
                    
                attn_unpool = self.config['model'].get("attn_unpool", False)
                if attn_unpool:
                    cos_power = self.config['model'].get("cos_power", 1)
                else:
                    cos_power = 0
                if fixed_logit_bias and fixed_logit_scale:
                    loss = self.test_criterion(output_bcos, output_clip, local_logit_scale.exp(), local_logit_bias, output_dict=False, cos_scaling=cos_power)
                else:
                    loss = self.test_criterion(output_bcos, output_clip, self.model.logit_scale.exp(), self.model.logit_bias, output_dict=False, cos_scaling=cos_power)
            else:
                loss = self.test_criterion(output_clip, output_bcos)

            self.log(f"{val_or_test}_loss", loss, sync_dist=True)

        else:
            outputs = self(images)
            loss = self.test_criterion(outputs, labels)
            self.eval_acc1(outputs, labels)
            self.eval_acc5(outputs, labels)

        if not clip_kd: 
            self.log(f"{val_or_test}_loss", loss)
            self.log(f"{val_or_test}_acc1", self.eval_acc1, on_epoch=True, prog_bar=True)
            self.log(f"{val_or_test}_acc5", self.eval_acc5, on_epoch=True, prog_bar=True)
        

        if self.ema is not None:
            ema = self.ema
            if clip_kd:
                transform_clip = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                transform_bcos = custom_transforms.AddInverse()
                output_clip = self.clip_model(transform_clip(images))
                output_bcos = ema.module(transform_bcos(images))
                ema_loss = self.test_criterion(output_clip, output_bcos)
            else:
                outputs = ema.module(images)
                ema_loss = self.test_criterion(outputs, labels)
                self.eval_acc1_ema(outputs, labels)
                self.eval_acc5_ema(outputs, labels)
                self.log(f"{val_or_test}_acc1_ema", self.eval_acc1_ema, on_epoch=True)
                self.log(f"{val_or_test}_acc5_ema", self.eval_acc5_ema, on_epoch=True)
            self.log(f"{val_or_test}_loss_ema", loss, sync_dist=True)
            return {"loss": loss, "loss_ema": ema_loss}
        else:
            return loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        # Note: this is called even if gradient_clip_val etc. is None
        if not self.use_agc:
            self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)
        else:
            adaptive_clip_grad_(self.parameters())

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        # we only care about total grad norm
        norm_type = float(self.trainer.track_grad_norm)
        total_norm = grad_norm_dict[f"grad_{norm_type}_norm_total"]
        del grad_norm_dict
        self.log(
            "gradients/total_norm",
            total_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.trainer.gradient_clip_val is not None:
            clipped_total_norm = min(
                float(self.trainer.gradient_clip_val), float(total_norm)
            )
            self.log(
                "gradients/clipped_total_norm",
                clipped_total_norm,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

def print_parameter_value(model_param, batch_idx):
    print(f"Batch {batch_idx}, Parameter Value: {model_param.item()}")

def put_trainer_args_into_trainer_config(args, trainer_config):
    if args.distributed:
        # https://github.com/Lightning-AI/lightning/discussions/6761#discussioncomment-2614296
        trainer_config["strategy"] = "ddp_find_unused_parameters_false"

    if args.fast_dev_run:
        trainer_config["fast_dev_run"] = True

    if hasattr(args, "nodes"):  # on slurm
        trainer_config["num_nodes"] = args.nodes

    if args.track_grad_norm:
        trainer_config["track_grad_norm"] = 2.0

    if hasattr(args, "amp") and args.amp:
        trainer_config["precision"] = 16

    if args.debug:
        trainer_config["deterministic"] = True

def setup_loggers(args):
    loggers = []
    save_dir = Path(
        args.base_directory, args.dataset, args.base_network, args.experiment_name
    )

    if args.wandb_logger:
        wandb_logger = pl_loggers.WandbLogger(
            name=args.wandb_name or args.experiment_name,
            save_dir=str(save_dir),
            project=args.wandb_project,
            id=args.wandb_id,
        )
        loggers.append(wandb_logger)

    if args.csv_logger:
        csv_logger = pl_loggers.CSVLogger(
            save_dir=str(save_dir / "csv_logs"),
            name="",
            flush_logs_every_n_steps=1000,
        )
        loggers.append(csv_logger)

    if args.tensorboard_logger:
        tensorboard_logger = pl_loggers.TensorBoardLogger(
            save_dir=Path(
                "tb_logs",
                args.base_directory,
                args.dataset,
                args.base_network,
                args.experiment_name,
            ),
            name=args.experiment_name,
        )
        loggers.append(tensorboard_logger)

    return loggers


def setup_callbacks(args, config):
    callbacks = []
    save_dir = Path(
        args.base_directory, args.dataset, args.base_network, args.experiment_name
    )
    clip_kd = config.get("clip_kd", False)
    if not clip_kd:
        # the most important one
        save_callback = pl_callbacks.ModelCheckpoint(
            dirpath=save_dir,
            monitor="val_acc1",
            mode="max",
            filename="{epoch}-{val_acc1:.4f}",
            save_last=True,
            save_top_k=3,
            verbose=True,
        )
        callbacks.append(save_callback)

        use_ema = config.get("ema", None) is not None
        if use_ema:
            save_callback = pl_callbacks.ModelCheckpoint(
                dirpath=save_dir,
                monitor="val_acc1_ema",
                mode="max",
                filename="{epoch}-{val_acc1_ema:.4f}",
                save_top_k=3,
                verbose=True,
            )
            callbacks.append(save_callback)
    
    if clip_kd:
        # the most important one
        save_callback = pl_callbacks.ModelCheckpoint(
            dirpath=save_dir,
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{val_loss:.4f}",
            save_last=True,
            save_top_k=3,
            verbose=True,
            # save_on_train_epoch_end=True,
        )
        callbacks.append(save_callback)

    # lr monitor
    has_logger = args.wandb_logger or args.tensorboard_logger or args.csv_logger
    if has_logger:  # ow it's useless
        callbacks.append(pl_callbacks.LearningRateMonitor())
    slurm_or_submitit = hasattr(args, "nodes") or "SLURM_JOB_ID" in os.environ
    refresh_rate = args.refresh_rate or (20 if slurm_or_submitit else 5)
    if HAS_RICH and not slurm_or_submitit:
        callbacks.append(pl_callbacks.RichProgressBar(refresh_rate=refresh_rate))
    else:
        callbacks.append(pl_callbacks.TQDMProgressBar(refresh_rate=refresh_rate))

    # save metrics to checkpoint
    callbacks.append(custom_callbacks.MetricsTracker())

    # do explanation logging
    if args.explanation_logging:
        log_every = args.explanation_logging_every_n_epochs
        rank_zero_info(f"Will log explanations every {log_every} epoch(s)!")
        callbacks.append(
            custom_callbacks.ExplanationsLogger(log_every_n_epochs=log_every, clip_kd=clip_kd)
        )
    else:
        rank_zero_info("Explanation logging is disabled!")

    # for debugging purposes
    if args.debug:
        callbacks.append(custom_callbacks.ModelUpdateHasher())

    # callbacks.append(custom_callbacks.InitialValidationCallback())
    callbacks.append(TeacherAlwaysEvalMode())
    callbacks.append(FreezeTeacher())
    callbacks.append(ImageNetEval())
    callbacks.append(ZeroshotEval())
    print('new callbacks')

    return callbacks

class TeacherAlwaysEvalMode(pl_callbacks.Callback):
    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if pl_module.clip_kd:
            pl_module.clip_model.eval()

# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BaseFinetuning.html
class FreezeTeacher(pl_callbacks.BaseFinetuning):

    def freeze_before_training(self, pl_module):
        if pl_module.clip_kd:
            self.freeze(pl_module.clip_model)

    def finetune_function(self, pl_module, current_epoch, optimizer, opt_idx):
        pass # Just HAD to be implemented :)

class ZeroshotEval(pl_callbacks.Callback):
    @rank_zero_only
    def on_validation_epoch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        print('Inside ZeroshotEval on_validation_epoch_end callback')
        # Clip_kd
        if pl_module.clip_kd:
            clip_zeroshot_evaluate(pl_module, pl_module.model)
            zeroshot_weights = get_imagenet_zeroshot_weights(pl_module.model_name)
            test_acc1_bcos, test_acc5_bcos = clip_evaluate(pl_module, pl_module.model, pl_module.bcos_loader, zeroshot_weights)
            print(f"Bcos acc1: {test_acc1_bcos}, Bcos acc5: {test_acc5_bcos}")    
            self.log("test_acc1_bcos", test_acc1_bcos, rank_zero_only=True, sync_dist=True)
            self.log("test_acc5_bcos", test_acc5_bcos, rank_zero_only=True, sync_dist=True)
        return
        

class ImageNetEval(pl_callbacks.Callback):
    @rank_zero_only
    def on_train_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        print('on_train_start callback')
        if not pl_module.clip_kd and trainer.current_epoch==0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_loader_standard = trainer.val_dataloaders[0]

            # Evaluate the model on the validation data
            print("Evaluating the Bcosifyed model on the validation data")
            acc1, acc5 = evaluate(pl_module, device, pl_module.model.to(device), test_loader_standard)
        return