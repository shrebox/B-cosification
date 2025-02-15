import os
import time
from typing import List

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_info
except ImportError:
    raise ImportError(
        "Please install pytorch-lightning for using data modules: "
        "`pip install pytorch-lightning`"
    )

import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import CIFAR10, ImageFolder
from PIL import Image
import random

import bcos.settings as settings

from .categories import CIFAR10_CATEGORIES, IMAGENET_CATEGORIES
from .sampler import RASampler
from .transforms import RandomCutmix, RandomMixup, SplitAndGrid

from .cc3m import (CC3MImg, CC3MText, CustomDataCollatorImg,
                  CustomDataCollatorText)

__all__ = ["ImageNetDataModule", "CIFAR10DataModule", "ClassificationDataModule", "VOCDataModule", "CC3MDataModule"]

def get_random_cut(dataset, cut_ratio):
    all_indices = [*range(0, len(dataset))]
    cut_idx = int(cut_ratio* len(all_indices))
    state = random.getstate()
    random.seed(42)
    random.shuffle(all_indices) #Always shuffle the same way.
    random.setstate(state)
    return all_indices[:cut_idx], all_indices[cut_idx:]

class ClassificationDataModule(pl.LightningDataModule):
    """Base class for data modules for classification tasks."""

    NUM_CLASSES: int = None
    """Number of classes in the dataset."""
    NUM_TRAIN_EXAMPLES: int = None
    """Number of training examples in the dataset. Need not be defined."""
    NUM_EVAL_EXAMPLES: int = None
    """Number of evaluation examples in the dataset. Need not be defined."""
    CATEGORIES: List[str] = None
    """List of categories in the dataset. Need not be defined."""

    # ===================================== [ Registry stuff ] ======================================
    __data_module_registry = {}
    """Registry of data modules."""

    def __init_subclass__(cls, **kwargs):
        # check that the class attributes are defined
        super().__init_subclass__(**kwargs)
        assert cls.NUM_CLASSES is not None
        # rest don't need to be defined

        # get name and remove DataModule suffix
        name = cls.__name__
        # check if name matches XXXDataModule
        if not name.endswith("DataModule"):
            raise ValueError(
                f"Data module class name '{name}' does not end with 'DataModule'"
            )
        name = name[: -len("DataModule")]
        # check if name is already registered
        if name in cls.__data_module_registry:
            raise ValueError(f"Data module {name} already registered")
        # register the class in the registry
        cls.__data_module_registry[name] = cls

    @classmethod
    def registry(cls):
        """Returns the registry of data modules."""
        return cls.__data_module_registry

    # ===================================== [ Normal stuff ] ======================================
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

        self.train_dataset = None
        self.eval_dataset = None

        mixup_alpha = config.get("mixup_alpha", 0.0)
        cutmix_alpha = config.get("cutmix_alpha", 0.0)
        p_gridified = config.get("p_gridified", 0.0)
        self.train_collate_fn = self.get_train_collate_fn(
            mixup_alpha, cutmix_alpha, p_gridified
        )

    def train_dataloader(self):
        train_sampler = self.get_train_sampler()
        shuffle = None if train_sampler is not None else True
        return data.DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @classmethod
    def get_train_collate_fn(
        cls,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        p_gridified: float = 0.0,
    ):
        assert not (p_gridified and mixup_alpha), "For now, do not use both."

        collate_fn = None
        if p_gridified:
            gridify = SplitAndGrid(p_gridified, num_classes=cls.NUM_CLASSES)

            def collate_fn(batch):
                return gridify(*data.default_collate(batch))

            rank_zero_info(f"Gridify active for training with {p_gridified=}")

        mixup_transforms = []
        if mixup_alpha > 0.0:
            mixup_transforms.append(
                RandomMixup(cls.NUM_CLASSES, p=1.0, alpha=mixup_alpha)
            )
            rank_zero_info(f"Mixup active for training with {mixup_alpha=}")
        if cutmix_alpha > 0.0:
            mixup_transforms.append(
                RandomCutmix(cls.NUM_CLASSES, p=1.0, alpha=cutmix_alpha)
            )
            rank_zero_info(f"Cutmix active for training with {cutmix_alpha=}")
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch):  # noqa: F811
                return mixupcutmix(*data.default_collate(batch))

        return collate_fn

    def get_train_sampler(self):
        train_sampler = None

        # see https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/pytorch_lightning/trainer/connectors/data_connector.py#L336
        # and https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/lightning_lite/utilities/seed.py#L54
        seed = int(os.getenv("PL_GLOBAL_SEED", 0))
        ra_reps = self.config.get("ra_repetitions", None)
        if ra_reps is not None:
            rank_zero_info(f"Activating RASampler with {ra_reps=}")
            train_sampler = RASampler(
                self.train_dataset,
                shuffle=True,
                seed=seed,
                repetitions=ra_reps,
            )

        return train_sampler


class ImageNetDataModule(ClassificationDataModule):
    # from https://image-net.org/download.php
    NUM_CLASSES: int = 1000

    NUM_TRAIN_EXAMPLES: int = 1_281_167
    NUM_EVAL_EXAMPLES: int = 50_000

    CATEGORIES: List[str] = IMAGENET_CATEGORIES

    def __init__(self, config):
        super().__init__(config)
        self.prepare_data_per_node = self.config.get("cache_dataset", None) == "shm"

    def prepare_data(self) -> None:
        cache_dataset = self.config.get("cache_dataset", None)
        if cache_dataset != "shm":
            return

        # print because we also want global non-zero rank's
        start = time.perf_counter()
        print("Caching dataset into SHM!...")
        from .caching import cache_tar_files_to_shm

        cache_tar_files_to_shm()
        end = time.perf_counter()
        print(f"Caching successful! Time taken {end - start:.2f}s")

    def setup(self, stage: str) -> None:
        # this way changes to the settings are reflected at function call time
        SHMTMPDIR = settings.SHMTMPDIR
        IMAGENET_PATH = settings.IMAGENET_PATH
        if stage == "fit":
            cache_dataset = self.config.get("cache_dataset", None)
            rank_zero_info("Setting up ImageNet train dataset...")
            start = time.perf_counter()
            train_root = os.path.join(
                SHMTMPDIR if cache_dataset == "shm" else IMAGENET_PATH,
                "train",
            )
            self.train_dataset = ImageFolder(
                root=train_root,
                transform=self.config["train_transform"],
            )
            assert len(self.train_dataset) == self.NUM_TRAIN_EXAMPLES
            rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")

            if cache_dataset == "onthefly":
                rank_zero_info("Trying to setup Bagua's cached dataset!")
                from .caching import CachedImageFolder

                self.train_dataset = CachedImageFolder(self.train_dataset)
                rank_zero_info("Successfully setup cached dataset!")

        start = time.perf_counter()
        rank_zero_info("Setting up ImageNet val dataset...")
        self.eval_dataset = ImageFolder(
            root=os.path.join(IMAGENET_PATH, "val"),
            transform=self.config["test_transform"],
        )
        assert len(self.eval_dataset) == self.NUM_EVAL_EXAMPLES
        rank_zero_info(f"Done! Took time {time.perf_counter() - start:.2f}s")


class CIFAR10DataModule(ClassificationDataModule):
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    NUM_CLASSES: int = 10

    NUM_TRAIN_EXAMPLES: int = 50_000
    NUM_EVAL_EXAMPLES: int = 10_000

    CATEGORIES: List[str] = CIFAR10_CATEGORIES

    def setup(self, stage: str) -> None:
        DATA_ROOT = settings.DATA_ROOT
        if stage == "fit":
            self.train_dataset = CIFAR10(
                root=DATA_ROOT,
                train=True,
                transform=self.config["train_transform"],
                download=True,
            )
            assert len(self.train_dataset) == self.NUM_TRAIN_EXAMPLES

        self.eval_dataset = CIFAR10(
            root=DATA_ROOT,
            train=False,
            transform=self.config["test_transform"],
            download=True,
        )
        assert len(self.eval_dataset) == self.NUM_EVAL_EXAMPLES

class VOCDataModule(ClassificationDataModule):
    NUM_CLASSES: int = 20

    def setup(self, stage: str) -> None:
        DATA_ROOT = settings.VOC_PATH
        if stage == "fit":
            if self.config.get('train_split_portion', None) is not None:
                    entire_train_data = VOCDataset(
                        root=DATA_ROOT,
                        image_set='train',
                        download=False,
                        # transform=self.config["train_transform"], # Don't pass it here!
                        year='2012',
                        preload = self.config['preload'],
                        also_annotation=self.config['also_annotation'],
                    )
                    train_indices, eval_indices = get_random_cut(entire_train_data, self.config.get('train_split_portion'))
                    self.train_dataset = MySubset(entire_train_data,
                                indices=train_indices,
                                transform=self.config["train_transform"] 
                                )
                    self.eval_dataset = MySubset(entire_train_data,
                                indices=eval_indices,
                                transform=self.config["test_transform"] 
                                )
                    self.train_idx_trans = lambda idx: train_indices[idx]
                    self.eval_idx_trans = lambda idx: eval_indices[idx]
                    rank_zero_info(f'[Fit and Eval Setup] {len(self.train_dataset), len(self.eval_dataset)} for train and eval.')
                    rank_zero_info(f'Eval indices hash is: {hash(tuple(sorted(eval_indices)))}')
                    return
            else:
                self.train_dataset = VOCDataset(
                    root=DATA_ROOT,
                    image_set='train',
                    transform=self.config["train_transform"],
                    download=False,
                    year='2012',
                    preload = self.config['preload'],
                    also_annotation=self.config['also_annotation'],
                )

        if stage in ['fit', 'val'] and self.config.get('train_split_portion', None) is not None:
            rank_zero_info("Not Setting up anything as val split is part of fit and should be done in fit setup!")
            return

        eval_stage = stage
        if stage == 'fit':
            eval_stage = 'val'

        self.eval_dataset = VOCDataset(
            root=DATA_ROOT,
            image_set=eval_stage,
            transform=self.config["test_transform"],
            download=False,
            year='2012',
            preload=self.config['preload'],
            also_annotation=self.config['also_annotation'],
        )

class VOCDataset(torchvision.datasets.VOCDetection):
    def __init__(self, *args, preload=False, also_annotation=False, **kwargs):
        super(VOCDataset, self).__init__(*args, **kwargs)
        self.transforms = None

        self.target_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6,
                'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
        self.reverse_target_dict = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7:
                       'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 
                15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}

        self.num_classes = 20

        if preload:
            self.preload = False
            self.load_data()
        self.preload = preload
        self.also_annotation = also_annotation
        assert self.transforms is None, f'Not considered as of now!'

    def load_data(self):
        rank_zero_info(f"Preloading all the data!")
        transform = self.transform
        target_transform = self.target_transform

        self.cached_images = []
        self.cached_targets = []
        self.transform = None
        self.target_transform = None
        for idx in range(len(self)):
            img, target = self[idx]
            self.cached_images[idx] = img
            self.cached_targets[idx] = target

        self.transform = transform
        self.target_transform = target_transform
        rank_zero_info(f"cached all the data successfully!")
        rank_zero_info(f"Putting back the transforms {self.transform=}, {self.target_transform=}")

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if self.preload:
            img = self.cached_images[index]
            target = self.cached_targets[index]
        else:
            img = Image.open(self.images[index]).convert("RGB")
            annotations = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

            objects = annotations['annotation']['object']
            target = torch.zeros(self.num_classes)
            object_names = [item['name'] for item in objects]
            for name in object_names:
                target[self.target_dict[name]] = 1

        if self.transform is not None:
            img = self.transform(img)
        
        if self.also_annotation:
            size = annotations['annotation']['size']
            width = int(size['width'])
            height = int(size['height'])
            wscale = 224 / width
            hscale = 224 / height

            object_bndboxes = [item['bndbox'] for item in objects]
            bbs = []
            for name, bndbox in zip(object_names, object_bndboxes):
                index = self.target_dict[name]
                xmin, xmax = int(bndbox['xmin']), int(bndbox['xmax'])
                ymin, ymax = int(bndbox['ymin']), int(bndbox['ymax'])

                new_xmin, new_xmax = int(min(max(xmin*wscale, 0), 223)), int(min(max(xmax*wscale, 0), 223))
                new_ymin, new_ymax = int(min(max(ymin*hscale, 0), 223)), int(min(max(ymax*hscale, 0), 223))

                bbs.append([index, new_xmin, new_ymin, new_xmax, new_ymax])
            return img, target, bbs
        else:
            return img, target
        
        # This applies transforms to both img and target (irrelevant for us!)
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

class MySubset(data.Subset):
    """
    Subset dataset with a few more things:
    - supporting a custom transform
    - delegate rest of attr./methods to internal dataset.

    Note: Mainly required for splitting and then using different transforms on
          created `Subset`s. (Otherwise, it's overwritten b/c internal is same.)
    Note: only for supervised data of form (x, y)
    """
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        super().__init__(dataset, indices)
        self.transform = transform
        self.target_transform = target_transform
        if hasattr(dataset, "transform") and dataset.transform is not None:
            rank_zero_info(f"Internal dataset has transform will apply transform on top: {dataset}")

    def __getitem__(self, item):
        x, y = super().__getitem__(item)
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __getattr__(self, item):
        if item in ['transform', 'target_transform']:
            return self.__dict__[item]
        # not found in attr so look in internal dataset
        return getattr(self.dataset, item)

class CC3MDataModule(ClassificationDataModule):
    NUM_CLASSES: int = -1 # How to handle this?

    def setup(self, stage: str) -> None:
        DATA_ROOT = settings.CC3M_PATH
        
        collator = CustomDataCollatorImg()
        cc3m_obj = CC3MImg()

        if stage == "fit":
            split_path = "training"
            tar_name = "{00000..00331}.tar"
            data_shard = os.path.join(DATA_ROOT, split_path, tar_name)
            self.train_dataset = cc3m_obj.get_wds_dataset(
                data_shard, 
                self.config["train_transform"], 
                self.batch_size, 
                collator=collator)

        split_path = "validation"
        tar_name = "{00000..00001}.tar"
        data_shard = os.path.join(DATA_ROOT, split_path, tar_name)
        self.eval_dataset = cc3m_obj.get_wds_dataset(
                data_shard, 
                self.config["test_transform"], 
                self.batch_size, 
                collator=collator)
    
    # # Following loaders are not the default but adapated as per the cc3m.py code from Sukrut
    def train_dataloader(self):
        train_sampler = self.get_train_sampler()
        # shuffle = None if train_sampler is not None else True
        shuffle = False
        return data.DataLoader(
            self.train_dataset,
            None, 
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )