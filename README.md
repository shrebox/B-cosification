<div align="center">
  
  <h1>B-cosification: Transforming Deep Neural Networks to be Inherently Interpretable</h1>
  
  <p>
    <a href="https://www.linkedin.com/in/shrebox/">Shreyash Arya*</a>,
    <a href="https://sukrutrao.github.io">Sukrut Rao*</a>,
    <a href="https://moboehle.github.io">Moritz Böhle*</a>,
    <a href="https://people.mpi-inf.mpg.de/~schiele">Bernt Schiele</a>
  </p>
  
  <h3>Neural Information Processing Systems (NeurIPS) 2024</h3>
    
  <h3>
    <a href="https://arxiv.org/abs/2411.00715">Paper</a> |
    <a href="https://openreview.net/forum?id=TA5zPfH8iI">OpenReview</a> |
    <a href="https://github.com/shrebox/B-cosification/">Code</a> |
    <a href="https://neurips.cc/media/PosterPDFs/NeurIPS%202024/95051.png?t=1733720266.7038476">Poster</a> |
    <a href="https://nips.cc/media/neurips-2024/Slides/95051.pdf">Slides</a> |
    Video (<a href="https://youtu.be/yvRXuysa5GI">5-mins</a>,
    <a href="https://youtu.be/zSEv3KBGlJQ">neptune.ai</a>,
    <a href="https://youtu.be/ETfYZrSBzVQ">Cohere</a>)
  </h3>
  
</div>

![teaser_bcosification](https://github.com/user-attachments/assets/b557591e-1625-4002-becb-cc177e9d2ef8)

## Installation

### Training Environment Setup
If you want to train your own B-cosified models using this repository or are interested in reproducing the results, you can set up the development environment as follows:

Using `conda`:
```bash
conda env create -f environment.yml
conda activate bcosification
```

Using `pip`:
```bash
conda create --name bcosification python=3.12
pip install -r requirements.txt
```

#### Setting Data Paths
You can either set the paths in [`bcos/settings.py`](bcos/settings.py) or set the environment variables
1. `DATA_ROOT`
2. `IMAGENET_PATH`
3. `CC3M_PATH`
4. `IMAGENET_RN50_ZEROSHOT_WEIGHTS_PATH` (for zeroshot evaluation of CLIP models)

to the paths of the data directories.

* For ImageNet, the `IMAGENET_PATH` environment variable should point to the directory containing the `train` and `val` directories.

* For CC3M, the `CC3M_PATH` environment variable should point to the directory containing the `training` and `validation` directories with `*.tar`, `*_stats.json`, and `*.parquet` files. For more instructions, please check [here](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md).

* For zeroshot evaluation of CLIP models during training, the `IMAGENET_RN50_ZEROSHOT_WEIGHTS_PATH` environment variable should point to the weights provided in the [release](https://github.com/shrebox/B-cosification/releases/tag/v0.0.2-CLIP-checkpoints).


<!-- =============================================================================================================== -->

## Usage

For evaluating or training the models, you can use the `evaluate.py` and `train.py` scripts, as follows:

### Training

For single-GPU training:
```bash
python train.py \
    --dataset ImageNet \
    --base_network bcosification \
    --experiment_name resnet_18 
```

For distributed training:
```bash
python run_with_submitit.py \
    --dataset ImageNet \
    --base_network vit_bcosification \
    --experiment_name bcosifyv2_bcos_simple_vit_ti_patch16_224_0.001_gapReorder-seed=5 \
    --distributed \
    --gpus 4 \
    --node 1 \
    --timeout 8 \
    --wandb_logger \
    --wandb_project bcosification \
    --explanation_logging
```

### Evaluation
You can use evaluate the <ins>accuracy</ins> of the models on the ImageNet validation set using:
```bash
python evaluate.py \
    --dataset ImageNet \
    --base_network bcosification \
    --experiment_name resnet_18 \
    --reload last
```
* `base_network`: `bcosification` for CNNs, or `vit_bcosification` for ViTs.
* `experiment_name`: check the [List of experiments](https://github.com/shrebox/B-cosification/tree/main?tab=readme-ov-file#list-of-experiments) section below.
* To evaluate the pre-trained B-cosified ImageNet models, please follow the instructions given below in the [Checkpoints](https://github.com/shrebox/B-cosification/tree/main?tab=readme-ov-file#checkpoints) section.

For evaluating <ins>B-cosified CLIP</ins> models, use [CLIP Benchmark](https://github.com/shrebox/B-cosification/blob/main/CLIP_benchmark) as follows:

Zeroshot:
```bash
 python CLIP_benchmark/clip_benchmark/cli.py eval \
    --dataset=wds/imagenet1k \
    --model_type=bcos_clip \
    --output=benchmark_{dataset}_{model}_{task}.json \
    --dataset_root=https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main \
    --model=resnet_50_clip_b2_noBias_randomResizedCrop_sigLip_ImageNet_bcosification \
    --pretrained=experiments/ImageNet/clip_bcosification
```

Linear Probe:
```bash
python CLIP_benchmark/clip_benchmark/cli.py eval \
    --task=linear_probe \
    --dataset=wds/imagenet1k \
    --model_type=bcos_clip \
    --output=benchmark_{dataset}_{model}_{task}.json \
    --dataset_root=https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main \
    --model=resnet_50_clip_b2_noBias_randomResizedCrop_sigLip_ImageNet_bcosification \
    --pretrained=experiments/ImageNet/clip_bcosification/ \
    --batch_size=512 \
    --fewshot_lr 0.1 \
    --fewshot_epochs 20 \
    --train_split train \
    --test_split test
```

* `--dataset`: use `wds/{dataset}` for available [Zeroshot](https://github.com/shrebox/B-cosification/blob/main/CLIP_benchmark/benchmark/datasets.txt) and [Linear Probe](https://github.com/shrebox/B-cosification/blob/main/CLIP_benchmark/benchmark/lp_webdatasets.txt) datasets.
* `--model_type`: `bcos_clip`, `bcos_clip_cc3m`, `text2concept_clip`, `standard_clip`.
* `--model`: for model types `bcos_clip` and `bcos_clip_cc3m`, check the [List of experiments](https://github.com/shrebox/B-cosification/tree/main?tab=readme-ov-file#list-of-experiments) section for model names; for `standard_clip` use `RN50`; for `text2concept_clip` not required.

<ins>Note</ins>: For CLIP models, automatic <ins>zeroshot evaluation</ins> is done at the start of every epoch.

#### Localisation

For <ins>localisation analysis</ins> of a trained model, [localisation.py](https://github.com/shrebox/B-cosification/blob/main/interpretability/analyses/localisation.py) can be used as follows:

```bash
python -m interpretability.analyses.localisation \
    --reload last \
    --analysis_config 500_3x3 \
    --explainer_name Ours \
    --smooth 15 \
    --batch_size 64 \
    --save_path "experiments/ImageNet/bcosification/resnet_18/"
```
* For ViTs, `--analysis_config 500_2x2-stride=112` and `--striding 112` are required.
* The results along with localisation plots are stored in the `localisation_analysis` directory automatically created in the experiments directory (`--save_path`).

For <ins>text-localisation using B-cosified CLIP model</ins>, [text_localisation.py](https://github.com/shrebox/B-cosification/blob/main/interpretability/analyses/text_localisation.py) can be used as follows:

on an ImageNet image:
```bash
python -m interpretability.analyses.text_localisation \
    --exp_name experiments/ImageNet/clip_bcosification/resnet_50_clip_b2_noBias_randomResizedCrop_sigLip_ImageNet_bcosification \
    --image_index 2 \
    --text_to_localize "green,blue,orange" \
    --save_path /path/to/save
```

* `--use_class_name` to localise the class name for a given ImageNet image.
* `--save_path` is by default set to path provided by `--exp_name` if not set.


on a random image:
```bash
python -m interpretability.analyses.text_localisation \
    --exp_name experiments/ImageNet/clip_bcosification/resnet_50_clip_b2_noBias_randomResizedCrop_sigLip_ImageNet_bcosification \
    --random_img_path /path/to/image \
    --text_to_localize "green,blue,orange"
    --save_path /path/to/save 
```

### List of experiments:

* CNNs: `resnet18`, `resnet_50`, `resnet_50_V1`, `densenet_121`
* ViTs: `bcosifyv2_{model_name}_0.001_lrWarmup_gapReorder`
```
{model_name}
    "vitc_ti_patch1_14",
    "vitc_s_patch1_14",
    "vitc_b_patch1_14",
    "vitc_l_patch1_14",
    "simple_vit_ti_patch16_224",
    "simple_vit_s_patch16_224",
    "simple_vit_b_patch16_224",
    "simple_vit_l_patch16_224"

Note: Only b and l models use lrWarmup in the final models.
```
* CLIP: `resnet_50_clip_b2_noBias_randomResizedCrop_sigLip_{dataset}_bcosification`; where `{dataset}` is either `ImageNet` or `CC3M`. Also, the `base_network` for CLIP models is `clip_bcosification`.


P.S. For more detailed training instructions, please also have a look at [TRAINING.md](https://github.com/B-cos/B-cos-v2/blob/main/TRAINING.md) from original B-cos-v2 repository.

### Checkpoints

The checkpoints for the B-cosified ImageNet CNN and ViT pre-trained models are available [here](https://github.com/shrebox/B-cosification/releases/tag/v0.0.1-checkpoints). For B-cosified CLIP pre-trained models, please check [here](https://github.com/shrebox/B-cosification/releases/tag/v0.0.2-CLIP-checkpoints).

* The checkpoints should be renamed to `last.ckpt`.
* The checkpoints should be placed under the path: `./experiments/{dataset}/{base_network}/{experiment_name}/{model_name}/last.ckpt`. 

## Acknowledgements

This repository uses code from the following repositories:

* [B-cos/B-cos-v2](https://github.com/B-cos/B-cos-v2)
* [openai/CLIP](https://github.com/openai/CLIP)
* [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
* [LAION-AI/CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark)
* [k1rezaei/Text-to-concept](https://github.com/k1rezaei/Text-to-concept/tree/main)

## License
This repository's code is licensed under the Apache 2.0 license which you can find in the [LICENSE](./LICENSE) file.

The pre-trained models are trained on ImageNet (and are hence derived from it), which is  licensed under the [ImageNet Terms of access](https://image-net.org/download), which among others things, only allows non-commercial use of the dataset. It is therefore your responsibility to check whether you have permission to use the  pre-trained models for *your* use case.

## Citation

Please cite as follows:

```tex
@inproceedings{arya2024bcosification,
 author = {Arya, Shreyash and Rao, Sukrut and B\"{o}hle, Moritz and Schiele, Bernt},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {62756--62786},
 publisher = {Curran Associates, Inc.},
 title = {B-cosification: Transforming Deep Neural Networks to be Inherently Interpretable},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/72d50a87b218d84c175d16f4557f7e12-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
