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
    <a href="https://github.com/shrebox/B-cosification/">Code</a> |
    <a href="https://neurips.cc/media/PosterPDFs/NeurIPS%202024/95051.png?t=1733720266.7038476">Poster</a> |
    <a href="https://nips.cc/media/neurips-2024/Slides/95051.pdf">Slides</a> |
    Video (<a href="https://youtu.be/yvRXuysa5GI">5-mins</a>,
    <a href="https://youtu.be/zSEv3KBGlJQ">neptune.ai</a>,
    <a href="https://youtu.be/ETfYZrSBzVQ">Cohere</a>)
  </h3>
  
</div>


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

to the paths of the data directories.

For ImageNet, the `IMAGENET_PATH` environment variable should point to the directory containing the `train` and `val` directories.


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
You can use evaluate the accuracy of the models on the ImageNet validation set using:
```bash
python evaluate.py \
--dataset ImageNet \
--base_network bcosification \
--experiment_name resnet_18 \
--reload last
```
* `base_network`: `bcosification` for CNNs and `vit_bcosification` for ViTs.
* `experiment_name`: Check the list of experiments below.
* To evaluate the pre-trained B-cosified ImageNet models, please follow the instructions given below in the "Checkpoints" section.

#### List of experiments:

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
* CLIP: Updating...

P.S. For more detailed training instructions, please also have a look at [TRAINING.md](https://github.com/B-cos/B-cos-v2/blob/main/TRAINING.md) from original B-cos-v2 repository.

### Checkpoints

The checkpoints for the B-cosified ImageNet pre-trained models are available [here](https://github.com/shrebox/B-cosification/releases/tag/v0.0.1-checkpoints). 

* The checkpoints should be renamed to `last.ckpt`.
* The checkpoints should be placed under the path: `./experiments/{dataset}/{base_network}/{experiment_name}/{model_name}/last.ckpt`. 

## Acknowledgements

This repository uses code from the following repositories:

* [B-cos/B-cos-v2](https://github.com/B-cos/B-cos-v2)
* [openai/CLIP](https://github.com/openai/CLIP)
* [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

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
