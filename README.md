# Occupancy Anticipation
This repository contains a PyTorch implementation of our ECCV-20 paper: 

[Occupancy Anticipation for Efficient Exploration and Navigation](http://vision.cs.utexas.edu/projects/occupancy_anticipation)<br />
Santhosh Kumar Ramakrishnan, Ziad Al-Halah, Kristen Grauman<br />
The University of Texas at Austin, Facebook AI Research

Project website: [http://vision.cs.utexas.edu/projects/occupancy_anticipation](http://vision.cs.utexas.edu/projects/occupancy_anticipation)

This code implements our winning entry to the [Habitat-20 PointNav challenge](https://aihabitat.org/challenge/2020/) which is a well-tuned version with improved ground-truth generation, faster training, and better heuristics for planning. 

## Abstract
State-of-the-art navigation methods leverage a spatial memory to generalize to new environments, but their occupancy maps are limited to capturing the geometric structures directly observed by the agent. We propose *occupancy anticipation*, where the agent uses its egocentric RGB-D observations to infer the occupancy state beyond the visible regions. In doing so, the agent builds its spatial awareness more rapidly, which facilitates efficient exploration and navigation in 3D environments.   By exploiting context in both the egocentric views and top-down maps our model successfully anticipates a broader map of the environment, with performance significantly better than strong baselines. Furthermore, when deployed for the sequential decision-making tasks of exploration and navigation, our model outperforms state-of-the-art methods on the Gibson and Matterport3D datasets. Our approach is the winning entry in the 2020 Habitat PointNav Challenge. 


## Installation

Clone the current repository and required submodules:

```
git clone https://github.com/facebookresearch/OccupancyAnticipation.git
cd OccupancyAnticipation
  
export OCCANT_ROOT_DIR=$PWD
    
git submodule init 
git submodule update
```
    
This project relies on specific commits of 3 different submodules ([habitat-api](https://github.com/facebookresearch/habitat-lab), [habitat-sim](https://github.com/facebookresearch/habitat-sim), [astar_pycpp](https://github.com/srama2512/astar_pycpp)). Ensure that their dependencies are satisfied and then install them as follows:

```	
# Install habitat-api
cd $OCCANT_ROOT_DIR/environments/habitat/habitat-api
python setup.py develop --all
	
# Install habitat-sim
cd $OCCANT_ROOT_DIR/environments/habitat/habitat-sim
python setup.py install --headles --with-cuda
	
# Install astar_pycpp
cd $OCCANT_ROOT_DIR/occant_utils/astar_pycpp
make
```
Install other requirements for this repository.

```
cd $OCCANT_ROOT_DIR
pip install -r requirements.txt
```

Add the OccupancyAnticipation directory to `PYTHONPATH` in `.bashrc`.

```
export PYTHONPATH=<path to OccupancyAnticipation>:$PYTHONPATH
```
Create a symlink to the habitat-api data directory.

```
cd $OCCANT_ROOT_DIR
ls -s <PATH TO habitat-api/data> data
```
Download the exploration dataset.

```
cd $OCCANT_ROOT_DIR/data/datasets
mkdir exploration
cd exploration

# Download gibson task dataset
wget https://dl.fbaipublicfiles.com/OccupancyAnticipation/gibson.zip
unzip gibson.zip && rm gibson.zip

# Download the matterport3d task dataset
wget https://dl.fbaipublicfiles.com/OccupancyAnticipation/mp3d.zip
unzip mp3d.zip && rm mp3d.zip
```



## Generating environment layouts

We generate top-down occupancy maps for:<br />
1. computing the occupancy anticipation ground-truth for training<br />
2. measuring map quality during evaluation. 

This is a one-time process and will not impact training or evaluation.

### Maps for occupancy ground-truth generation
To compute the occupancy ground-truth, we cast rays from the agent's position till it reaches a wall. We take the ground-truth for all locations that the ray traversed, which may include obstacles within the room. This allows us to exclude regions outside the current room which may be hard to predict. 

```
# Gibson splits

# Need to set this to train / val
SPLIT="train"
cd $OCCANT_ROOT_DIR
python occant_baselines/generate_topdown_maps/generate_occant_gt_maps.py \
    --config-path occant_baselines/generate_topdown_maps/configs/occant_gt/gibson_${SPLIT}.yaml \
    --save-dir data/datasets/exploration/gibson/v1/${SPLIT}/occant_gt_maps \
    --global-map-size 961


# Matterport3D splits

# Need to set this to train / val / test
SPLIT="train"
cd $OCCANT_ROOT_DIR
python occant_baselines/generate_topdown_maps/generate_occant_gt_maps.py \
    --config-path occant_baselines/generate_topdown_maps/configs/occant_gt/mp3d_${SPLIT}.yaml \
    --save-dir data/datasets/exploration/mp3d/v1/${SPLIT}/occant_gt_maps \
    --global-map-size 2001
```

Note that generating the maps for train and test splits may take several hours.


### Maps for evaluating map quality
The top-down layout of objects, walls and free-space needs to be generated for each episode in the evaluation data. 

```
# Gibson validation
cd $OCCANT_ROOT_DIR
python occant_baselines/generate_topdown_maps/generate_environment_layouts.py \
    --config-path occant_baselines/generate_topdown_maps/configs/environment_layouts/gibson_val.yaml \
    --save-dir data/datasets/exploration/gibson/v1/val/environment_layouts \
    --global-map-size 961
    
# Matterport3D test
cd $OCCANT_ROOT_DIR
python occant_baselines/generate_topdown_maps/generate_environment_layouts.py \
    --config-path occant_baselines/generate_topdown_maps/configs/environment_layouts/mp3d_test.yaml \
    --save-dir data/datasets/exploration/mp3d/v1/test/environment_layouts \
    --global-map-size 2001
```
Note that the Matterport3D test set generation may take ~ 3 hours.


## Training models
We provide the training code for the following baselines and variants of our models:

| Model         | Config directory                            |
|---------------|---------------------------------------------|
| ANS(rgb)      | `configs/model_configs/ans_rgb`             |
| ANS(depth)    | `configs/model_configs/ans_depth`           |
| OccAnt(rgb)   | `configs/model_configs/occant_rgb`          |
| OccAnt(depth) | `configs/model_configs/occant_depth`        |
| OccAnt(rgbd)  | `configs/model_configs/occant_rgbd`         |
| OccAnt(GT)    | `configs/model_configs/occant_ground_truth` |

To train a model, run the following:

```
cd $OCCANT_ROOT_DIR

python -u run.py --exp-config <CONFIG-DIRECTORY>/ppo_exploration.yaml --run-type train
```

## Evaluating models

The  evaluation configurations for both exploration and navigation under noise_free and noisy conditions are provided for OccAnt(depth) in `configs/. Similar configs can be generated for other models.

To continuously evaluate a series of checkpoints while training, just run the following after setting the right paths:

```
cd $OCCANT_ROOT_DIR

python -u run.py --exp-config <CONFIG-DIRECTORY>/ppo_<TASK>_evaluate_<NOISE_CONDITION>.yaml --run-type eval
```

To evaluate a single checkpoint, change `EVAL_CKPT_PATH_DIR` in the evaluation yaml files to the model path and run the same command.


## Pretrained models

We provide sample pretrained models for all methods and the corresponding configurations. These are trained with the latest version of the code for ~24 hours each. We also provide the benchmarked navigation results on Gibson under noise-free and noisy (N) conditions. 


| Model name      | Training data |       Val SPL      |    Val SPL (N)  | Checkpoint URL | Train config URL |
|-----------------|:-------------:|:------------------:|:---------------:|:--------------:|:----------------:|
| ANS(rgb)        |   Gibson 4+   |      0.825         |      0.385      | [ckpt.12.pth](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/ans_rgb/ckpt.12.pth) | [ppo_exploration.yaml](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/ans_rgb/ppo_exploration.yaml) |
| ANS(depth)      |   Gibson 4+   |      0.894         |      0.466      | [ckpt.14.pth](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/ans_depth/ckpt.14.pth) | [ppo_exploration.yaml](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/ans_depth/ppo_exploration.yaml) |
| OccAnt(rgb)     |   Gibson 4+   |      0.825         |      0.432      | [ckpt.11.pth](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/occant_rgb/ckpt.11.pth) | [ppo_exploration.yaml](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/occant_rgb/ppo_exploration.yaml) |
| OccAnt(depth)   |   Gibson 4+   |      0.912         |      0.492      | [ckpt.11.pth](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/occant_depth/ckpt.11.pth) | [ppo_exploration.yaml](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/occant_depth/ppo_exploration.yaml) |
| OccAnt(rgbd)    |   Gibson 4+   |      0.911         |      0.510      | [ckpt.10.pth](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/occant_rgbd/ckpt.10.pth) | [ppo_exploration.yaml](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/occant_rgbd/ppo_exploration.yaml) |


We also provide a habitat challenge version of our model with the navigation results on Gibson under challenge conditions.

| Model name      | Training data |       Val SPL      |    Test Std. SPL   | Checkpoint URL |  Eval config URL |
|-----------------|:-------------:|:------------------:|:------------------:|:--------------:|:----------------:|
| OccAnt(depth)   |   Gibson 2+   |       0.463        |       0.190        | [ckpt.13.pth](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/occant_depth_ch/ckpt.13.pth) | [ppo_navigation_evaluate.yaml](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models/occant_depth_ch/ppo_navigation_evaluate.yaml) |


## Replicating ECCV results

Since this version of code is a well-tuned version, it results is much higher performance than those reported in our paper. To replicate results from the paper, we provide the models used to generate the ECCV results and the corresponding evaluation scripts. 
Checkout the `eccv_2020_eval` branch for the evaluation code. The pretrained models are provided below along with the benchmarked navigation results on Gibson under noise-free conditions:

| Model name      | Training data |     Val SPL     |   Checkpoint URL  |     Eval config URL   |
|-----------------|:-------------:|:---------------:|:-----------------:|:---------------------:|
| ANS(rgb)        |    Gibson 4+  |      0.586      | [ckpt.16.pth](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models_eccv_2020/pretrained_models/ans_rgb/ckpt.16.pth) | [ppo_navigation\_evaluate.yaml](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models_eccv_2020/pretrained_models/ans_rgb/ppo_navigation_evaluate.yaml) |
| ANS(depth)      |    Gibson 4+  |      0.749      | [ckpt.16.pth](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models_eccv_2020/pretrained_models/ans_depth/ckpt.16.pth) | [ppo_navigation\_evaluate.yaml](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models_eccv_2020/pretrained_models/ans_depth/ppo_navigation_evaluate.yaml) |
| OccAnt(rgb)     |    Gibson 4+  |      0.707      | [ckpt.8.pth](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models_eccv_2020/pretrained_models/occant_rgb/ckpt.8.pth) | [ppo_navigation\_evaluate.yaml](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models_eccv_2020/pretrained_models/occant_rgb/ppo_navigation_evaluate.yaml) |
| OccAnt(depth)   |    Gibson 4+  |      0.784      | [ckpt.6.pth](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models_eccv_2020/pretrained_models/occant_depth/ckpt.6.pth) | [ppo_navigation\_evaluate.yaml](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models_eccv_2020/pretrained_models/occant_depth/ppo_navigation_evaluate.yaml) |
| OccAnt(rgbd)    |    Gibson 4+  |      0.782      | [ckpt.7.pth](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models_eccv_2020/pretrained_models/occant_rgbd/ckpt.7.pth) | [ppo_navigation\_evaluate.yaml](https://dl.fbaipublicfiles.com/OccupancyAnticipation/pretrained_models_eccv_2020/pretrained_models/occant_rgbd/ppo_navigation_evaluate.yaml) |

## Acknowledgements
This repository uses parts of [Habitat Lab](https://github.com/facebookresearch/habitat-lab) and extends it. We also used [Neural SLAM](https://github.com/devendrachaplot/Neural-SLAM) as a reference for the planning heuristics. We thank Devendra Singh Chaplot for helping replicate the Active Neural SLAM results during the early stages of the project.

## Citation
```
@inproceedings{ramakrishnan2020occant,
  title={Occupancy Anticipation for Efficient Exploration and Navigation},
  author={Santhosh Kumar Ramakrishnan, Ziad Al-Halah and Kristen Grauman},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

# License
This project is released under the MIT license, as found in the LICENSE file.
