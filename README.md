# Weather_Conditioned_Branch_Routing

Official repository for weather-conditioned branch routing built on top of the RL_3DOD / K-Radar codebase. The original RL_3DOD project is associated with "Towards Robust 3D Object Detection with LiDAR and 4D Radar Fusion in Various Weather Conditions", CVPR 2024. [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Chae_Towards_Robust_3D_Object_Detection_with_LiDAR_and_4D_Radar_CVPR_2024_paper.pdf)

## Overview

This repository contains:

- training and evaluation code for the main detection model
- stage-1 image-based weather classifier training code
- dataset split files and small resource files required by the pipeline
- utility scripts for evaluation, export, and result inspection

This repository does not include:

- the K-Radar dataset
- trained checkpoints
- experiment logs or generated visualizations

## Environment

The codebase has been tested with:

- Python 3.8
- CUDA 11.1
- PyTorch 1.10.1
- torchvision 0.11.2
- spconv-cu111 2.1.25
- open3d 0.15.2
- opencv-python 4.8.1.78
- matplotlib 3.5.3
- numba 0.53.0
- nms 0.1.6

## Installation

Create and activate your environment first, then install dependencies with either:

```bash
pip install -r requirements.txt
```

or:

```bash
bash install_deps.sh
```

## Dataset Setup

Prepare the K-Radar dataset locally and point the config to your local copy in [`configs/cfg_rl_3df_gate.yml`](./configs/cfg_rl_3df_gate.yml).

Default expected dataset root:

```bash
./data/k_radar_dataset
```

Main fields to update when needed:

- `DATASET.DIR.LIST_DIR`
- `DATASET.DIR.DIR_DOPPLER_CB`
- `DATASET.DIR.DIR_SPARSE_CB`

## Checkpoints

Stage-2 training expects a pretrained stage-1 weather classifier checkpoint via:

```yaml
MODEL:
  IMG_CLS:
    MODEL_PATH: ./models/img_cls/<your_classifier_checkpoint>.pth
```

If you do not already have this checkpoint, train it first with `models/img_cls/cls_train.py` and update the config accordingly.

## Usage

### Stage 1: Train the Weather Classifier

```bash
python models/img_cls/cls_train.py
```

### Stage 2: Train the Detector

```bash
python main_train_0.py
```

### Evaluate a Trained Detector

```bash
python main_cond_0.py \
  --config ./logs/<experiment_name>/config.yml \
  --checkpoint ./logs/<experiment_name>/models/model_<epoch>.pt \
  --epoch <epoch>
```

### Utility Scripts

```bash
python tools/eval_model.py --config configs/cfg_rl_3df_gate.yml --checkpoint <path_to_model.pt>
python tools/export_sample_modalities.py --dataset_root ./data/k_radar_dataset
python tools/export_detection_outputs.py --dataset_root ./data/k_radar_dataset
python tools/plot_loss_curves.py --log-dir ./logs/<experiment_name>
python tools/summarize_training.py --log-dir ./logs/<experiment_name>
```

### KITTI-Style Evaluation Helper

```bash
python utils/kitti_eval/eval_python.py --header <path_to_kitti_eval_folder>
```

The header directory is expected to contain:

- `pred/`
- `gt/`
- `val.txt`

## Repository Layout

```text
Weather_Conditioned_Branch_Routing/
├── configs/        # experiment configuration
├── datasets/       # K-Radar dataset loader
├── models/         # detector, classifier, and backbone modules
├── pipelines/      # training / evaluation pipeline
├── resources/      # small static resources and split files
├── tools/          # export, plotting, and evaluation helpers
├── utils/          # geometry, NMS, metrics, and utility code
├── main_train_0.py # detector training entrypoint
└── main_cond_0.py  # conditional evaluation entrypoint
```

## Notes for Open-Source Use

- Keep datasets, checkpoints, and logs outside version control.
- Update config paths to your own local environment before training.
- Some utility scripts assume GPU/CUDA availability.

## Citation

If this repository is helpful for your work, please cite the corresponding paper:

```bibtex
@InProceedings{Chae_2024_CVPR,
    author    = {Chae, Yujeong and Kim, Hyeonseong and Yoon, Kuk-Jin},
    title     = {Towards Robust 3D Object Detection with LiDAR and 4D Radar Fusion in Various Weather Conditions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {15162-15172}
}
```

## Acknowledgements

This work is developed based on the [K-Radar dataset and codebase](https://github.com/kaist-avelab/K-Radar).
