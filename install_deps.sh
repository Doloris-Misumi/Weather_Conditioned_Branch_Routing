#!/bin/bash
source ~/miniconda3/bin/activate rl_3dod
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install spconv-cu111==2.1.25
pip install open3d==0.15.2
pip install opencv-python==4.8.1.78
pip install matplotlib==3.5.3
pip install numba==0.53.0
pip install nms==0.1.6
pip install easydict scipy tqdm pyyaml
