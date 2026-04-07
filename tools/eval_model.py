
import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0
from utils.util_config import cfg, cfg_from_yaml_file

def evaluate_model(model_path, config_path, split='test'):
    print(f"* Evaluating model: {model_path}")
    print(f"* Config: {config_path}")
    
    # Initialize pipeline
    pline = PipelineDetection_v1_0(path_cfg=config_path, mode='test')
    
    # Load model
    print(f"* Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle DDP state dict (remove 'module.' prefix)
    state_dict = checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    pline.network.load_state_dict(new_state_dict)
    pline.network.eval()
    pline.network.cuda()
    
    # Run validation
    print("* Starting validation...")
    # Use conditional validation to get detailed metrics
    pline.validate_kitti_conditional(epoch=18, list_conf_thr=[0.3], is_subset=False, is_print_memory=False)

if __name__ == "__main__":
    # Hardcoded paths based on user request
    model_path = '/home/hongsheng/RL_3DOD/logs/exp_260223_225750_RL_3df_gate/models/model_18.pt'
    config_path = '/home/hongsheng/RL_3DOD/configs/cfg_rl_3df_gate.yml'
    
    evaluate_model(model_path, config_path)
