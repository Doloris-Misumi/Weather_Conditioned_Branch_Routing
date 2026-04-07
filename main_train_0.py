'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import os
import argparse

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

PATH_CONFIG = './configs/cfg_rl_3df_gate.yml' 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', dest='local_rank_dash', type=int, default=-1)
    args, _ = parser.parse_known_args()

    local_rank = args.local_rank if args.local_rank != -1 else args.local_rank_dash
    if local_rank != -1 and os.environ.get('LOCAL_RANK') is None:
        os.environ['LOCAL_RANK'] = str(local_rank)

    pline = PipelineDetection_v1_0(path_cfg=PATH_CONFIG, mode='train')

    ### Save this file for checking ###
    import shutil
    if getattr(pline, 'is_logging', False):
        shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
    ### Save this file for checking ###

    pline.train_network()

    if (not pline.is_distributed) or (pline.local_rank == 0):
        pline.validate_kitti_conditional(list_conf_thr=[0.3], is_subset=False, is_print_memory=False)
    
    if pline.is_distributed:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    
