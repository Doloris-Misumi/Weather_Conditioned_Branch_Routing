import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

EXP_NAME = 'exp_260221_012535_RL_3df_gate'
MODEL_EPOCH = 19

if __name__ == '__main__':
    PATH_CONFIG = f'./logs/{EXP_NAME}/config.yml'
    PATH_MODEL = f'./logs/{EXP_NAME}/models/model_{MODEL_EPOCH}.pt'

    pline = PipelineDetection_v1_0(path_cfg=PATH_CONFIG, mode='test')
    
    # Only copy code if logging is enabled (Rank 0)
    import shutil
    if getattr(pline, 'is_logging', False):
        shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
        
    pline.load_dict_model(PATH_MODEL)
    
    # Conditional validation
    pline.validate_kitti_conditional(epoch=MODEL_EPOCH, list_conf_thr=[0.3], is_subset=False, is_print_memory=False)
    
    if pline.is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()
    
