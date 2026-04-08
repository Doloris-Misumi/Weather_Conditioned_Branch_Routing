import os
import argparse

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='Path to the experiment config file.',
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to the trained model checkpoint.',
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='Epoch number used only for naming/logging in validation outputs.',
    )
    parser.add_argument(
        '--conf-thr',
        type=float,
        default=0.3,
        help='Confidence threshold for validation.',
    )
    parser.add_argument(
        '--subset',
        action='store_true',
        help='Run conditional validation on the subset split if supported.',
    )
    args = parser.parse_args()

    pline = PipelineDetection_v1_0(path_cfg=args.config, mode='test')
    
    # Only copy code if logging is enabled (Rank 0)
    import shutil
    if getattr(pline, 'is_logging', False):
        shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
        
    pline.load_dict_model(args.checkpoint)
    
    # Conditional validation
    pline.validate_kitti_conditional(
        epoch=args.epoch,
        list_conf_thr=[args.conf_thr],
        is_subset=args.subset,
        is_print_memory=False,
    )
    
    if pline.is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()
