import torch
import numpy as np
import open3d as o3d
import os
import math
from tqdm import tqdm
import shutil
from torch.utils.data import Subset
import cv2
import matplotlib.pyplot as plt
import time
import re
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Ingnore numba warning
from numba.core.errors import NumbaWarning
import warnings
import logging
import csv
import json
warnings.simplefilter('ignore', category=NumbaWarning)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils.util_pipeline import *
from utils.util_point_cloud import *
from utils.util_config import cfg, cfg_from_yaml_file
from utils.util_ui_labeling import *

from utils.util_point_cloud import Object3D
import utils.kitti_eval.kitti_common as kitti
from utils.kitti_eval.eval import get_official_eval_result

class WeightedDistributedSampler(torch.utils.data.Sampler):
    """Distributed sampler with weighted sampling support."""
    def __init__(self, weights, num_replicas=None, rank=None, replacement=True, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.dataset_size = int(self.weights.numel())
        self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        self.total_size = int(self.num_samples * self.num_replicas)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights, self.total_size, self.replacement, generator=generator
        )
        indices = sampled[self.rank:self.total_size:self.num_replicas].tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class PipelineDetection_v1_0():
    def __init__(self, path_cfg=None, mode='train'):
        '''
        * mode in ['train', 'test', 'vis']
        *   'train' denotes both train & test
        *   'test'  denotes mode for inference
        '''
        self.cfg = cfg_from_yaml_file(path_cfg, cfg)
        self.mode = mode
        # self.update_cfg_regarding_mode()
        
        # Distributed Setup
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.local_rank != -1
        
        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            ddp_timeout_minutes = int(self.cfg.GENERAL.get('DDP_TIMEOUT_MINUTES', 240))
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=datetime.timedelta(minutes=ddp_timeout_minutes),
            )
            self.device = torch.device(f'cuda:{self.local_rank}')
            print(f'* Init process group: rank {self.local_rank}/{self.world_size}')
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if self.cfg.GENERAL.SEED is not None:
            try:
                set_random_seed(cfg.GENERAL.SEED, cfg.GENERAL.IS_CUDA_SEED, cfg.GENERAL.IS_DETERMINISTIC)
            except:
                print('* Exception error: check cfg.GENERAL for seed')
                set_random_seed(cfg.GENERAL.SEED)
        
        self.dataset_train = build_dataset(self, split='train') if self.mode == 'train' else None
        self.dataset_test = build_dataset(self, split='test')
        if mode == 'train':
            self.cfg.DATASET.NUM = len(self.dataset_train)
        elif mode in ['test', 'vis']:
            self.cfg.DATASET.NUM = len(self.dataset_test)
        # print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) # check if it is updated

        self.network = build_network(self).to(self.device)
        self.optimizer = build_optimizer(self, self.network)
        self.scheduler = build_scheduler(self, self.optimizer)
        self.epoch_start = 0

        # DDP Wrapping
        if self.is_distributed:
            # self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(self.network, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
            # Scale LR for distributed training
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] *= self.world_size
        
        # Logging
        self.is_logging = False
        if self.cfg.GENERAL.LOGGING.IS_LOGGING:
            if self.is_distributed:
                if self.local_rank == 0:
                    self.set_logging(path_cfg)
            else:
                self.set_logging(path_cfg)

        # Validation
        if self.cfg.VAL.IS_VALIDATE:
            if self.is_distributed:
                if self.local_rank == 0:
                    self.set_validate()
                else:
                    self.is_validate = False
            else:
                self.set_validate()
        else:
            self.is_validate = False
        
        if self.cfg.GENERAL.RESUME.IS_RESUME:
            self.resume_network()

        # Vis
        self.set_vis()
        
        if (not self.is_distributed) or self.local_rank == 0:
            self.show_pline_description()
        
    def update_cfg_regarding_mode(self):
        '''
        * You don't have to update values in cfg changed in dataset
        * They are related in pointer
        * e.g., check print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) after dataset initialization
        '''
        if self.mode == 'train':
            pass
        elif self.mode == 'test':
            self.cfg.OPTIMIZER.NUM_WORKERS = 16
        elif self.mode == 'vis':
            self.cfg.OPTIMIZER.NUM_WORKERS = 16
            self.cfg.GET_ITEM = {
                'rdr_sparse_cube'   : True,
                'rdr_tesseract'     : False,
                'rdr_cube'          : True,
                'rdr_cube_doppler'  : False,
                'ldr_pc_64'         : True,
                'cam_front_img'     : True,
            }
        else:
            print('* Exception error (Pipeline): check modify_cfg')
        return

    def set_validate(self):
        self.is_validate = True
        self.is_consider_subset = self.cfg.VAL.IS_CONSIDER_VAL_SUBSET
        self.val_per_epoch_subset = self.cfg.VAL.VAL_PER_EPOCH_SUBSET
        self.val_num_subset = self.cfg.VAL.NUM_SUBSET
        self.val_per_epoch_full = self.cfg.VAL.VAL_PER_EPOCH_FULL
        self.val_full_only_at_end = bool(self.cfg.VAL.get('FULL_ONLY_AT_END', True))
        self.run_conditional_at_end = bool(self.cfg.VAL.get('RUN_CONDITIONAL_AT_END', True))
        self.skip_standard_full_when_conditional = bool(
            self.cfg.VAL.get('SKIP_STANDARD_FULL_WHEN_CONDITIONAL', True)
        )

        self.val_keyword = self.cfg.VAL.CLASS_VAL_KEYWORD # for kitti_eval
        list_val_keyword_keys = list(self.val_keyword.keys()) # same order as VAL.CLASS_VAL_KEYWORD.keys()
        self.list_val_care_idx = []

        # index matching with kitti_eval
        for cls_name in self.cfg.VAL.LIST_CARE_VAL:
            idx_val_cls = list_val_keyword_keys.index(cls_name)
            self.list_val_care_idx.append(idx_val_cls)
        # print(self.list_val_care_idx)

        ### Consider output of network and dataset ###
        if self.cfg.VAL.REGARDING == 'anchor':
            self.val_regarding = 0 # anchor
            self.list_val_conf_thr = self.cfg.VAL.LIST_VAL_CONF_THR
        else:
            print('* Exception error: check VAL.REGARDING')
        ### Consider output of network and dataset ###

    def set_vis(self):
        self.dict_cls_name_to_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
        self.dict_cls_id_to_name = dict()
        for k, v in self.dict_cls_name_to_id.items():
            if v != -1:
                self.dict_cls_id_to_name[v] = k
        self.dict_cls_name_to_bgr = self.cfg.VIS.CLASS_BGR
        self.dict_cls_name_to_rgb = self.cfg.VIS.CLASS_RGB
    
    def show_pline_description(self):
        print('* newtork (description start) -------')
        print(self.network)
        print('* newtork (description end) ---------')
        print('* optimizer (description start) -----')
        print(self.optimizer)
        print('* optimizer (description end) -------')
        print(f'* mode = {self.mode}')
        len_data = self.cfg.DATASET.NUM
        print(f'* dataset length = {len_data}')
        self.print_runtime_monitors()

    def print_runtime_monitors(self):
        if self.mode != 'train':
            return

        model = self.network.module if self.is_distributed else self.network
        backbone = getattr(model, 'backbone_3d', None)
        if backbone is None:
            return

        floor = getattr(backbone, 'branch_weight_floor', None)
        if floor is not None:
            print(f'* Branch weight floor = {float(floor):.4f}')
        force_branch = getattr(backbone, 'force_branch', 'none')
        print(f'* Branch force mode = {force_branch}')

        init_w = getattr(backbone, 'initial_branch_weights', None)
        if isinstance(init_w, (list, tuple)) and len(init_w) >= 3:
            print(
                f"* Branch initial weights (L/R/F) = "
                f"{float(init_w[0]):.4f}/{float(init_w[1]):.4f}/{float(init_w[2]):.4f}"
            )
        use_weather_sampler = bool(self.cfg.OPTIMIZER.get('USE_WEATHER_BALANCED_SAMPLER', False))
        print(f'* Weather balanced sampler = {use_weather_sampler}')
        weather_aux_cfg = self.cfg.MODEL.get('WEATHER_AUX', {})
        aux_weights = weather_aux_cfg.get('CLASS_WEIGHTS', [])
        aux_weights_str = ''
        if isinstance(aux_weights, (list, tuple)) and len(aux_weights) > 0:
            aux_weights_str = f", class_weights={list(aux_weights)}"
        print(
            '* Weather auxiliary cls = '
            f"{bool(weather_aux_cfg.get('ENABLED', False))} "
            f"(lambda={float(weather_aux_cfg.get('LAMBDA', 0.0)):.4f}{aux_weights_str})"
        )
        weather_div_cfg = self.cfg.MODEL.get('WEATHER_DIVERSITY', {})
        print(
            '* Weather diversity loss = '
            f"{bool(weather_div_cfg.get('ENABLED', False))} "
            f"(lambda={float(weather_div_cfg.get('LAMBDA', 0.0)):.4f})"
        )
        branch_entropy_cfg = self.cfg.MODEL.get('BRANCH_ENTROPY', {})
        print(
            '* Branch entropy reg = '
            f"{bool(branch_entropy_cfg.get('ENABLED', False))} "
            f"(lambda={float(branch_entropy_cfg.get('LAMBDA', 0.0)):.4f}, "
            f"target_ratio={float(branch_entropy_cfg.get('TARGET_RATIO', 0.0)):.3f})"
        )
    
    def set_logging(self, path_cfg, is_print_where=True):
        self.is_logging = True
        str_local_time = get_local_time_str()
        str_exp = 'exp_' + str_local_time + '_' + self.cfg.GENERAL.NAME
        self.path_log = os.path.join(self.cfg.GENERAL.LOGGING.PATH_LOGGING, str_exp)
        if is_print_where:
            print(f'* Start logging in {str_exp}')
        if not (os.path.exists(self.path_log)):
            os.makedirs(self.path_log)
        else:
            print('* Exception error (Pipeline): same folder exists, try again')
            exit()

        self.log_train_iter = SummaryWriter(os.path.join(self.path_log, 'train_iter'), comment='iteration')
        self.log_train_epoch = SummaryWriter(os.path.join(self.path_log, 'train_epoch'), comment='epoch')
        self.log_test = SummaryWriter(os.path.join(self.path_log, 'test'), comment='test')
        self.log_iter_start = None

        self.is_save_model = self.cfg.GENERAL.LOGGING.IS_SAVE_MODEL
        try:
            self.interval_epoch_model = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_MODEL
            self.interval_epoch_util = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_UTIL
        except:
            self.interval_epoch_model = 1
            self.interval_epoch_util = 5
            print('* Exception error (Pipeline): check LOGGING.INTERVAL_EPOCH_MODEL/UTIL')
        if self.is_save_model:
            os.makedirs(os.path.join(self.path_log, 'models'))
            os.makedirs(os.path.join(self.path_log, 'utils'))

        # cfg backup (same files, just for identification)
        name_file_origin = path_cfg.split('/')[-1] # original cfg file name
        name_file_cfg = 'config.yml'
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_origin))
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_cfg))

        # code backup (TBD)

    def resume_network(self):
        path_exp = self.cfg.GENERAL.RESUME.PATH_EXP
        path_state_dict = os.path.join(path_exp, 'utils')
        epoch = self.cfg.GENERAL.RESUME.START_EP
        list_epochs = sorted(list(map(lambda x: int(x.split('.')[0].split('_')[1]), os.listdir(path_state_dict))))
        epoch = list_epochs[-1] if epoch is None else epoch

        path_state_dict = os.path.join(path_state_dict, f'util_{epoch}.pt')
        print('* Start resume, path_state_dict =  ', path_state_dict)
        state_dict = torch.load(path_state_dict)

        try:
            self.epoch_start = epoch + 1
            self.network.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.log_iter_start = state_dict['idx_log_iter']
            print(f'* Network & Optimizer are loaded / Resume epoch is {epoch} / Start from {self.epoch_start} ...')
        except:
            print('* Exception error (Pipeline): check resume network')
            exit()
        if ('scheduler_state_dict' in state_dict.keys()) and (not (self.scheduler is None)):
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            print('* Scheduler is loaded')
        else:
            print('* Scheduler is started from vanilla')

        ### Copy logging folder ###
        list_copy_dirs = ['train_epoch', 'train_iter', 'test', 'test_kitti']
        if (self.cfg.GENERAL.RESUME.IS_COPY_LOGS) and (self.is_logging):
            for copy_dir in list_copy_dirs:
                shutil.copytree(os.path.join(path_exp, copy_dir), \
                    os.path.join(self.path_log, copy_dir), dirs_exist_ok=True)
        ### Copy logging folder ###

        return

    def _extract_weather_label_from_meta(self, meta):
        if not isinstance(meta, dict):
            return -1
        try:
            if 'image_cls_label' in meta:
                return int(meta['image_cls_label'])
        except Exception:
            pass

        desc = meta.get('desc', {})
        if isinstance(desc, dict) and hasattr(self.dataset_train, 'get_weather_label'):
            try:
                return int(self.dataset_train.get_weather_label(desc.get('climate', 'unknown')))
            except Exception:
                return -1
        return -1

    def _get_weather_labels_from_dataset(self, dataset):
        cached = getattr(dataset, '_cached_weather_labels', None)
        if cached is not None and len(cached) == len(dataset):
            return cached

        weather_labels = []
        if hasattr(dataset, 'list_dict_item') and len(getattr(dataset, 'list_dict_item', [])) > 0:
            for item in dataset.list_dict_item:
                weather_labels.append(self._extract_weather_label_from_meta(item.get('meta', {})))
        elif hasattr(dataset, 'list_path_label'):
            for path_label in dataset.list_path_label:
                try:
                    dict_path = dataset.get_path_data_from_path_label(path_label)
                    desc = dataset.get_description(dict_path['path_desc'])
                    weather_labels.append(int(dataset.get_weather_label(desc.get('climate', 'unknown'))))
                except Exception:
                    weather_labels.append(-1)
        else:
            for idx in range(len(dataset)):
                try:
                    item = dataset[idx]
                    weather_labels.append(self._extract_weather_label_from_meta(item.get('meta', {})))
                except Exception:
                    weather_labels.append(-1)

        dataset._cached_weather_labels = weather_labels
        return weather_labels

    def _build_weather_sample_weights(self, dataset):
        weather_labels = self._get_weather_labels_from_dataset(dataset)
        num_samples = len(weather_labels)
        sample_weights = np.ones(num_samples, dtype=np.float64)

        valid_labels = [lbl for lbl in weather_labels if lbl >= 0]
        if len(valid_labels) == 0:
            dataset._weather_sampler_stats = {'enabled': False, 'reason': 'no_valid_weather_label'}
            return torch.as_tensor(sample_weights, dtype=torch.double)

        values, counts = np.unique(valid_labels, return_counts=True)
        count_dict = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}
        max_count = max(count_dict.values())

        power = float(self.cfg.OPTIMIZER.get('WEATHER_BALANCE_POW', 0.5))
        max_ratio = float(self.cfg.OPTIMIZER.get('WEATHER_BALANCE_MAX_RATIO', 3.0))
        min_ratio = float(self.cfg.OPTIMIZER.get('WEATHER_BALANCE_MIN_RATIO', 1.0))

        ratio_dict = {}
        for label_idx, label_count in count_dict.items():
            ratio = (max_count / max(label_count, 1)) ** power
            ratio = max(min_ratio, min(max_ratio, ratio))
            ratio_dict[label_idx] = float(ratio)

        for idx_sample, label_idx in enumerate(weather_labels):
            if label_idx >= 0 and label_idx in ratio_dict:
                sample_weights[idx_sample] = ratio_dict[label_idx]

        dataset._weather_sampler_stats = {
            'enabled': True,
            'count_dict': count_dict,
            'ratio_dict': ratio_dict,
            'max_count': int(max_count),
            'power': power,
            'min_ratio': min_ratio,
            'max_ratio': max_ratio,
        }
        return torch.as_tensor(sample_weights, dtype=torch.double)

    def _compute_weather_diversity_loss(self, dict_net, dict_datum):
        cfg_div = self.cfg.MODEL.get('WEATHER_DIVERSITY', {})
        if not bool(cfg_div.get('ENABLED', False)):
            return None, {}

        if ('branch_weights' not in dict_net) or (dict_net['branch_weights'] is None):
            return None, {}
        branch_weights = dict_net['branch_weights']
        if (not torch.is_tensor(branch_weights)) or branch_weights.ndim != 2 or branch_weights.shape[1] < 3:
            return None, {}

        meta_list = dict_datum.get('meta', None)
        if isinstance(meta_list, tuple):
            meta_list = list(meta_list)
        if not isinstance(meta_list, list) or len(meta_list) == 0:
            return None, {}

        weather_labels = torch.tensor(
            [self._extract_weather_label_from_meta(meta) for meta in meta_list],
            dtype=torch.long,
            device=branch_weights.device
        )
        valid_mask = weather_labels >= 0
        if int(valid_mask.sum().item()) < 2:
            return None, {}

        branch_weights = branch_weights[valid_mask, :3]
        weather_labels = weather_labels[valid_mask]
        unique_weather = torch.unique(weather_labels)
        if int(unique_weather.numel()) == 0:
            return None, {}

        min_samples_per_class = int(cfg_div.get('MIN_SAMPLES_PER_CLASS', 2))
        class_centers = []
        intra_terms = []
        for weather_idx in unique_weather:
            class_mask = weather_labels == weather_idx
            class_weights = branch_weights[class_mask]
            if class_weights.shape[0] == 0:
                continue
            center = class_weights.mean(dim=0)
            class_centers.append(center)
            if class_weights.shape[0] >= min_samples_per_class:
                intra_terms.append(((class_weights - center) ** 2).mean())

        if len(class_centers) == 0:
            return None, {}

        if len(intra_terms) > 0:
            intra_loss = torch.stack(intra_terms).mean()
        else:
            intra_loss = branch_weights.new_tensor(0.0)

        inter_hinge = branch_weights.new_tensor(0.0)
        inter_mean_dist = branch_weights.new_tensor(0.0)
        if len(class_centers) >= 2:
            centers = torch.stack(class_centers, dim=0)
            center_dists = torch.pdist(centers, p=2)
            if center_dists.numel() > 0:
                margin = float(cfg_div.get('INTER_MARGIN', 0.12))
                inter_hinge = torch.relu(margin - center_dists).mean()
                inter_mean_dist = center_dists.mean()

        inter_weight = float(cfg_div.get('INTER_WEIGHT', 1.0))
        div_raw = intra_loss + inter_weight * inter_hinge
        lambda_div = float(cfg_div.get('LAMBDA', 0.0))
        div_loss = lambda_div * div_raw

        log_dict = {
            'loss_weather_div': float(div_loss.detach().item()),
            'loss_weather_div_raw': float(div_raw.detach().item()),
            'weather_div_intra': float(intra_loss.detach().item()),
            'weather_div_inter_hinge': float(inter_hinge.detach().item()),
            'weather_div_inter_dist': float(inter_mean_dist.detach().item()),
            'weather_div_num_weather_in_batch': float(unique_weather.numel()),
        }
        return div_loss, log_dict

    def _compute_weather_aux_loss(self, dict_net, dict_datum):
        cfg_aux = self.cfg.MODEL.get('WEATHER_AUX', {})
        if not bool(cfg_aux.get('ENABLED', False)):
            return None, {}

        weather_logits = dict_net.get('weather_logits_aux', None)
        if weather_logits is None or (not torch.is_tensor(weather_logits)) or weather_logits.ndim != 2:
            return None, {}

        meta_list = dict_datum.get('meta', None)
        if isinstance(meta_list, tuple):
            meta_list = list(meta_list)
        if not isinstance(meta_list, list) or len(meta_list) == 0:
            return None, {}

        weather_labels = torch.tensor(
            [self._extract_weather_label_from_meta(meta) for meta in meta_list],
            dtype=torch.long,
            device=weather_logits.device
        )
        valid_mask = (weather_labels >= 0) & (weather_labels < weather_logits.shape[1])
        if int(valid_mask.sum().item()) == 0:
            return None, {}

        class_weights = cfg_aux.get('CLASS_WEIGHTS', None)
        ce_weight = None
        if isinstance(class_weights, (list, tuple)) and len(class_weights) > 0:
            ce_weight = torch.tensor(class_weights, dtype=weather_logits.dtype, device=weather_logits.device)
            if ce_weight.numel() != weather_logits.shape[1]:
                ce_weight = None

        aux_raw = F.cross_entropy(
            weather_logits[valid_mask],
            weather_labels[valid_mask],
            weight=ce_weight
        )
        lambda_aux = float(cfg_aux.get('LAMBDA', 0.0))
        aux_loss = lambda_aux * aux_raw

        pred = torch.argmax(weather_logits[valid_mask], dim=-1)
        acc = (pred == weather_labels[valid_mask]).float().mean()
        log_dict = {
            'loss_weather_aux': float(aux_loss.detach().item()),
            'loss_weather_aux_raw': float(aux_raw.detach().item()),
            'weather_aux_acc': float(acc.detach().item()),
            'weather_aux_valid_count': float(valid_mask.sum().item()),
        }
        if ce_weight is not None:
            weak_weather = ['fog', 'sleet', 'heavysnow']
            weather_names = getattr(self.dataset_train, 'weather_list', weak_weather)
            for weather_name in weak_weather:
                if weather_name in weather_names:
                    idx = weather_names.index(weather_name)
                    if idx < int(ce_weight.numel()):
                        log_dict[f'weather_aux_weight_{weather_name}'] = float(ce_weight[idx].detach().item())
        return aux_loss, log_dict

    def _compute_branch_entropy_loss(self, dict_net):
        cfg_entropy = self.cfg.MODEL.get('BRANCH_ENTROPY', {})
        if not bool(cfg_entropy.get('ENABLED', False)):
            return None, {}

        branch_weights = dict_net.get('branch_weights', None)
        if branch_weights is None or (not torch.is_tensor(branch_weights)) or branch_weights.ndim != 2:
            return None, {}
        if branch_weights.shape[1] < 3:
            return None, {}

        branch_weights = branch_weights[:, :3]
        branch_weights = branch_weights.clamp(min=1e-12)
        branch_weights = branch_weights / branch_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

        entropy = -(branch_weights * torch.log(branch_weights)).sum(dim=1)
        entropy_ratio = entropy.mean() / math.log(3.0)

        target_ratio = float(cfg_entropy.get('TARGET_RATIO', 0.78))
        target_ratio = min(max(target_ratio, 0.0), 1.0)
        entropy_raw = torch.relu(branch_weights.new_tensor(target_ratio) - entropy_ratio)

        lambda_entropy = float(cfg_entropy.get('LAMBDA', 0.0))
        entropy_loss = lambda_entropy * entropy_raw

        log_dict = {
            'loss_branch_entropy': float(entropy_loss.detach().item()),
            'loss_branch_entropy_raw': float(entropy_raw.detach().item()),
            'branch_entropy_ratio_batch': float(entropy_ratio.detach().item()),
            'branch_entropy_target_ratio': float(target_ratio),
        }
        return entropy_loss, log_dict

    def train_network(self, is_shuffle=True):
        self.network.train()

        use_weather_sampler = bool(self.cfg.OPTIMIZER.get('USE_WEATHER_BALANCED_SAMPLER', False))
        sampler = None
        shuffle = bool(is_shuffle)
        if use_weather_sampler:
            sample_weights = self._build_weather_sample_weights(self.dataset_train)
            if self.is_distributed:
                sampler = WeightedDistributedSampler(
                    weights=sample_weights,
                    num_replicas=self.world_size,
                    rank=self.local_rank,
                    replacement=True,
                    seed=int(self.cfg.GENERAL.get('SEED', 0) or 0),
                )
            else:
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=int(sample_weights.numel()),
                    replacement=True,
                )
            shuffle = False
        else:
            sampler = DistributedSampler(self.dataset_train, shuffle=is_shuffle) if self.is_distributed else None
            shuffle = is_shuffle and (sampler is None)

        if self.local_rank == 0 or not self.is_distributed:
            sampler_stats = getattr(self.dataset_train, '_weather_sampler_stats', {})
            if use_weather_sampler and sampler_stats.get('enabled', False):
                print(
                    '* Weather sampler enabled: '
                    f"pow={sampler_stats.get('power', 0.0):.3f}, "
                    f"ratio=[{sampler_stats.get('min_ratio', 1.0):.2f}, {sampler_stats.get('max_ratio', 1.0):.2f}]"
                )
            elif use_weather_sampler:
                print(
                    '* Weather sampler requested but disabled internally '
                    f"(reason={sampler_stats.get('reason', 'unknown')})"
                )

        data_loader_train = torch.utils.data.DataLoader(self.dataset_train, \
            batch_size = self.cfg.OPTIMIZER.BATCH_SIZE, shuffle = shuffle, \
            collate_fn = self.dataset_train.collate_fn,
            sampler=sampler,
            num_workers = self.cfg.OPTIMIZER.NUM_WORKERS, drop_last = True)

        epoch_start = self.epoch_start
        epoch_end = self.cfg.OPTIMIZER.MAX_EPOCH

        if self.is_logging:
            idx_log_iter = 0 if self.log_iter_start is None else self.log_iter_start

        for epoch in range(epoch_start, epoch_end):
            if self.is_distributed and (sampler is not None) and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
            
            torch.cuda.empty_cache()
            if self.local_rank == 0 or not self.is_distributed:
                print(f'* Training epoch = {epoch}/{epoch_end-1}')
            if self.is_logging:
                print(f'* Logging path = {self.path_log}')
            
            self.network.train()
            # Handle DDP .module access if needed, but training=True is on the wrapper too
            if self.is_distributed:
                self.network.module.training = True
            else:
                self.network.training = True
            
            avg_loss = []
            branch_monitor = self._init_branch_weight_monitor(
                num_conditions=getattr(self.dataset_train, 'num_condition_classes', 0),
                num_bins=50
            )
            
            # Only show tqdm on rank 0
            if self.local_rank == 0 or not self.is_distributed:
                pbar = tqdm(data_loader_train)
            else:
                pbar = data_loader_train
                
            for idx_iter, dict_datum in enumerate(pbar):
                if (idx_iter % 50) == 49:
                    torch.cuda.empty_cache()
                
                # In DDP, forward call is on self.network (the wrapper)
                dict_datum['epoch'] = epoch
                dict_datum['idx_iter'] = idx_iter
                dict_datum['local_rank'] = self.local_rank
                dict_net = self.network(dict_datum)
                batch_branch_log = self._update_branch_weight_monitor(branch_monitor, dict_net)
                if batch_branch_log:
                    if 'logging' not in dict_net:
                        dict_net['logging'] = {}
                    dict_net['logging'].update(batch_branch_log)
                
                # Loss calculation
                # Accessing .head requires .module in DDP
                if self.is_distributed:
                    loss = self.network.module.head.loss(dict_net)
                    if hasattr(self.network.module, 'point_head'):
                        point_loss = self.network.module.point_head.loss(dict_net)
                        loss += point_loss
                    if hasattr(self.network.module, 'roi_head'):
                        roi_loss = self.network.module.roi_head.loss(dict_net)
                        loss += roi_loss
                else:
                    loss = self.network.head.loss(dict_net)
                    if hasattr(self.network, 'point_head'): # PVRCNN_PP
                        point_loss = self.network.point_head.loss(dict_net)
                        loss += point_loss
                    if hasattr(self.network, 'roi_head'): # PVRCNN_PP
                        roi_loss = self.network.roi_head.loss(dict_net)
                        loss += roi_loss 
                
                if 'contrastive_loss' in dict_net:
                    loss += dict_net['contrastive_loss']

                weather_aux_loss, weather_aux_logs = self._compute_weather_aux_loss(dict_net, dict_datum)
                if weather_aux_loss is not None:
                    loss += weather_aux_loss
                    if 'logging' not in dict_net:
                        dict_net['logging'] = {}
                    dict_net['logging'].update(weather_aux_logs)

                weather_div_loss, weather_div_logs = self._compute_weather_diversity_loss(dict_net, dict_datum)
                if weather_div_loss is not None:
                    loss += weather_div_loss
                    if 'logging' not in dict_net:
                        dict_net['logging'] = {}
                    dict_net['logging'].update(weather_div_logs)

                branch_entropy_loss, branch_entropy_logs = self._compute_branch_entropy_loss(dict_net)
                if branch_entropy_loss is not None:
                    loss += branch_entropy_loss
                    if 'logging' not in dict_net:
                        dict_net['logging'] = {}
                    dict_net['logging'].update(branch_entropy_logs)
                
                try:
                    log_avg_loss = loss.cpu().detach().item()
                except:
                    log_avg_loss = loss
                avg_loss.append(log_avg_loss)

                if torch.isfinite(loss):
                    loss.backward()
                else:
                    print('* Exception error (pipeline): nan or inf loss happend')
                    print('* Meta: ', dict_datum['meta'])

                self.optimizer.step()
                if not (self.scheduler is None):
                    self.scheduler.step()
                
                self.optimizer.zero_grad()

                if self.is_logging:
                    dict_logging = dict_net.get('logging', {})
                    idx_log_iter +=1
                    for k, v in dict_logging.items():
                        self.log_train_iter.add_scalar(f'train/{k}', v, idx_log_iter)
                    if not (self.scheduler is None):
                        lr = self.scheduler.get_last_lr()
                        self.log_train_iter.add_scalar(f'train/learning_rate', lr[0], idx_log_iter)

            branch_summary = self._summarize_branch_weight_monitor(branch_monitor)
            if branch_summary is not None:
                if self.local_rank == 0 or not self.is_distributed:
                    self._print_branch_weight_summary(epoch, branch_summary)
                    self._save_branch_weight_summary(epoch, branch_summary)
                if self.is_logging and (self.local_rank == 0 or not self.is_distributed):
                    means = branch_summary['branch_mean']
                    dom = branch_summary['dominance_ratio']
                    self.log_train_epoch.add_scalar('train/branch_weight_lidar_epoch_mean', means[0], epoch)
                    self.log_train_epoch.add_scalar('train/branch_weight_radar_epoch_mean', means[1], epoch)
                    self.log_train_epoch.add_scalar('train/branch_weight_fusion_epoch_mean', means[2], epoch)
                    self.log_train_epoch.add_scalar('train/branch_dom_lidar_epoch_ratio', dom[0], epoch)
                    self.log_train_epoch.add_scalar('train/branch_dom_radar_epoch_ratio', dom[1], epoch)
                    self.log_train_epoch.add_scalar('train/branch_dom_fusion_epoch_ratio', dom[2], epoch)
                    self.log_train_epoch.add_scalar('train/branch_entropy_epoch_mean', branch_summary['entropy_mean'], epoch)
                    self.log_train_epoch.add_scalar('train/branch_max_weight_epoch_mean', branch_summary['max_weight_mean'], epoch)

            if getattr(self, 'is_save_model', False):
                # epoch: indexing from 0
                path_dict_model = os.path.join(self.path_log, 'models', f'model_{epoch}.pt')
                path_dict_util = os.path.join(self.path_log, 'utils', f'util_{epoch}.pt')

                if (epoch+1) % self.interval_epoch_model == 0:
                    model_to_save = self.network.module if self.is_distributed else self.network
                    torch.save(model_to_save.state_dict(), path_dict_model)
                if (epoch+1) % self.interval_epoch_util == 0:
                    model_to_save = self.network.module if self.is_distributed else self.network
                    dict_util = {
                        'epoch': epoch,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'idx_log_iter': idx_log_iter, 
                    }
                    if not (self.scheduler is None):
                        dict_util.update({'scheduler_state_dict': self.scheduler.state_dict()})
                    torch.save(dict_util, path_dict_util)

            if self.is_logging:
                self.log_train_epoch.add_scalar(f'train/avg_loss', np.mean(avg_loss), epoch)

            if self.is_distributed:
                dist.barrier()
            if self.is_validate:
                self.network.training=False
                if self.is_consider_subset:
                    if ((epoch + 1) % self.val_per_epoch_subset) == 0:
                        self.validate_kitti(epoch, list_conf_thr=self.list_val_conf_thr, is_subset=True)
                run_standard_full_eval = False
                if self.val_full_only_at_end:
                    run_standard_full_eval = ((epoch + 1) == epoch_end)
                else:
                    run_standard_full_eval = (((epoch + 1) % self.val_per_epoch_full) == 0)

                if run_standard_full_eval:
                    if self.skip_standard_full_when_conditional and self.run_conditional_at_end:
                        if self.local_rank == 0 or not self.is_distributed:
                            print('* Skip standard full validation: conditional full validation will run once at training end')
                    else:
                        self.validate_kitti(epoch, list_conf_thr=self.list_val_conf_thr)

            if self.is_distributed:
                dist.barrier()

    def _init_branch_weight_monitor(self, num_conditions=0, num_bins=50):
        monitor = {
            'num_bins': int(num_bins),
            'num_conditions': int(max(0, num_conditions)),
            'count': torch.zeros(1, device=self.device, dtype=torch.float32),
            'weight_sum': torch.zeros(3, device=self.device, dtype=torch.float32),
            'weight_sq_sum': torch.zeros(3, device=self.device, dtype=torch.float32),
            'dominance_count': torch.zeros(3, device=self.device, dtype=torch.float32),
            'entropy_sum': torch.zeros(1, device=self.device, dtype=torch.float32),
            'entropy_sq_sum': torch.zeros(1, device=self.device, dtype=torch.float32),
            'max_weight_sum': torch.zeros(1, device=self.device, dtype=torch.float32),
            'hist': torch.zeros(3, int(num_bins), device=self.device, dtype=torch.float32),
        }
        if monitor['num_conditions'] > 0:
            monitor['condition_count'] = torch.zeros(
                monitor['num_conditions'], device=self.device, dtype=torch.float32
            )
            monitor['condition_weight_sum'] = torch.zeros(
                monitor['num_conditions'], 3, device=self.device, dtype=torch.float32
            )
        return monitor

    def _update_branch_weight_monitor(self, monitor, dict_net):
        if ('branch_weights' not in dict_net) or (dict_net['branch_weights'] is None):
            return {}

        branch_weights = dict_net['branch_weights']
        if not torch.is_tensor(branch_weights):
            return {}
        if branch_weights.ndim != 2 or branch_weights.shape[1] < 3:
            return {}

        branch_weights = branch_weights[:, :3].detach().to(self.device)
        branch_weights = branch_weights.clamp(min=0.0)
        branch_weights = branch_weights / branch_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

        batch_size = branch_weights.shape[0]
        if batch_size == 0:
            return {}

        monitor['count'] += float(batch_size)
        monitor['weight_sum'] += branch_weights.sum(dim=0)
        monitor['weight_sq_sum'] += (branch_weights * branch_weights).sum(dim=0)
        dominance = torch.argmax(branch_weights, dim=1)
        monitor['dominance_count'] += F.one_hot(dominance, num_classes=3).float().sum(dim=0)

        entropy = -(branch_weights * torch.log(branch_weights.clamp_min(1e-12))).sum(dim=1)
        entropy = entropy / np.log(3.0)
        monitor['entropy_sum'] += entropy.sum()
        monitor['entropy_sq_sum'] += (entropy * entropy).sum()
        max_weight = branch_weights.max(dim=1).values
        monitor['max_weight_sum'] += max_weight.sum()

        num_bins = monitor['num_bins']
        bin_indices = torch.clamp((branch_weights * num_bins).long(), min=0, max=num_bins - 1)
        ones = torch.ones(batch_size, device=self.device, dtype=torch.float32)
        for idx_branch in range(3):
            monitor['hist'][idx_branch].index_add_(0, bin_indices[:, idx_branch], ones)

        if ('condition_ids' in dict_net) and ('condition_count' in monitor):
            condition_ids = dict_net['condition_ids']
            if not torch.is_tensor(condition_ids):
                condition_ids = torch.tensor(condition_ids, dtype=torch.long, device=self.device)
            condition_ids = condition_ids.long().view(-1).to(self.device)
            valid = (condition_ids >= 0) & (condition_ids < monitor['num_conditions'])
            if torch.any(valid):
                valid_ids = condition_ids[valid]
                valid_weights = branch_weights[valid]
                valid_ones = torch.ones(valid_ids.shape[0], device=self.device, dtype=torch.float32)
                monitor['condition_count'].index_add_(0, valid_ids, valid_ones)
                monitor['condition_weight_sum'].index_add_(0, valid_ids, valid_weights)

        batch_log = {
            'branch_weight_lidar': float(branch_weights[:, 0].mean().item()),
            'branch_weight_radar': float(branch_weights[:, 1].mean().item()),
            'branch_weight_fusion': float(branch_weights[:, 2].mean().item()),
            'branch_entropy': float(entropy.mean().item()),
            'branch_max_weight': float(max_weight.mean().item()),
        }
        return batch_log

    def _merge_branch_weight_monitor_ddp(self, monitor):
        if not self.is_distributed:
            return
        tensor_keys = [
            'count', 'weight_sum', 'weight_sq_sum', 'dominance_count',
            'entropy_sum', 'entropy_sq_sum', 'max_weight_sum', 'hist'
        ]
        for key in tensor_keys:
            dist.all_reduce(monitor[key], op=dist.ReduceOp.SUM)
        if 'condition_count' in monitor:
            dist.all_reduce(monitor['condition_count'], op=dist.ReduceOp.SUM)
            dist.all_reduce(monitor['condition_weight_sum'], op=dist.ReduceOp.SUM)

    def _hist_quantile(self, hist_row, q):
        total = hist_row.sum().item()
        if total <= 0:
            return float('nan')
        cdf = torch.cumsum(hist_row, dim=0) / total
        idx = int(torch.searchsorted(cdf, torch.tensor(q, device=cdf.device), right=False).item())
        idx = max(0, min(idx, hist_row.shape[0] - 1))
        return float((idx + 0.5) / hist_row.shape[0])

    def _extract_weather_from_prompt(self, prompt):
        if not isinstance(prompt, str):
            return 'unknown'
        prompt = prompt.strip()
        # Prompt format: "A {weather} driving scene ..."
        match = re.search(r'^A\s+(.+?)\s+driving scene', prompt, flags=re.IGNORECASE)
        if match is None:
            return 'unknown'
        weather = match.group(1).strip().lower().replace(' ', '')
        return weather if len(weather) > 0 else 'unknown'

    def _get_condition_weather_names(self, num_conditions):
        weather_names = ['unknown'] * int(max(num_conditions, 0))
        if self.dataset_train is None:
            return weather_names

        list_condition_tuple = getattr(self.dataset_train, 'list_condition_tuple', None)
        condition_level = str(getattr(self.dataset_train, 'condition_level', '')).lower()
        if isinstance(list_condition_tuple, list) and len(list_condition_tuple) >= num_conditions:
            for idx in range(num_conditions):
                condition_tuple = list_condition_tuple[idx]
                weather = 'unknown'
                if isinstance(condition_tuple, (tuple, list)):
                    if (condition_level == 'weather_time') and (len(condition_tuple) >= 1):
                        weather = str(condition_tuple[0]).strip().lower()
                    elif len(condition_tuple) >= 2:
                        weather = str(condition_tuple[1]).strip().lower()
                weather_names[idx] = weather if len(weather) > 0 else 'unknown'
            return weather_names

        prompt_vocab = getattr(self.dataset_train, 'condition_prompt_vocab', None)
        if isinstance(prompt_vocab, list):
            for idx in range(min(num_conditions, len(prompt_vocab))):
                weather_names[idx] = self._extract_weather_from_prompt(prompt_vocab[idx])
        return weather_names

    def _summarize_branch_weight_monitor(self, monitor):
        self._merge_branch_weight_monitor_ddp(monitor)
        sample_count = float(monitor['count'].item())
        if sample_count <= 0:
            return None

        mean = monitor['weight_sum'] / sample_count
        var = monitor['weight_sq_sum'] / sample_count - mean * mean
        std = torch.sqrt(torch.clamp(var, min=0.0))
        dom_ratio = monitor['dominance_count'] / sample_count
        entropy_mean = float((monitor['entropy_sum'] / sample_count).item())
        entropy_var = float((monitor['entropy_sq_sum'] / sample_count).item()) - entropy_mean * entropy_mean
        entropy_std = float(np.sqrt(max(entropy_var, 0.0)))
        max_weight_mean = float((monitor['max_weight_sum'] / sample_count).item())

        quantiles = []
        for idx_branch in range(3):
            hist_row = monitor['hist'][idx_branch]
            quantiles.append({
                'p10': self._hist_quantile(hist_row, 0.10),
                'p50': self._hist_quantile(hist_row, 0.50),
                'p90': self._hist_quantile(hist_row, 0.90),
            })

        condition_topk = []
        weather_breakdown = []
        if 'condition_count' in monitor:
            cond_count = monitor['condition_count']
            non_zero_idx = torch.where(cond_count > 0)[0]
            if len(non_zero_idx) > 0:
                topk = min(6, len(non_zero_idx))
                top_vals, top_pos = torch.topk(cond_count[non_zero_idx], k=topk, largest=True)
                top_cond_idx = non_zero_idx[top_pos]
                prompt_vocab = getattr(self.dataset_train, 'condition_prompt_vocab', None)
                for cidx, cnum in zip(top_cond_idx.tolist(), top_vals.tolist()):
                    cond_mean_w = (monitor['condition_weight_sum'][cidx] / max(cnum, 1e-12)).tolist()
                    cond_name = str(cidx)
                    if prompt_vocab is not None and cidx < len(prompt_vocab):
                        cond_name = prompt_vocab[cidx]
                    condition_topk.append({
                        'condition_id': int(cidx),
                        'condition_name': cond_name,
                        'count': int(round(cnum)),
                        'mean_weights': [float(cond_mean_w[0]), float(cond_mean_w[1]), float(cond_mean_w[2])],
                    })

            weather_names = self._get_condition_weather_names(cond_count.shape[0])
            weather_count = {}
            weather_weight_sum = {}
            for cidx in range(cond_count.shape[0]):
                cnum = float(cond_count[cidx].item())
                if cnum <= 0:
                    continue
                weather = weather_names[cidx] if cidx < len(weather_names) else 'unknown'
                weather = weather if len(weather) > 0 else 'unknown'
                weather_count[weather] = weather_count.get(weather, 0.0) + cnum
                if weather not in weather_weight_sum:
                    weather_weight_sum[weather] = monitor['condition_weight_sum'][cidx].clone()
                else:
                    weather_weight_sum[weather] += monitor['condition_weight_sum'][cidx]

            preferred_weather = []
            weather_list = getattr(self.dataset_train, 'weather_list', None)
            if isinstance(weather_list, list):
                preferred_weather = [str(w).strip().lower() for w in weather_list]

            emitted = set()
            for weather in preferred_weather:
                cnum = weather_count.get(weather, 0.0)
                if cnum > 0:
                    mean_w = (weather_weight_sum[weather] / max(cnum, 1e-12)).tolist()
                else:
                    mean_w = [0.0, 0.0, 0.0]
                weather_breakdown.append({
                    'weather': weather,
                    'count': int(round(cnum)),
                    'mean_weights': [float(mean_w[0]), float(mean_w[1]), float(mean_w[2])],
                })
                emitted.add(weather)

            for weather in sorted(weather_count.keys()):
                if weather in emitted:
                    continue
                cnum = weather_count[weather]
                mean_w = (weather_weight_sum[weather] / max(cnum, 1e-12)).tolist()
                weather_breakdown.append({
                    'weather': weather,
                    'count': int(round(cnum)),
                    'mean_weights': [float(mean_w[0]), float(mean_w[1]), float(mean_w[2])],
                })

        summary = {
            'sample_count': int(round(sample_count)),
            'branch_mean': [float(mean[0].item()), float(mean[1].item()), float(mean[2].item())],
            'branch_std': [float(std[0].item()), float(std[1].item()), float(std[2].item())],
            'dominance_ratio': [float(dom_ratio[0].item()), float(dom_ratio[1].item()), float(dom_ratio[2].item())],
            'branch_quantiles': quantiles,
            'entropy_mean': entropy_mean,
            'entropy_std': entropy_std,
            'max_weight_mean': max_weight_mean,
            'condition_topk': condition_topk,
            'weather_breakdown': weather_breakdown,
        }
        return summary

    def _print_branch_weight_summary(self, epoch, summary):
        names = ['Lidar', 'Radar', 'Fusion']
        print(f"[Branch Weight Monitor][Epoch {epoch}] samples={summary['sample_count']}")
        for idx_name, name in enumerate(names):
            q = summary['branch_quantiles'][idx_name]
            print(
                f"  {name:6s} mean={summary['branch_mean'][idx_name]:.4f} std={summary['branch_std'][idx_name]:.4f} "
                f"p10={q['p10']:.3f} p50={q['p50']:.3f} p90={q['p90']:.3f} "
                f"dom_ratio={summary['dominance_ratio'][idx_name]:.4f}"
            )
        print(
            f"  Entropy(mean/std)={summary['entropy_mean']:.4f}/{summary['entropy_std']:.4f} "
            f"MaxWeight(mean)={summary['max_weight_mean']:.4f}"
        )
        if summary['condition_topk']:
            print("  Top conditions by sample count:")
            for item in summary['condition_topk']:
                w = item['mean_weights']
                cond_name = item['condition_name']
                if len(cond_name) > 80:
                    cond_name = cond_name[:77] + '...'
                print(
                    f"    id={item['condition_id']:2d} count={item['count']:4d} "
                    f"L/R/F={w[0]:.4f}/{w[1]:.4f}/{w[2]:.4f} | {cond_name}"
                )
        if summary.get('weather_breakdown'):
            print("  Weather breakdown (aggregated over road/time):")
            for item in summary['weather_breakdown']:
                w = item['mean_weights']
                print(
                    f"    {item['weather']:10s} count={item['count']:4d} "
                    f"L/R/F={w[0]:.4f}/{w[1]:.4f}/{w[2]:.4f}"
                )


    def _save_branch_weight_summary(self, epoch, summary):
        if not hasattr(self, 'path_log') or self.path_log is None:
            return

        out_dir = os.path.join(self.path_log, 'branch_monitor')
        os.makedirs(out_dir, exist_ok=True)

        summary_json_path = os.path.join(out_dir, f'epoch_{epoch:03d}_summary.json')
        weather_csv_path = os.path.join(out_dir, f'epoch_{epoch:03d}_weather.csv')
        summary_csv_path = os.path.join(out_dir, 'branch_epoch_summary.csv')
        weather_all_csv_path = os.path.join(out_dir, 'branch_weather_summary.csv')

        payload = {
            'epoch': int(epoch),
            'sample_count': int(summary['sample_count']),
            'branch_mean': [float(x) for x in summary['branch_mean']],
            'branch_std': [float(x) for x in summary['branch_std']],
            'dominance_ratio': [float(x) for x in summary['dominance_ratio']],
            'branch_quantiles': summary.get('branch_quantiles', []),
            'entropy_mean': float(summary['entropy_mean']),
            'entropy_std': float(summary['entropy_std']),
            'max_weight_mean': float(summary['max_weight_mean']),
            'condition_topk': summary.get('condition_topk', []),
            'weather_breakdown': summary.get('weather_breakdown', []),
        }
        with open(summary_json_path, 'w') as f:
            json.dump(payload, f, indent=2)

        with open(weather_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['epoch', 'weather', 'count', 'weight_lidar', 'weight_radar', 'weight_fusion']
            )
            writer.writeheader()
            for item in summary.get('weather_breakdown', []):
                w = item['mean_weights']
                writer.writerow({
                    'epoch': int(epoch),
                    'weather': item['weather'],
                    'count': int(item['count']),
                    'weight_lidar': float(w[0]),
                    'weight_radar': float(w[1]),
                    'weight_fusion': float(w[2]),
                })

        summary_row = {
            'epoch': int(epoch),
            'sample_count': int(summary['sample_count']),
            'branch_lidar': float(summary['branch_mean'][0]),
            'branch_radar': float(summary['branch_mean'][1]),
            'branch_fusion': float(summary['branch_mean'][2]),
            'branch_std_lidar': float(summary['branch_std'][0]),
            'branch_std_radar': float(summary['branch_std'][1]),
            'branch_std_fusion': float(summary['branch_std'][2]),
            'dom_lidar': float(summary['dominance_ratio'][0]),
            'dom_radar': float(summary['dominance_ratio'][1]),
            'dom_fusion': float(summary['dominance_ratio'][2]),
            'entropy_mean': float(summary['entropy_mean']),
            'entropy_std': float(summary['entropy_std']),
            'max_weight_mean': float(summary['max_weight_mean']),
        }
        summary_fieldnames = list(summary_row.keys())
        write_header = not os.path.exists(summary_csv_path)
        with open(summary_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(summary_row)

        weather_fieldnames = ['epoch', 'weather', 'count', 'weight_lidar', 'weight_radar', 'weight_fusion']
        write_weather_header = not os.path.exists(weather_all_csv_path)
        with open(weather_all_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=weather_fieldnames)
            if write_weather_header:
                writer.writeheader()
            for item in summary.get('weather_breakdown', []):
                w = item['mean_weights']
                writer.writerow({
                    'epoch': int(epoch),
                    'weather': item['weather'],
                    'count': int(item['count']),
                    'weight_lidar': float(w[0]),
                    'weight_radar': float(w[1]),
                    'weight_fusion': float(w[2]),
                })

    def load_dict_model(self, path_dict_model, is_strict=False):
        pt_dict_model = torch.load(path_dict_model)
        self.network.load_state_dict(pt_dict_model, strict=is_strict)

    # V2
    def vis_infer(self, sample_indices, conf_thr=0.7, is_nms=True, vis_mode=['lpc', 'spcube', 'cube'], is_train=False):
        '''
        * sample_indices: e.g. [0, 1, 2, 3, 4]
        * assume batch_size = 1 for convenience
        * vis_mode (TBD)
        '''
        model_to_eval = self.network.module if self.is_distributed else self.network
        model_to_eval.eval()
        
        if is_train:
            dataset_loaded = self.dataset_train
        else:
            dataset_loaded = self.dataset_test
        subset = Subset(dataset_loaded, sample_indices)
        data_loader = torch.utils.data.DataLoader(subset,
                batch_size = 1, shuffle = False,
                collate_fn = self.dataset_test.collate_fn,
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        
        for dict_datum in data_loader:
            dict_out = self.network(dict_datum)
            model_to_eval = self.network.module if self.is_distributed else self.network
            dict_out = model_to_eval.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms)
            ### Vis data ###
            pc_lidar = dict_datum['ldr_pc_64']
            # rdr_spcube = dict_datum['rdr_sparse_cube']
            # rdr_cube = dict_datum['rdr_cube']
            ### Vis data ###

            ### Labels ###
            labels = dict_out['label'][0]
            list_obj_label = []
            for label_obj in labels:
                cls_name, cls_id, (xc, yc, zc, rot, xl, yl, zl), obj_idx = label_obj
                obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                list_obj_label.append(obj)
            ### Labels ###

            ### Preds: post processing bbox ###
            list_obj_pred = []
            list_cls_pred = []
            if dict_datum['pp_num_bbox'] == 0:
                pass
            else:
                pp_cls = dict_datum['pp_cls']
                for idx_pred, pred_obj in enumerate(dict_datum['pp_bbox']):
                    conf_score, xc, yc, zc, xl, yl, zl, rot = pred_obj
                    obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                    list_obj_pred.append(obj)
                    list_cls_pred.append(self.dict_cls_id_to_name[pp_cls[idx_pred]])
            ### Preds: post processing bbox ###

            ### Vis for open3d ###
            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [6, 7], #[5, 6],[4, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    [0, 2], [1, 3], [4, 6], [5, 7]]
            colors_label = [[0, 0, 0] for _ in range(len(lines))]
            list_line_set_label = []
            list_line_set_pred = []
            for label_obj in list_obj_label:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(label_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors_label)
                list_line_set_label.append(line_set)
            
            for idx_pred, pred_obj in enumerate(list_obj_pred):
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(pred_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                colors_pred = [self.dict_cls_name_to_rgb[list_cls_pred[idx_pred]] for _ in range(len(lines))]
                line_set.colors = o3d.utility.Vector3dVector(colors_pred)
                list_line_set_pred.append(line_set)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
            o3d.visualization.draw_geometries([pcd] + list_line_set_label + list_line_set_pred)
            ### Vis for open3d ###

        return list_obj_label, list_obj_pred
    

    #############        
    def func_show_rlbev_bbox(self, dict_item):
        # radar bev
        # rdr_cube, mask, rdr_cube_cnt = dataset_loaded.get_cube(dict_item['meta'][0]['path']['rdr_cube'], mode=0)
        rdr_cube, rdr_cube_cnt = dict_item['rdr_cube'][0].numpy(), dict_item['rdr_cube_cnt'][0].numpy()
        # arr_x, arr_y, arr_z = p_pline.arr_x_cb, p_pline.arr_y_cb, p_pline.arr_z_cb
        arr_x, arr_y, arr_z = np.arange(0, 99.2, 0.4), np.arange(-40, 40, 0.4), np.arange(-2, 6.0, 0.4)
        num_z, num_y, num_x = rdr_cube.shape
        rdr_cube_bev = np.sum(rdr_cube, axis=0) # 200, 248
        rdr_cube_bev = rdr_cube_bev/rdr_cube_cnt
        rdr_cube_bev = np.maximum(rdr_cube_bev, 1.)
        rdr_cube_bev = 10*np.log10(rdr_cube_bev)
        rdr_cube_bev = rdr_cube_bev/np.max(rdr_cube_bev)
        arr_0, arr_1 = np.meshgrid(arr_x, arr_y)
              
        rdr_cube_bev_vis = cv2.cvtColor((rdr_cube_bev*255.).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        bboxes = dict_item['label'][0]
        list_line_order = [[0,1], [0,2], [1,3], [2,3]]
        lthick = 1
        alpha = 0.5

        rdr_cube_bev[np.where(rdr_cube_bev==0.)] = -np.inf # for visualization
        plt.figure(figsize=(2.48, 2))
        plt.pcolormesh(arr_0, arr_1, rdr_cube_bev, cmap='jet')
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.savefig('vis/lidar_bev/radar_bev.png', pad_inches=0, dpi=100) # 1200, 1488, 3
        plt.close()
        
        # lidar bev9
        # pc_lidar = dataset_loaded.get_pc_lidar(dict_item['meta'][0]['path']['ldr_pc_64'], dict_item['calib'][0])
        pc_lidar = dict_item['ldr_pc_64']
        lpc_roi = [0,100,-40,40,-10,60] # dict_item['meta']['default_roi'] # [0,98.8, -6.4, 6.4, -2, 6.0]
        lpc_roi[1] = 99
        pc_lidar = pc_lidar[
            np.where(
                (pc_lidar[:, 0] > lpc_roi[0]) & (pc_lidar[:, 0] < lpc_roi[1]) &
                (pc_lidar[:, 1] > lpc_roi[2]) & (pc_lidar[:, 1] < lpc_roi[3]) &
                (pc_lidar[:, 2] > lpc_roi[4]) & (pc_lidar[:, 2] < lpc_roi[5])
            )]

        # 200:248 -> 80:100
        rdr_cube = cv2.imread('vis/lidar_bev/radar_bev.png')
        lidar_cube = (np.ones((200, 248, 3)) * 255).astype(np.uint8)
        for i in range(len(pc_lidar)):
            lidar_cube[int((pc_lidar[i, 1]+40)*2.5), int(pc_lidar[i, 0]*2.5), 0] = 0
            lidar_cube[int((pc_lidar[i, 1]+40)*2.5), int(pc_lidar[i, 0]*2.5), 1] = 0
            lidar_cube[int((pc_lidar[i, 1]+40)*2.5), int(pc_lidar[i, 0]*2.5), 2] = 0
        lidar_cube = cv2.flip(lidar_cube, 0)
        lr_bev = cv2.addWeighted(rdr_cube, alpha, lidar_cube, 1-alpha, 0)
        lr_bev = cv2.flip(lr_bev, 0)
        # cv2.imwrite('vis/12200_lidarbev.png', lidar_cube)
 

        for bbox in bboxes:
            _, _, [x, y, z, theta, xl, yl, zl], _ = bbox
            obj3d = Object3D(x, y, z, xl, yl, zl, theta)
            idx_x = np.argmin(np.abs(arr_x-x))
            idx_y = np.argmin(np.abs(arr_y-y))
            pts = [obj3d.corners[0,:], obj3d.corners[2,:], obj3d.corners[4,:], obj3d.corners[6,:]]
            pt_list = []
            for pt in pts:
                idx_x = np.argmin(np.abs(arr_x-pt[0]))
                idx_y = np.argmin(np.abs(arr_y-pt[1]))
                pt_list.append([idx_x, idx_y])
            for idx_1, idx_2 in list_line_order:
                p1_x, p1_y = pt_list[idx_1]
                p2_x, p2_y = pt_list[idx_2]
                color = (0, 0, 255)
                lr_bev = cv2.line(lr_bev, (p1_x,p1_y), (p2_x,p2_y), color, thickness=2)
                    
        preds = dict_item['pp_bbox']
        if preds is None:
            pass
        else:
            for pred in preds:
                conf_score, x, y, z, xl, yl, zl, theta = pred
                obj3d = Object3D(x, y, z, xl, yl, zl, theta)
                idx_x = np.argmin(np.abs(arr_x-x))
                idx_y = np.argmin(np.abs(arr_y-y))
                pts = [obj3d.corners[0,:], obj3d.corners[2,:], obj3d.corners[4,:], obj3d.corners[6,:]]
                pt_list = []
                for pt in pts:
                    idx_x = np.argmin(np.abs(arr_x-pt[0]))
                    idx_y = np.argmin(np.abs(arr_y-pt[1]))
                    pt_list.append([idx_x, idx_y])
                for idx_1, idx_2 in list_line_order:
                    p1_x, p1_y = pt_list[idx_1]
                    p2_x, p2_y = pt_list[idx_2]
                    color = (0, 0, 0)
                    lr_bev = cv2.line(lr_bev, (p1_x,p1_y), (p2_x,p2_y), color, thickness=lthick)

        lr_bev = cv2.flip(lr_bev, 0)
        lr_bev = lr_bev.transpose((1, 0, 2))
        lr_bev = cv2.flip(lr_bev, 0)
        # cv2.imwrite('vis/lidar_bev/lidar_radar_bev_bbox.png', lr_bev) # 200, 248, 3 
        return lr_bev

    # V2
    def validate_kitti(self, epoch=None, list_conf_thr=None, is_subset=False):
        if self.is_distributed and self.local_rank != 0:
            return None

        model_to_eval = self.network.module if self.is_distributed else self.network
        model_to_eval.eval()

        with torch.no_grad():
            ### Check is_validate with small dataset ###
            if is_subset:
                is_shuffle = False
                # minival_id = list(range(0, 17500, 3)) # num:5834
                minival_id = list(range(0, len(self.dataset_test), 3))
                tqdm_bar = tqdm(total=len(minival_id), desc='* MiniVal (Subset): ')
                log_header = 'minival'
                dataset_test1 = Subset(self.dataset_test, minival_id)
                data_loader = torch.utils.data.DataLoader(dataset_test1, \
                    batch_size=1, shuffle=is_shuffle, collate_fn=self.dataset_test.collate_fn, \
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS) 
            else:
                is_shuffle = False
                tqdm_bar = tqdm(total=len(self.dataset_test), desc='* Test (Total): ')
                log_header = 'val_tot'
                data_loader = torch.utils.data.DataLoader(self.dataset_test, \
                    batch_size=1, shuffle=is_shuffle, collate_fn=self.dataset_test.collate_fn, \
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
            list_val_loss = []
            
            if epoch is None:
                dir_epoch = 'none'
            else:
                dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

            # initialize via VAL.LIST_VAL_CONF_THR
            path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
            # print(path_dir)
            for conf_thr in list_conf_thr:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + 'val.txt', 'w') as f:
                    f.write('')
                f.close()

            for idx_datum, dict_datum in enumerate(data_loader):
                if (idx_datum % 50) == 49:
                    torch.cuda.empty_cache()
                if is_subset & (idx_datum >= self.val_num_subset):
                    break
                
                dict_out = dict_datum
                try:
                    dict_datum['idx_iter'] = idx_datum
                    dict_datum['local_rank'] = self.local_rank
                    dict_out = model_to_eval(dict_datum) # inference
                    is_feature_inferenced = True
                    # Track validation loss curve (without grad) for later plotting.
                    try:
                        loss_val = None
                        if hasattr(model_to_eval, 'head'):
                            loss_val = model_to_eval.head.loss(dict_out)
                            if hasattr(model_to_eval, 'point_head'):
                                loss_val += model_to_eval.point_head.loss(dict_out)
                            if hasattr(model_to_eval, 'roi_head'):
                                loss_val += model_to_eval.roi_head.loss(dict_out)
                            if ('contrastive_loss' in dict_out) and torch.is_tensor(dict_out['contrastive_loss']):
                                loss_val += dict_out['contrastive_loss']

                        if loss_val is not None:
                            if torch.is_tensor(loss_val):
                                list_val_loss.append(float(loss_val.detach().cpu().item()))
                            else:
                                list_val_loss.append(float(loss_val))
                    except Exception:
                        pass
                except:
                    print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                    # print('* Meta info: ', dict_out['meta'])
                    is_feature_inferenced = False

                idx_name = str(idx_datum).zfill(6)

                ### for every conf in list_conf_thr ###
                for conf_thr in list_conf_thr:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + 'val.txt'
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)
                        
                    if is_feature_inferenced:
                        dict_out = model_to_eval.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                    else:
                        dict_out = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)

                    if dict_out is None:
                        print('* Exception error (Pipeline): dict_item is None in validation')
                        continue

                    dict_out = dict_datum_to_kitti(self, dict_out)

                    if len(dict_out['kitti_gt']) == 0: # no eval for emptry obj label
                        pass
                    else:
                        ### Gt ###
                        for idx_label, label in enumerate(dict_out['kitti_gt']):
                            open_mode = 'w' if idx_label == 0 else 'a'
                            with open(labels_dir + '/' + idx_name + '.txt', open_mode) as f:
                                f.write(label+'\n')
                        ### Gt ###

                        ### Process description ###
                        with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out['kitti_desc'])
                        ### Process description ###

                        ### Pred: do not care len 0 with if else: already care as dummy ###
                        for idx_pred, pred in enumerate(dict_out['kitti_pred']):
                            open_mode = 'w' if idx_pred == 0 else 'a'
                            with open(preds_dir + '/' + idx_name + '.txt', open_mode) as f:
                                f.write(pred+'\n')
                        ### Pred: do not care len 0 with if else: already care as dummy ###

                        str_log = idx_name + '\n'
                        with open(split_path, 'a') as f:
                            f.write(str_log)
                
                tqdm_bar.update(1)
            tqdm_bar.close()

            if self.is_logging and (len(list_val_loss) > 0):
                loss_step = 0 if epoch is None else epoch
                mean_val_loss = float(np.mean(list_val_loss))
                self.log_test.add_scalar(f'{log_header}/loss', mean_val_loss, loss_step)
                print(f'* {log_header} mean loss: {mean_val_loss:.6f} (n={len(list_val_loss)})')

            ### Validate per conf ###
            for conf_thr in list_conf_thr:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
                split_path = path_dir + f'/{conf_thr}/' + 'val.txt'

                dt_annos = kitti.get_label_annos(preds_dir)
                val_ids = read_imageset_file(split_path)
                gt_annos = kitti.get_label_annos(labels_dir, val_ids)

                list_metrics = []
                for idx_cls_val in self.list_val_care_idx:
                    dict_metrics, result_log = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                    print(f'-----conf{conf_thr}-----')
                    print(result_log)
                    list_metrics.append(dict_metrics)

                for dict_metrics in list_metrics:
                    cls_name = dict_metrics['cls']
                    ious = dict_metrics['iou']
                    bevs = dict_metrics['bev']
                    ap3ds = dict_metrics['3d']
                    self.log_test.add_scalars(f'{log_header}/BEV_conf_thr_{conf_thr}', {
                        f'iou_{ious[0]}_{cls_name}': bevs[0],
                        f'iou_{ious[1]}_{cls_name}': bevs[1],
                        f'iou_{ious[2]}_{cls_name}': bevs[2],
                    }, epoch)
                    self.log_test.add_scalars(f'{log_header}/3D_conf_thr_{conf_thr}', {
                        f'iou_{ious[0]}_{cls_name}': ap3ds[0],
                        f'iou_{ious[1]}_{cls_name}': ap3ds[1],
                        f'iou_{ious[2]}_{cls_name}': ap3ds[2],
                    }, epoch)
            ### Validate per conf ###
            return
            
    def validate_kitti_conditional(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False):
        if self.is_distributed and self.local_rank != 0:
            return
            
        model_to_eval = self.network.module if self.is_distributed else self.network
        model_to_eval.eval()

        with torch.no_grad():
            road_cond_list = ['urban', 'highway', 'countryside', 'alleyway', 'parkinglots', 'shoulder', 'mountain', 'university']
            time_cond_list = ['day', 'night']
            weather_cond_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']

            ### Check is_validate with small dataset ###
            if is_subset:
                is_shuffle = False
                minival_id = list(range(0, 17500, 10)) # num:1750
                tqdm_bar = tqdm(total=len(minival_id), desc='* MiniVal (Subset): ')
                log_header = 'minival'
                dataset_test1 = Subset(self.dataset_test, minival_id)
                data_loader = torch.utils.data.DataLoader(dataset_test1, \
                    batch_size=1, shuffle=is_shuffle, collate_fn=self.dataset_test.collate_fn, \
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS) 
            else:
                is_shuffle = False
                tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')
                data_loader = torch.utils.data.DataLoader(self.dataset_test, \
                        batch_size = 1, shuffle = is_shuffle, collate_fn = self.dataset_test.collate_fn, \
                        num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
            
            if epoch is None:
                dir_epoch = 'none'
            else:
                dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

            # initialize via VAL.LIST_VAL_CONF_THR
            path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
            for conf_thr in list_conf_thr:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)

                os.makedirs(os.path.join(path_dir, f'{conf_thr}', 'all'), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + 'all/val.txt', 'w') as f:
                    f.write('')

                for road_cond in road_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', road_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + road_cond + '/val.txt', 'w') as f:
                        f.write('')

                for time_cond in time_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', time_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + time_cond + '/val.txt', 'w') as f:
                        f.write('')

                for weather_cond in weather_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', weather_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + weather_cond + '/val.txt', 'w') as f:
                        f.write('')

                pred_dir_list = []
                label_dir_list = []
                desc_dir_list = []
                split_path_list = []

                ### For All Conditions ###
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
                                
                ### For Specific Conditions ###
                for road_cond in road_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + road_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)
                    
                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)
                
                for time_cond in time_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + time_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)

                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)
                
                for weather_cond in weather_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + weather_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)

                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)

            # Creating gts and preds txt files for evaluation
            for idx_datum, dict_datum in enumerate(data_loader):
                if is_subset & (idx_datum >= self.val_num_subset):
                    break

                if (idx_datum % 1000) == 0:
                    torch.cuda.empty_cache()

                dict_datum['idx_iter'] = idx_datum
                dict_datum['local_rank'] = self.local_rank
                dict_out = dict_datum

                try:
                    dict_out = model_to_eval(dict_datum) # inference
                    is_feature_inferenced = True
                except:
                    print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                    print('* Meta info: ', dict_out['meta'])
                    is_feature_inferenced = False

                if is_print_memory:
                    print('max_memory: ', torch.cuda.max_memory_allocated(device='cuda'))
                    
                idx_name = str(idx_datum).zfill(6)
                
                road_cond_tag, time_cond_tag, weather_cond_tag = \
                    dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
                # print(dict_out['desc'][0])

                ### for every conf in list_conf_thr ###
                for conf_thr in list_conf_thr:
                    ### For All Conditions ###
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                    preds_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'preds')
                    labels_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'gts')
                    desc_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'desc')
                    split_path_road =path_dir + f'/{conf_thr}/' + road_cond_tag + '/val.txt'

                    preds_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'preds')
                    labels_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'gts')
                    desc_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'desc')
                    split_path_time = path_dir + f'/{conf_thr}/' + time_cond_tag + '/val.txt'

                    preds_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'preds')
                    labels_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'gts')
                    desc_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'desc')
                    split_path_weather =path_dir + f'/{conf_thr}/' + weather_cond_tag + '/val.txt'

                    os.makedirs(labels_dir_road, exist_ok=True)
                    os.makedirs(labels_dir_time, exist_ok=True)
                    os.makedirs(labels_dir_weather, exist_ok=True)
                    os.makedirs(desc_dir_road, exist_ok=True)
                    os.makedirs(desc_dir_time, exist_ok=True)
                    os.makedirs(desc_dir_weather, exist_ok=True)
                    os.makedirs(preds_dir_road, exist_ok=True)
                    os.makedirs(preds_dir_time, exist_ok=True)
                    os.makedirs(preds_dir_weather, exist_ok=True)

                    if is_feature_inferenced:
                        model_to_eval = self.network.module if self.is_distributed else self.network
                        dict_out_current = model_to_eval.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                    else:
                        dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)

                    if dict_out_current is None:
                        print('* Exception error (Pipeline): dict_item is None in validation')
                        continue

                    dict_out_current = dict_datum_to_kitti(self, dict_out_current)

                    if len(dict_out_current['kitti_gt']) == 0: # not eval emptry label
                        pass
                    else:
                        ### Gt ###
                        for idx_label, label in enumerate(dict_out_current['kitti_gt']):
                            if idx_label == 0:
                                mode = 'w'
                            else:
                                mode = 'a'

                            with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')
                            with open(labels_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')
                            with open(labels_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')
                            with open(labels_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')

                        ### Process description ###
                        with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])
                        with open(desc_dir_road + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])
                        with open(desc_dir_time + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])
                        with open(desc_dir_weather + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])

                        ### Process description ###
                        if len(dict_out_current['kitti_pred']) == 0:
                            with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                            with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                            with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                            with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                        else:
                            for idx_pred, pred in enumerate(dict_out_current['kitti_pred']):
                                if idx_pred == 0:
                                    mode = 'w'
                                else:
                                    mode = 'a'

                                with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')
                                with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')
                                with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')
                                with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')

                        str_log = idx_name + '\n'
                        with open(split_path, 'a') as f:
                            f.write(str_log)
                        with open(split_path_road, 'a') as f:
                            f.write(str_log)
                        with open(split_path_time, 'a') as f:
                            f.write(str_log)
                        with open(split_path_weather, 'a') as f:
                            f.write(str_log)
                tqdm_bar.update(1)
            tqdm_bar.close()

            ### Validate per conf ###
            all_condition_list = ['all'] + road_cond_list + time_cond_list + weather_cond_list
            for conf_thr in list_conf_thr:
                for condition in all_condition_list:
                    try:
                        preds_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'preds')
                        labels_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'gts')
                        desc_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'desc')
                        split_path = path_dir + f'/{conf_thr}/' + condition + '/val.txt'

                        dt_annos = kitti.get_label_annos(preds_dir)
                        val_ids = read_imageset_file(split_path)
                        gt_annos = kitti.get_label_annos(labels_dir, val_ids)
                        list_metrics = []
                        list_results = []
                        for idx_cls_val in self.list_val_care_idx:
                            dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                            list_metrics.append(dict_metrics)
                            list_results.append(result)
                        print('Conf thr: ', str(conf_thr), ', Condition: ', condition)
                        with open(os.path.join(path_dir, f'{conf_thr}', 'complete_results.txt'), 'a') as f:
                            for dic_metric in list_metrics:
                                print('='*50)
                                print('Cls: ', dic_metric['cls'])
                                print('IoU:', dic_metric['iou'])
                                print('BEV: ', dic_metric['bev'])
                                print('3D: ', dic_metric['3d'])
                                print('-'*50)
                                
                                f.write('Conf thr: ' + str(conf_thr) +  ', Condition: ' + condition + '\n')
                                f.write('cls: ' + dic_metric['cls'] + '\n')
                                f.write('iou: ')
                                for iou in dic_metric['iou']:
                                    f.write(str(iou) + ' ')
                                f.write('\n')
                                f.write('bev: ')
                                for bev in dic_metric['bev']:
                                    f.write(str(bev) + ' ')
                                f.write('\n')
                                f.write('3d  :')
                                for det3d in dic_metric['3d']:
                                    f.write(str(det3d) + ' ')
                                f.write('\n\n')
                        print('\n')
                    except:
                        print('* Exception error (Pipeline): Samples for the codition are not found')

            path_check = os.path.join(path_dir, 'Conf_thr', 'complete_results.txt')
            print(f'* Check {path_check}')
            ### Validate per conf ###

