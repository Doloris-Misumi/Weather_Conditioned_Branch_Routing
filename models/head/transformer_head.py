import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.nn.init import xavier_uniform_, constant_, normal_
import math

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)
            out_bbox = outputs["pred_boxes"][b]
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]
            if len(tgt_ids) == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            # Cost class: use negative probability of target class
            cost_class = -out_prob[:, tgt_ids]
            # Cost bbox: L1 distance
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses, eos_coef=0.02):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.eos_coef = eos_coef
        
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        return {'loss_bbox': loss_bbox.sum() / num_boxes}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {'labels': self.loss_labels, 'boxes': self.loss_boxes}
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if num_boxes < 1: num_boxes = torch.tensor([1.0], device=num_boxes.device)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses

class TransformerDecoderHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.MODEL.HEAD.DIM
        self.num_classes = cfg.DATASET.CLASS_INFO.NUM_CLS
        self.num_queries = cfg.MODEL.HEAD.NUM_QUERIES
        self.hidden_dim = cfg.MODEL.HEAD.HIDDEN_DIM
        self.nhead = cfg.MODEL.HEAD.NHEAD
        self.num_layers = cfg.MODEL.HEAD.NUM_LAYERS
        
        # ROI for normalization
        self.roi = cfg.DATASET.RDR_SP_CUBE.ROI
        self.x_min, self.x_max = self.roi['x']
        self.y_min, self.y_max = self.roi['y']
        self.z_min, self.z_max = self.roi['z']
        
        # Projection from backbone to transformer dimension
        self.input_proj = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=1)
        
        # Positional Embeddings
        self.row_embed = nn.Parameter(torch.rand(200, self.hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(200, self.hidden_dim // 2))
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
        
        # Object Queries
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        
        # Prediction Heads
        # Class head: +1 for background (no object)
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes + 1) 
        # Box head: xc, yc, zc, l, w, h, cos, sin (8 dims)
        self.bbox_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 8) 
        )
        
        # Matcher and Loss
        self.matcher = HungarianMatcher(cost_class=1, cost_bbox=5)
        self.criterion = SetCriterion(self.num_classes, matcher=self.matcher, weight_dict={'loss_ce': 1, 'loss_bbox': 5}, losses=['labels', 'boxes'])
        
        # Initialize weights
        xavier_uniform_(self.input_proj.weight, gain=1)
        constant_(self.input_proj.bias, 0)
        constant_(self.class_embed.bias, 0)
        
        # Initialize bbox_embed bias to output reasonable boxes
        # xc, yc, zc (sigmoid): 0 -> 0.5 (center)
        # l, w, h (linear): normalized by 20.0 -> 0.2, 0.1, 0.1 corresponds to 4m, 2m, 2m
        # cos, sin: 1, 0 (0 rad)
        bbox_init_bias = torch.tensor([0.0, 0.0, 0.0, 0.2, 0.1, 0.1, 1.0, 0.0])
        self.bbox_embed[-1].bias.data = bbox_init_bias
        constant_(self.bbox_embed[-1].weight, 0)

    def build_pos_embed(self, bs, h, w, device):
        # self.col_embed has shape (Max_W, D/2)
        # self.row_embed has shape (Max_H, D/2)
        
        # Check width
        if w <= self.col_embed.shape[0]:
            x_embed = self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1) # H, W, C/2
        else:
            # Interpolate if needed
            x_embed = F.interpolate(self.col_embed.unsqueeze(0).transpose(1,2), size=w, mode='linear', align_corners=False).transpose(1,2).squeeze(0)
            x_embed = x_embed.unsqueeze(0).repeat(h, 1, 1)

        # Check height
        if h <= self.row_embed.shape[0]:
            y_embed = self.row_embed[:h].unsqueeze(1).repeat(1, w, 1) # H, W, C/2
        else:
             # Interpolate if needed
            y_embed = F.interpolate(self.row_embed.unsqueeze(0).transpose(1,2), size=h, mode='linear', align_corners=False).transpose(1,2).squeeze(0)
            y_embed = y_embed.unsqueeze(1).repeat(1, w, 1)
            
        pos_embed = torch.cat([x_embed, y_embed], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(bs, 1, 1, 1) # B, C, H, W
        return pos_embed

    def forward(self, dict_item):
        # Input: (B, C, H, W)
        x = dict_item['bev_feat']
        bs, c, h, w = x.shape
        
        # Project to hidden dim
        x = self.input_proj(x) # (B, Hidden, H, W)
        
        # Prepare for Transformer
        # Flatten spatial dimensions: (B, Hidden, H, W) -> (B, Hidden, H*W) -> (H*W, B, Hidden)
        src = x.flatten(2).permute(2, 0, 1)
        
        # Positional Embeddings
        pos_embed = self.build_pos_embed(bs, h, w, x.device)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        
        # Queries: (Num_Queries, B, Hidden)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        
        # Decoder with fixed positional embeddings injection
        memory = src + pos_embed
        tgt = tgt + query_embed
        
        hs = self.transformer_decoder(tgt, memory)
        # hs: (Num_Queries, B, Hidden) -> Permute to (B, Num_Queries, Hidden)
        hs = hs.transpose(0, 1)
        
        # Predictions
        outputs_class = self.class_embed(hs) # (B, Num_Queries, Num_Classes + 1)
        outputs_coord = self.bbox_embed(hs) # (B, Num_Queries, 8)
        
        # Sigmoid for coordinates to keep them in range (optional but recommended for DETR-like)
        # We output normalized coordinates [0, 1] for xc, yc, zc
        # For dimensions and orientation, we can leave them or normalize
        # Here we apply sigmoid to first 3 (centers)
        outputs_coord[..., 0:3] = outputs_coord[..., 0:3].sigmoid()
        # Dimensions: exp or sigmoid? Let's use relu or softplus to be safe, or just raw and handle in loss
        # For now, let's keep them raw and assume the network learns
        
        dict_item['pred_logits'] = outputs_class
        dict_item['pred_boxes'] = outputs_coord
        
        return dict_item

    def loss(self, dict_item):
        outputs = {
            'pred_logits': dict_item['pred_logits'],
            'pred_boxes': dict_item['pred_boxes']
        }
        
        # Parse targets from dict_item['label']
        targets = []
        for list_objs in dict_item['label']:
            target = {'labels': [], 'boxes': []}
            for label in list_objs:
                cls_name, cls_id, (xc, yc, zc, rz, xl, yl, zl), _ = label
                
                # Normalize coordinates
                n_xc = (xc - self.x_min) / (self.x_max - self.x_min)
                n_yc = (yc - self.y_min) / (self.y_max - self.y_min)
                n_zc = (zc - self.z_min) / (self.z_max - self.z_min)
                
                # Normalize dimensions (simple scaling or log)
                # Let's use simple scaling assuming max dim is 20m
                n_xl = xl / 20.0
                n_yl = yl / 20.0
                n_zl = zl / 20.0
                
                # Orientation: cos, sin
                cos_t = math.cos(rz)
                sin_t = math.sin(rz)
                
                target['labels'].append(cls_id - 1)
                target['boxes'].append([n_xc, n_yc, n_zc, n_xl, n_yl, n_zl, cos_t, sin_t])
            
            if len(target['labels']) > 0:
                target['labels'] = torch.tensor(target['labels'], dtype=torch.long, device=outputs['pred_logits'].device)
                target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32, device=outputs['pred_boxes'].device)
            else:
                target['labels'] = torch.tensor([], dtype=torch.long, device=outputs['pred_logits'].device)
                target['boxes'] = torch.tensor([], dtype=torch.float32, device=outputs['pred_boxes'].device).reshape(0, 8)
            
            targets.append(target)
            
        loss_dict = self.criterion(outputs, targets)
        
        # Aggregate losses
        total_loss = sum(loss_dict[k] * self.criterion.weight_dict[k] for k in loss_dict.keys() if k in self.criterion.weight_dict)
        
        # Log losses
        if 'logging' not in dict_item:
            dict_item['logging'] = {}
        for k, v in loss_dict.items():
            dict_item['logging'][k] = v.item()
            
        return total_loss

    def get_nms_pred_boxes_for_single_sample(self, dict_item, conf_thr, is_nms=True):
        import numpy as np # Import numpy here
        pred_logits = dict_item['pred_logits'][0]
        pred_boxes = dict_item['pred_boxes'][0]
        prob = pred_logits.softmax(-1)
        scores, labels = prob[..., :-1].max(-1)
        
        keep = scores > conf_thr
        
        scores = scores[keep]
        labels = labels[keep]
        boxes = pred_boxes[keep]

        pp_bbox = []
        pp_cls = []

        for i in range(len(scores)):
             n_xc, n_yc, n_zc, n_xl, n_yl, n_zl, cos_t, sin_t = boxes[i].tolist()
             
             xc = n_xc * (self.x_max - self.x_min) + self.x_min
             yc = n_yc * (self.y_max - self.y_min) + self.y_min
             zc = n_zc * (self.z_max - self.z_min) + self.z_min
             xl = n_xl * 20.0
             yl = n_yl * 20.0
             zl = n_zl * 20.0
             rot = math.atan2(sin_t, cos_t)
             
             pp_bbox.append([scores[i].item(), xc, yc, zc, xl, yl, zl, rot])
             pp_cls.append(labels[i].item() + 1)

        dict_item['pp_bbox'] = pp_bbox
        dict_item['pp_cls'] = pp_cls

        if is_nms and len(pp_bbox) > 0:
            try:
                from nms.nms import rboxes
            except ImportError:
                print("Warning: nms package not found, skipping NMS")
                rboxes = None

            if rboxes:
                # Prepare data for NMS: [(x,y), (w,h), angle_deg]
                list_tuple_for_nms = []
                list_scores = []
                for bbox in pp_bbox:
                    # bbox: [score, xc, yc, zc, xl, yl, zl, rot_rad]
                    rot_deg = math.degrees(bbox[7])
                    list_tuple_for_nms.append([(bbox[1], bbox[2]), (bbox[4], bbox[5]), rot_deg])
                    list_scores.append(bbox[0])
                
                # nms.nms.rboxes expects scores as list, not numpy array according to docstring
                # but let's check if it handles list
                keep_indices = rboxes(list_tuple_for_nms, list_scores, nms_threshold=0.1)
                
                pp_bbox = [pp_bbox[i] for i in keep_indices]
                pp_cls = [pp_cls[i] for i in keep_indices]
                dict_item['pp_bbox'] = pp_bbox
                dict_item['pp_cls'] = pp_cls

        dict_item['pp_num_bbox'] = len(pp_bbox)
        
        # Add pp_desc for kitti evaluation
        if 'meta' in dict_item and len(dict_item['meta']) > 0:
            dict_item['pp_desc'] = dict_item['meta'][0]['desc']
        
        return dict_item



def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()
