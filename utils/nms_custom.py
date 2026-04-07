import torch
import numpy as np
try:
    from utils.Rotated_IoU.oriented_iou_loss import cal_iou
except ImportError:
    # Try relative import if run from within utils
    from Rotated_IoU.oriented_iou_loss import cal_iou

def rboxes(list_tuple_for_nms, conf_score, nms_threshold=0.3):
    """
    NMS for rotated boxes.
    Args:
        list_tuple_for_nms: list of [(x,y), (w,h), angle]
        conf_score: numpy array of scores (N,) or (N,1)
        nms_threshold: float
    Returns:
        indices: list of kept indices
    """
    if len(list_tuple_for_nms) == 0:
        return []
    
    # Convert to tensor (N, 5)
    boxes_list = []
    for item in list_tuple_for_nms:
        (x, y), (w, h), angle = item
        boxes_list.append([x, y, w, h, angle])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    boxes = torch.tensor(boxes_list, dtype=torch.float32).to(device)
    scores = torch.tensor(conf_score, dtype=torch.float32).to(device).view(-1)
    
    # Sort by score
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices]
    # scores = scores[sorted_indices] # scores not needed anymore
    
    keep = []
    original_indices = sorted_indices.clone()
    
    while boxes.shape[0] > 0:
        # Pick the box with highest score
        current_idx = original_indices[0].item()
        keep.append(current_idx)
        
        if boxes.shape[0] == 1:
            break
            
        current_box = boxes[0:1] # (1, 5)
        rest_boxes = boxes[1:]   # (N-1, 5)
        
        # Expand current_box to match rest_boxes size for pair-wise comparison
        current_box_expanded = current_box.repeat(rest_boxes.shape[0], 1) # (N-1, 5)
        
        # Prepare for cal_iou (B, N, 5)
        cb = current_box_expanded.unsqueeze(0) # (1, N-1, 5)
        rb = rest_boxes.unsqueeze(0)           # (1, N-1, 5)
        
        # Calculate IoU
        # cal_iou returns (B, N) iou
        iou, _, _, _ = cal_iou(cb, rb) 
        iou = iou[0] # (N-1)
        
        # Filter boxes with high overlap
        mask = iou <= nms_threshold
        
        boxes = rest_boxes[mask]
        original_indices = original_indices[1:][mask]
        
    return keep
