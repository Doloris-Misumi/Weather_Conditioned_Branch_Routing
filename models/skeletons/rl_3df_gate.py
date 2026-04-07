import torch
import torch.nn as nn
import torch.nn.functional as F

from models import pre_processor, backbone_3d, head, roi_head, img_cls
from models.text_encoder.clip_encoder import TextEncoder

class RL3DF_gate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.MODEL
        
        self.text_encoder = TextEncoder(freeze=True)
        self.text_encoder.eval()
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # ln(14.28)
        self.weather_vocab = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']
        weather_prompts = [f"A {w} driving scene" for w in self.weather_vocab]
        with torch.no_grad():
            weather_features = self.text_encoder(weather_prompts)
            weather_features = F.normalize(weather_features, dim=-1)
        self.register_buffer('weather_features', weather_features)
        self.prompt_token_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        self.list_module_names = [
            'pre_processor', 'pre_processor2', 'img_cls', 'backbone_3d', 'head', 'roi_head', 
        ]
        self.list_modules = []
        self.build_rl_detector()

    def build_rl_detector(self):
        for name_module in self.list_module_names:
            module = getattr(self, f'build_{name_module}')()
            if module is not None:
                self.add_module(name_module, module) # override nn.Module
                self.list_modules.append(module)

    def build_img_cls(self):
        if self.cfg_model.get('IMG_CLS', None) is None:
            return None
        
        module = img_cls.__all__[self.cfg_model.IMG_CLS.NAME]()
        return module 

    def build_pre_processor(self):
        if self.cfg_model.get('PRE_PROCESSOR', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR.NAME](self.cfg)
        return module 
    
    def build_pre_processor2(self):
        if self.cfg_model.get('PRE_PROCESSOR2', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR2.NAME](self.cfg)
        return module 

    def build_backbone_3d(self):
        cfg_backbone = self.cfg_model.get('BACKBONE', None)
        return backbone_3d.__all__[cfg_backbone.NAME](self.cfg)

    def build_head(self):
        if (self.cfg.MODEL.get('HEAD', None)) is None:
            return None
        module = head.__all__[self.cfg_model.HEAD.NAME](self.cfg)
        return module

    def build_roi_head(self):
        if (self.cfg.MODEL.get('ROI_HEAD', None)) is None:
            return None
        head_module = roi_head.__all__[self.cfg_model.ROI_HEAD.NAME](self.cfg)
        return head_module

    def forward(self, x):
        if 'condition_prompts' in x:
            with torch.no_grad():
                prompt_features = self.text_encoder(x['condition_prompts'])
                prompt_features = F.normalize(prompt_features, dim=-1)
            weather_logits = prompt_features @ self.weather_features.t()
            weather_probs = torch.softmax(weather_logits, dim=-1)
            prompt_weather_token = weather_probs @ self.weather_features
            x['prompt_weather_token'] = self.prompt_token_proj(prompt_weather_token)
            x['weather_probs'] = weather_probs

        for module in self.list_modules:
            x = module(x)

        if self.training and 'condition_prompts' in x:
            if 'img_embedding' in x:
                condition_token = x['img_embedding']
            else:
                pass 
            
            if 'img_embedding' in x:
                with torch.no_grad():
                    text_features = self.text_encoder(x['condition_prompts'])
                
                condition_token = F.normalize(condition_token, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                logit_scale = self.logit_scale.exp()
                logits_per_image = logit_scale * condition_token @ text_features.t()
                logits_per_text = logits_per_image.t()
                
                batch_size = condition_token.shape[0]
                labels = torch.arange(batch_size, device=condition_token.device)
                
                loss_i2t = F.cross_entropy(logits_per_image, labels)
                loss_t2i = F.cross_entropy(logits_per_text, labels)
                contrastive_loss = (loss_i2t + loss_t2i) / 2
                
                lambda_contrastive = 0.1 
                x['contrastive_loss'] = lambda_contrastive * contrastive_loss
                if 'logging' not in x: x['logging'] = {}
                x['logging']['loss_contrastive'] = contrastive_loss.item()

        return x
