import torch
import torch.nn as nn

import spconv.pytorch as spconv
from einops.layers.torch import Rearrange
from sklearn.neighbors import NearestNeighbors
import numpy as np


class RL3DFBackbone_knngate(nn.Module):
    def __init__(self, cfg):
        super(RL3DFBackbone_knngate, self).__init__()
        self.cfg = cfg
        self.roi = cfg.DATASET.RDR_SP_CUBE.ROI
        grid_size = cfg.DATASET.RDR_SP_CUBE.GRID_SIZE

        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        z_min, z_max = self.roi['z']

        z_shape = int((z_max-z_min) / grid_size)
        y_shape = int((y_max-y_min) / grid_size)
        x_shape = int((x_max-x_min) / grid_size)

        self.spatial_shape = [z_shape, y_shape, x_shape]

        cfg_model = self.cfg.MODEL
        input_dim = cfg_model.PRE_PROCESSOR.INPUT_DIM

        list_enc_channel = cfg_model.BACKBONE.ENCODING.CHANNEL
        list_enc_padding = cfg_model.BACKBONE.ENCODING.PADDING
        list_enc_stride  = cfg_model.BACKBONE.ENCODING.STRIDE
        
        # 1x1 conv / 4->ENCODING.CHANNEL[0]
        self.input_convR = spconv.SparseConv3d(
            in_channels=input_dim, out_channels=list_enc_channel[0],
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key = 'sp0') 
        self.input_convL = spconv.SparseConv3d(
            in_channels=input_dim, out_channels=list_enc_channel[0],
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key = 'sp0') 
        
        # encoder
        self.num_layer = len(list_enc_channel)
        for idx_enc in range(self.num_layer):
            if idx_enc == 0:
                temp_in_ch = list_enc_channel[0] 
            else:
                temp_in_ch = list_enc_channel[idx_enc-1] 
            temp_ch = list_enc_channel[idx_enc]
            temp_pd = list_enc_padding[idx_enc]
            setattr(self, f'spconv{idx_enc}R', \
                spconv.SparseConv3d(in_channels=temp_in_ch, out_channels=temp_ch, kernel_size=3, \
                    stride=list_enc_stride[idx_enc], padding=temp_pd, dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'bn{idx_enc}R', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}aR', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}aR', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}bR', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}bR', nn.BatchNorm1d(temp_ch))
            
            setattr(self, f'spconv{idx_enc}L', \
                spconv.SparseConv3d(in_channels=temp_in_ch, out_channels=temp_ch, kernel_size=3, \
                    stride=list_enc_stride[idx_enc], padding=temp_pd, dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'bn{idx_enc}L', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}aL', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}aL', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}bL', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}bL', nn.BatchNorm1d(temp_ch))

            # Condition Token (512) -> Gate Feature
            setattr(self, f'img_layer{idx_enc}', nn.Linear(512, temp_ch))
            setattr(self, f'value_layer{idx_enc}', nn.Linear(temp_ch, temp_ch))
            setattr(self, f'gate_layer{idx_enc}', nn.Linear(2*temp_ch, temp_ch))
            setattr(self, f'gap_layer{idx_enc}', nn.AdaptiveAvgPool1d(1))
        
        # to BEV
        list_bev_channel = cfg_model.BACKBONE.TO_BEV.CHANNEL
        list_bev_kernel = cfg_model.BACKBONE.TO_BEV.KERNEL_SIZE
        list_bev_stride = cfg_model.BACKBONE.TO_BEV.STRIDE
        list_bev_padding = cfg_model.BACKBONE.TO_BEV.PADDING
        if cfg_model.BACKBONE.TO_BEV.IS_Z_EMBED:
            self.is_z_embed = True
            for idx_bev in range(self.num_layer):
                setattr(self, f'chzcat{idx_bev}R', Rearrange('b c z y x -> b (c z) y x'))
                temp_in_channel = int(list_enc_channel[idx_bev]*z_shape/(2**idx_bev))
                temp_out_channel = list_bev_channel[idx_bev]
                setattr(self, f'convtrans2d{idx_bev}R', \
                    nn.ConvTranspose2d(in_channels=temp_in_channel, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev], padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}R', nn.BatchNorm2d(temp_out_channel))
                
                setattr(self, f'chzcat{idx_bev}L', Rearrange('b c z y x -> b (c z) y x'))
                temp_in_channel = int(list_enc_channel[idx_bev]*z_shape/(2**idx_bev))
                temp_out_channel = list_bev_channel[idx_bev]
                setattr(self, f'convtrans2d{idx_bev}L', \
                    nn.ConvTranspose2d(in_channels=temp_in_channel, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev], padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}L', nn.BatchNorm2d(temp_out_channel))
        else:
            self.is_z_embed = False
            for idx_bev in range(self.num_layer):
                temp_enc_ch = list_enc_channel[idx_bev] 
                temp_out_channel = list_bev_channel[idx_bev]
                z_kernel_size = int(z_shape/(2**idx_bev))

                setattr(self, f'toBEV{idx_bev}R', \
                    spconv.SparseConv3d(in_channels=temp_enc_ch, \
                        out_channels=temp_enc_ch, kernel_size=(z_kernel_size, 1, 1)))
                setattr(self, f'bnBEV{idx_bev}R', \
                    nn.BatchNorm1d(temp_enc_ch))
                setattr(self, f'convtrans2d{idx_bev}R', \
                    nn.ConvTranspose2d(in_channels=temp_enc_ch, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev],  padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}R', nn.BatchNorm2d(temp_out_channel))
                
                setattr(self, f'toBEV{idx_bev}L', \
                    spconv.SparseConv3d(in_channels=temp_enc_ch, \
                        out_channels=temp_enc_ch, kernel_size=(z_kernel_size, 1, 1)))
                setattr(self, f'bnBEV{idx_bev}L', \
                    nn.BatchNorm1d(temp_enc_ch))
                setattr(self, f'convtrans2d{idx_bev}L', \
                    nn.ConvTranspose2d(in_channels=temp_enc_ch, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev],  padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}L', nn.BatchNorm2d(temp_out_channel))
        # activation
        self.relu = nn.ReLU()

    def forward(self, dict_item):
        sparse_featuresR, sparse_indicesR = dict_item['sp_features'], dict_item['sp_indices']
        sparse_featuresL, sparse_indicesL = dict_item['sp_features_l'], dict_item['sp_indices_l']
        
        # Use Condition Token (img_embedding) for environment-aware gating
        if 'img_embedding' in dict_item:
            condition_token = dict_item['img_embedding'] # (B, 512)
        else:
            # Fallback if not available (should not happen if img_cls is updated)
            # print("Warning: img_embedding not found, using dummy")
            condition_token = torch.zeros((dict_item['batch_size'], 512), device=sparse_featuresR.device)

        # img_cls_output, img_cls_gap, img_cls_feat = dict_item['img_cls_output'], dict_item['img_cls_gap'], dict_item['img_cls_feat']
     
        input_sp_tensorR = spconv.SparseConvTensor(
            features=sparse_featuresR,
            indices=sparse_indicesR.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )
        xR = self.input_convR(input_sp_tensorR) 
        
        input_sp_tensorL = spconv.SparseConvTensor(
            features=sparse_featuresL,
            indices=sparse_indicesL.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )

        xL = self.input_convL(input_sp_tensorL) 

        list_bev_featuresR = []
        list_bev_featuresL = []
        
        for idx_layer in range(self.num_layer):
            xR = getattr(self, f'spconv{idx_layer}R')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}R')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            xR = getattr(self, f'subm{idx_layer}aR')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}aR')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            xR = getattr(self, f'subm{idx_layer}bR')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}bR')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            
            xL = getattr(self, f'spconv{idx_layer}L')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}L')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
            xL = getattr(self, f'subm{idx_layer}aL')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}aL')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
            xL = getattr(self, f'subm{idx_layer}bL')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}bL')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
  
            xL2 = xL
            # Use Condition Token for gating
            img_layer_feat = getattr(self, f'img_layer{idx_layer}')(condition_token)
            if len(img_layer_feat.shape) == 1:
                img_layer_feat = img_layer_feat.unsqueeze(0)
      
            batch = dict_item['batch_size']
            xR_feat, xR_indices = xR.features, xR.indices
            xL_feat, xL_indices = xL.features, xL.indices
            for batch_idx in range(batch):
                radar_indices = np.array(xR_indices[xR_indices[:, 0] == batch_idx][:, 1:].cpu())
                lidar_indices = np.array(xL_indices[xL_indices[:, 0] == batch_idx][:, 1:].cpu())
                nbrs = NearestNeighbors(n_neighbors=int(64 / (2**(idx_layer))), radius=int(8 / (2**(idx_layer)))).fit(radar_indices)
                lidar_indices_temp = lidar_indices 
                distances, knn_indices = nbrs.kneighbors(lidar_indices_temp) 
                knn_indices = torch.from_numpy(knn_indices).to(device=sparse_indicesR.device) 

                lidar_feat = xL_feat[xL_indices[:, 0] == batch_idx] 
                radar_feat = xR_feat[xR_indices[:, 0] == batch_idx] 
                query = lidar_feat
                key = radar_feat[knn_indices] 

                value = getattr(self, f'value_layer{idx_layer}')(key) 
                attn = torch.bmm(query.unsqueeze(1), key.permute(0,2,1)) 
                attn = torch.softmax(attn, dim=-1) 
                attn_value = torch.bmm(attn, value)

                img_layer_feat_batch = img_layer_feat[batch_idx].unsqueeze(0).unsqueeze(0).repeat(key.shape[0], key.shape[1], 1) 
                gate_feat = getattr(self, f'gate_layer{idx_layer}')(torch.cat((key, img_layer_feat_batch), -1)) 
                gate_gap_feat = getattr(self, f'gap_layer{idx_layer}')(gate_feat.permute(0, 2, 1)) 
                attn_value_gate = torch.einsum('abc, abc -> abc', attn_value, torch.sigmoid(gate_gap_feat.permute(0, 2, 1))) 

                if batch_idx == 0:
                    new_xL_feat = attn_value_gate.squeeze() + query
                    Fl_feat = attn_value.squeeze() 
                    Gl_feat = gate_gap_feat.squeeze()
                    FlGl_feat = attn_value_gate.squeeze() 
                else:
                    new_xL_feat = torch.cat((new_xL_feat, attn_value_gate.squeeze() + query), 0)  
                    Fl_feat = torch.cat((Fl_feat, attn_value.squeeze()), 0)
                    Gl_feat = torch.cat((Gl_feat, gate_gap_feat.squeeze()), 0)
                    FlGl_feat = torch.cat((FlGl_feat, attn_value_gate.squeeze()), 0)
               
            new_spatial_shape = [int(self.spatial_shape[0] / (2**(idx_layer))), int(self.spatial_shape[1] / (2**(idx_layer))), int(self.spatial_shape[2] / (2**(idx_layer)))]
            xL = spconv.SparseConvTensor(
                features=new_xL_feat,
                indices=xL_indices,
                spatial_shape=new_spatial_shape,
                batch_size=dict_item['batch_size']
            )

            if self.is_z_embed:
                bev_denseR = getattr(self, f'chzcat{idx_layer}R')(xL2.dense())
                bev_denseR = getattr(self, f'convtrans2d{idx_layer}R')(bev_denseR)
                bev_denseL = getattr(self, f'chzcat{idx_layer}L')(xL.dense())
                bev_denseL = getattr(self, f'convtrans2d{idx_layer}L')(bev_denseL)
            else:
                bev_spR = getattr(self, f'toBEV{idx_layer}R')(xL2) # Lidar original feature
                bev_spR = bev_spR.replace_feature(getattr(self, f'bnBEV{idx_layer}R')(bev_spR.features))
                bev_spR = bev_spR.replace_feature(self.relu(bev_spR.features))
                
                bev_spL = getattr(self, f'toBEV{idx_layer}L')(xL)
                bev_spL = bev_spL.replace_feature(getattr(self, f'bnBEV{idx_layer}L')(bev_spL.features))
                bev_spL = bev_spL.replace_feature(self.relu(bev_spL.features))

                bev_denseR = getattr(self, f'convtrans2d{idx_layer}R')(bev_spR.dense().squeeze(2))
                bev_denseL = getattr(self, f'convtrans2d{idx_layer}L')(bev_spL.dense().squeeze(2))
            
            bev_denseR = getattr(self, f'bnt{idx_layer}R')(bev_denseR)
            bev_denseR = self.relu(bev_denseR)
            bev_denseL = getattr(self, f'bnt{idx_layer}L')(bev_denseL)
            bev_denseL = self.relu(bev_denseL)

            list_bev_featuresR.append(bev_denseR)
            list_bev_featuresL.append(bev_denseL)

        bev_featuresR = torch.cat(list_bev_featuresR, dim = 1)
        bev_featuresL = torch.cat(list_bev_featuresL, dim = 1)

        dict_item['bev_feat'] = torch.cat((bev_featuresR, bev_featuresL), 1)        

        return dict_item


class RL3DFBackbone_Branching(RL3DFBackbone_knngate):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sensor_token_proj = nn.Linear(cfg.MODEL.BACKBONE.ENCODING.CHANNEL[0], 512)
        left_transformer_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu'
        )
        self.left_transformer = nn.TransformerEncoder(left_transformer_layer, num_layers=2)
        self.left_transformer_norm = nn.LayerNorm(512)
        self.branch_router = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def _pool_sparse_tokens(self, features, indices, batch_size):
        pooled = []
        for batch_idx in range(batch_size):
            mask = indices[:, 0] == batch_idx
            if mask.any():
                pooled.append(features[mask].mean(dim=0))
            else:
                pooled.append(torch.zeros(features.shape[1], device=features.device, dtype=features.dtype))
        return torch.stack(pooled, dim=0)
        
    def forward(self, dict_item):
        sparse_featuresR, sparse_indicesR = dict_item['sp_features'], dict_item['sp_indices']
        sparse_featuresL, sparse_indicesL = dict_item['sp_features_l'], dict_item['sp_indices_l']

        img_condition_token = dict_item.get('img_embedding', None)
        prompt_condition_token = dict_item.get('prompt_weather_token', None)
        if img_condition_token is None:
            img_condition_token = torch.zeros((dict_item['batch_size'], 512), device=sparse_featuresR.device)
        if prompt_condition_token is None:
            prompt_condition_token = img_condition_token
        condition_token = 0.5 * (img_condition_token + prompt_condition_token)

        input_sp_tensorR = spconv.SparseConvTensor(
            features=sparse_featuresR,
            indices=sparse_indicesR.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )
        xR = self.input_convR(input_sp_tensorR) 
        
        input_sp_tensorL = spconv.SparseConvTensor(
            features=sparse_featuresL,
            indices=sparse_indicesL.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )
        xL = self.input_convL(input_sp_tensorL) 

        radar_token = self._pool_sparse_tokens(xR.features, xR.indices, dict_item['batch_size'])
        lidar_token = self._pool_sparse_tokens(xL.features, xL.indices, dict_item['batch_size'])
        radar_token = self.sensor_token_proj(radar_token)
        lidar_token = self.sensor_token_proj(lidar_token)

        tokens = torch.stack((condition_token, radar_token, lidar_token), dim=0)
        encoded_tokens = self.left_transformer(tokens)
        condition_token = self.left_transformer_norm(condition_token + encoded_tokens[0])
        dict_item['condition_token'] = condition_token

        branch_logits = self.branch_router(condition_token)
        branch_weights = torch.softmax(branch_logits, dim=-1)
        dict_item['branch_weights'] = branch_weights

        if dict_item.get('idx_iter', 0) == 0 and dict_item.get('local_rank', 0) == 0:
            with torch.no_grad():
                avg_weights = branch_weights.mean(dim=0)
                w_L, w_R, w_F = avg_weights[0].item(), avg_weights[1].item(), avg_weights[2].item()
                weather_str = "Unknown"
                if 'meta' in dict_item and len(dict_item['meta']) > 0:
                    first_meta = dict_item['meta'][0]
                    if isinstance(first_meta, dict) and 'desc' in first_meta and 'climate' in first_meta['desc']:
                        weather_str = first_meta['desc']['climate']
                print(f"[Weight Monitor] Weather: {weather_str:10s} | Lidar: {w_L:.3f} | Radar: {w_R:.3f} | Fusion: {w_F:.3f}")

        xL_pure = xL
        xL_fused = xL 

        list_bev_L = []
        list_bev_R = []
        list_bev_F = []
        
        for idx_layer in range(self.num_layer):
            # 1. Process Radar Stream (xR)
            xR = getattr(self, f'spconv{idx_layer}R')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}R')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            xR = getattr(self, f'subm{idx_layer}aR')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}aR')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            xR = getattr(self, f'subm{idx_layer}bR')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}bR')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            
            # 2. Process Pure Lidar Stream (xL_pure)
            xL_pure = getattr(self, f'spconv{idx_layer}L')(xL_pure)
            xL_pure = xL_pure.replace_feature(getattr(self, f'bn{idx_layer}L')(xL_pure.features))
            xL_pure = xL_pure.replace_feature(self.relu(xL_pure.features))
            xL_pure = getattr(self, f'subm{idx_layer}aL')(xL_pure)
            xL_pure = xL_pure.replace_feature(getattr(self, f'bn{idx_layer}aL')(xL_pure.features))
            xL_pure = xL_pure.replace_feature(self.relu(xL_pure.features))
            xL_pure = getattr(self, f'subm{idx_layer}bL')(xL_pure)
            xL_pure = xL_pure.replace_feature(getattr(self, f'bn{idx_layer}bL')(xL_pure.features))
            xL_pure = xL_pure.replace_feature(self.relu(xL_pure.features))
            
            # 3. Process Fusion Stream (xL_fused)
            # Use same weights as Lidar stream but on different tensor
            if idx_layer == 0:
                 # Clone first layer output to branch off
                 xL_fused = spconv.SparseConvTensor(
                    features=xL_pure.features.clone(),
                    indices=xL_pure.indices,
                    spatial_shape=xL_pure.spatial_shape,
                    batch_size=xL_pure.batch_size
                )
            else:
                xLf = xL_fused
                xLf = getattr(self, f'spconv{idx_layer}L')(xLf)
                xLf = xLf.replace_feature(getattr(self, f'bn{idx_layer}L')(xLf.features))
                xLf = xLf.replace_feature(self.relu(xLf.features))
                xLf = getattr(self, f'subm{idx_layer}aL')(xLf)
                xLf = xLf.replace_feature(getattr(self, f'bn{idx_layer}aL')(xLf.features))
                xLf = xLf.replace_feature(self.relu(xLf.features))
                xLf = getattr(self, f'subm{idx_layer}bL')(xLf)
                xLf = xLf.replace_feature(getattr(self, f'bn{idx_layer}bL')(xLf.features))
                xLf = xLf.replace_feature(self.relu(xLf.features))
                xL_fused = xLf

            # 4. Apply Fusion Logic to xL_fused
            img_layer_feat = getattr(self, f'img_layer{idx_layer}')(condition_token)
            if len(img_layer_feat.shape) == 1:
                img_layer_feat = img_layer_feat.unsqueeze(0)
                
            batch = dict_item['batch_size']
            xR_feat, xR_indices = xR.features, xR.indices
            xLf_feat, xLf_indices = xL_fused.features, xL_fused.indices
            
            new_xLf_feat_list = []
            
            # Simplified fusion loop (assuming batch alignment for brevity in this snippet)
            # In production code, we'd need the full loop. For now, assume single batch or implement loop.
            # I'll implement the loop properly.
            for batch_idx in range(batch):
                radar_indices = np.array(xR_indices[xR_indices[:, 0] == batch_idx][:, 1:].cpu())
                lidar_indices = np.array(xLf_indices[xLf_indices[:, 0] == batch_idx][:, 1:].cpu())
                
                if len(radar_indices) == 0 or len(lidar_indices) == 0:
                     new_xLf_feat_list.append(xLf_feat[xLf_indices[:, 0] == batch_idx])
                     continue

                nbrs = NearestNeighbors(n_neighbors=min(len(radar_indices), int(64 / (2**(idx_layer)))), radius=int(8 / (2**(idx_layer)))).fit(radar_indices)
                distances, knn_indices = nbrs.kneighbors(lidar_indices) 
                knn_indices = torch.from_numpy(knn_indices).to(device=sparse_indicesR.device) 

                lidar_feat_b = xLf_feat[xLf_indices[:, 0] == batch_idx] 
                radar_feat_b = xR_feat[xR_indices[:, 0] == batch_idx] 
                query = lidar_feat_b
                key = radar_feat_b[knn_indices] 
                
                value = getattr(self, f'value_layer{idx_layer}')(key) 
                attn = torch.bmm(query.unsqueeze(1), key.permute(0,2,1)) 
                attn = torch.softmax(attn, dim=-1) 
                attn_value = torch.bmm(attn, value)

                img_layer_feat_batch = img_layer_feat[batch_idx].unsqueeze(0).unsqueeze(0).repeat(key.shape[0], key.shape[1], 1) 
                gate_feat = getattr(self, f'gate_layer{idx_layer}')(torch.cat((key, img_layer_feat_batch), -1)) 
                gate_gap_feat = getattr(self, f'gap_layer{idx_layer}')(gate_feat.permute(0, 2, 1)) 
                attn_value_gate = torch.einsum('abc, abc -> abc', attn_value, torch.sigmoid(gate_gap_feat.permute(0, 2, 1))) 
                
                new_xLf_feat_list.append(attn_value_gate.squeeze() + query)
            
            new_xLf_feat = torch.cat(new_xLf_feat_list, 0)
            
            xL_fused = spconv.SparseConvTensor(
                features=new_xLf_feat,
                indices=xLf_indices,
                spatial_shape=xL_fused.spatial_shape,
                batch_size=xL_fused.batch_size
            )

            # 5. BEV Conversion
            def to_bev(tensor, idx_layer, suffix):
                if self.is_z_embed:
                    bev_dense = getattr(self, f'chzcat{idx_layer}{suffix}')(tensor.dense())
                    bev_dense = getattr(self, f'convtrans2d{idx_layer}{suffix}')(bev_dense)
                else:
                    bev_sp = getattr(self, f'toBEV{idx_layer}{suffix}')(tensor)
                    bev_sp = bev_sp.replace_feature(getattr(self, f'bnBEV{idx_layer}{suffix}')(bev_sp.features))
                    bev_sp = bev_sp.replace_feature(self.relu(bev_sp.features))
                    bev_dense = getattr(self, f'convtrans2d{idx_layer}{suffix}')(bev_sp.dense().squeeze(2))
                
                bev_dense = getattr(self, f'bnt{idx_layer}{suffix}')(bev_dense)
                bev_dense = self.relu(bev_dense)
                return bev_dense

            list_bev_L.append(to_bev(xL_pure, idx_layer, 'L'))
            list_bev_R.append(to_bev(xR, idx_layer, 'R'))
            list_bev_F.append(to_bev(xL_fused, idx_layer, 'L'))

        bev_feat_L = torch.cat(list_bev_L, dim=1)
        bev_feat_R = torch.cat(list_bev_R, dim=1)
        bev_feat_F = torch.cat(list_bev_F, dim=1)
        
        # 6. Construct Final Output based on Soft Weighting
        # bev_feat_L: Pure Lidar Features
        # bev_feat_R: Radar Features
        # bev_feat_F: Fused Features
        # Target Output Shape: Concatenation of two feature maps (e.g., Radar-like and Lidar-like channels)
        # To maintain compatibility with the head which expects cat(R, L) shape, we can construct weighted features.
        
        # Structure of final_bev was:
        # Lidar Only: [Zeros, L_pure]
        # Radar Only: [R, Zeros]
        # Fusion:     [R, F]
        
        # Soft Weighting Logic:
        # We will create a weighted sum for the "Radar-side" channel and "Lidar-side" channel.
        # Let weights be w_L, w_R, w_F
        # Radar-side channel: w_R * R + w_F * R = (w_R + w_F) * R  (Since Lidar Only has Zeros here)
        # Lidar-side channel: w_L * L_pure + w_F * F (Since Radar Only has Zeros here)
        
        w_L = branch_weights[:, 0].view(-1, 1, 1, 1) # (B, 1, 1, 1)
        w_R = branch_weights[:, 1].view(-1, 1, 1, 1)
        w_F = branch_weights[:, 2].view(-1, 1, 1, 1)
        
        # 1. Radar-side Component (First half of channels)
        # In Hard Selection:
        # if Lidar Only (w_L=1): Zeros
        # if Radar Only (w_R=1): R
        # if Fusion (w_F=1):     R
        # Soft equivalent: (w_R + w_F) * bev_feat_R
        final_bev_R_side = (w_R + w_F) * bev_feat_R
        
        # 2. Lidar-side Component (Second half of channels)
        # In Hard Selection:
        # if Lidar Only (w_L=1): L_pure
        # if Radar Only (w_R=1): Zeros
        # if Fusion (w_F=1):     F
        # Soft equivalent: w_L * bev_feat_L + w_F * bev_feat_F
        final_bev_L_side = w_L * bev_feat_L + w_F * bev_feat_F
        
        final_bev = torch.cat((final_bev_R_side, final_bev_L_side), dim=1)
                
        dict_item['bev_feat'] = final_bev
        return dict_item
