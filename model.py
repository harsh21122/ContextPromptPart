import os
import cv2
import clip
import torch
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from einops import rearrange
import torch.cuda.amp as amp
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from skimage.transform import resize
from PIL import Image
import clip

import torch
from torch import nn
from torch.nn import init
from prompt import PromptLearner
from timm.models.layers import drop, drop_path, trunc_normal_


device = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    def __init__(self, clip_model, unique_part_names, type_):
        super().__init__()
        input_resolution=224
        width=64
        visual_dim = 2048
        new_dim = 1024
        text_dim=1024
        output_dim=1024
        self.partnames = unique_part_names
        self.partnames.insert(0, 'background')
        # self.partnames.append('background')
        print("self.partnames : ", self.partnames)
        prompt = [" the " + name + " of the cat." for name in self.partnames]
        print("prompts : ", prompt)
        print("model device : ", device)
        self.image_encoder = nn.Sequential(*nn.ModuleList(clip_model.visual.children())[:-1]).to(device)
        self.type_ = type_

        self.prompts = clip.tokenize(prompt).to(device)
        print("self.prompts :", self.prompts.shape)
        self.prompts = clip_model.encode_text(self.prompts)
        print("self.prompts :", self.prompts.shape)
        print("self.prompts rquires grad : ", self.prompts.requires_grad_)


        # self.image_encoder = clip_model.visual
        # self.features = {}
        # self.image_encoder.layer4.register_forward_hook(self.get_features('layer4'))
        # self.image_encoder.layer3.register_forward_hook(self.get_features('layer3'))
        # self.image_encoder.layer2.register_forward_hook(self.get_features('layer2'))
        # self.image_encoder.layer1.register_forward_hook(self.get_features('layer1'))


        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim) # dense clip code

        # self.text_encoder = TextEncoder(clip_model)
        # self.prompt_learner = PromptLearner(clip_model.to(device), unique_part_names)

        self.align_context = ContextDecoder() #denseclip code

        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-3)
        self.dtype = clip_model.dtype
        
        self.decoder = Decoder(in_channels = 2048)

        




    # def get_features(self, name):
    #     def hook(model, input, output):
    #         self.features[name] = output.detach()
    #     return hook
    
    
    def forward(self, input_batch):

        x4 = self.image_encoder(input_batch.type(self.dtype)) 
        # print("image_features : ", np.unique(x4.detach().cpu().numpy()))

        # ImageEncoder = self.image_encoder(input_batch.type(self.dtype))
        # print(" ImageEncoder : ", ImageEncoder.shape)
        # x4 = self.features['layer4']
        # x3 = self.features['layer3']
        # x2 = self.features['layer2']
        # x1 = self.features['layer1']
        # print("x4, x3, x2, x1 : ", x4.shape, x3.shape, x2.shape, x1.shape)
        
        x_global, x_local = self.attnpool(x4) # dense clip code
        # print("x_global : ", np.unique(x_global.detach().cpu().numpy()))
        # print("x_local : ", np.unique(x_local.detach().cpu().numpy()))
        # print("x_global : ", x_global.shape)
        # print("x_local : ", x_local.shape)
        B, C, H, W = x_local.shape

        visual_context = torch.cat([x_global.reshape(B, C, 1), x_local.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C
        
        # print("visual_context :", visual_context.shape, visual_context.dtype, visual_context.is_cuda)

        
        
        # B, C, H, W = self.features['layer4'].size()
        
        # print("B, C, H, W : ", B, C, H, W)
        # # print("S : ", S)
        # x4 = self.features['layer4'].reshape([B, C, H*W]).permute(0, 2, 1)
        # print("X4 :", x4.shape)
        # x4 = self.ln_proj_x4(x4)
        # print("X4 after linear proj:", x4.shape)

      
        # x4_bar = self.features['attnpool'].unsqueeze(1)
        # print("x4_bar :", x4_bar.shape)





        # prompts = self.prompt_learner()
        # print("prompts : ", np.unique(prompts.detach().cpu().numpy()))
        # # print("prompts :", prompts.shape, prompts.dtype)
        # # print("tokenized_prompts :", self.tokenized_prompts, self.tokenized_prompts.dtype)
        # text_features = self.text_encoder(prompts, self.tokenized_prompts)
        # print("text_features : ", torch.unique(text_features))
        # print("text_features : ", text_features.shape, text_features.dtype)
        # text_features = F.normalize(text_features, p=2.0, dim = 1)
        # print("text_features : ", torch.unique(text_features))

        # print("text_features : ", text_features.shape, text_features.dtype)


        text_features = self.prompts.expand(B, -1, -1)
        # print("text_features : ", text_features.shape, text_features.dtype)
        text_features = text_features.type(torch.cuda.FloatTensor)
        # print("text_features : ", text_features.shape, text_features.dtype)



        text_diff = self.align_context(text_features, visual_context)
        # print("text_diff :", text_diff.shape)
        text_features = text_features + self.gamma * text_diff
        # print("text_features : ", text_features.shape)
        # print("text_features : ", np.unique(text_features.detach().cpu().numpy()))



        # compute score map and concat
        B, K, C = text_features.shape
        x_local = F.normalize(x_local, dim=1, p=2)
        text = F.normalize(text_features, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', x_local, text)

        if self.type_ == 'attention':
            x_bar = torch.einsum('bchw,bkhw->bckhw', x4, score_map)
            print("attention x_bar : ", x_bar.shape)
            # sum of attention of all the part-channels
            x_bar = torch.mean(x_bar, dim = 2)
            print("attention after mean x_bar : ", x_bar.shape)
        
        # print("score_map : ", np.unique(score_map.detach().cpu().numpy()))
        # print("score_map : ", score_map.shape)
        # x_concat = torch.cat([x_local, score_map], dim=1)
        # print("x_concat : ", x_concat.shape) #  torch.Size([2, 1024 + 6, 7, 7])
        
        ## Need to add FPN decoder here to generate final map.
        final_map = self.decoder(x_bar)
        # print("final_map : ", final_map.shape)
        # print("final_map : ", np.unique(final_map.detach().cpu().numpy()))
        
        
        # print("Concatenated features along the channels : ", features.size())
        return final_map
        

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        # print("x1 : ", x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = x.permute(0, 2, 1, 3)  # NLD -> LND
        # print("x2 : ", x.shape)
        x = self.transformer(x)
        # print("x3 : ", x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print("x4 : ", x.shape)
        x = self.ln_final(x).type(self.dtype)
        # print("x5 : ", x.shape)

        # print("self.text_projection : ", self.text_projection.shape)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # print("x : ", x.shape)

        return x


class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    
    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual)
        
        return self.out_proj(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.positional_embedding[0:1, :]
        spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map


class Decoder(nn.Module):
    def __init__(self, in_channels, channels = [512, 256, 128, 6], resolution = 224):
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels[0]),
            nn.Conv2d(channels[0], channels[0], kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels[0]))
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels[1]),
            nn.Conv2d(channels[1], channels[1], kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels[1]))

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels[2]),
            nn.Conv2d(channels[2], channels[2], kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels[2]))

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels[3]),
            nn.Conv2d(channels[3], channels[3], kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels[3]))

        self.resolution = resolution
        self.m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

    def forward(self, score):
        y5 = score
        y4 = self.conv_layer1(y5)
        # print("y4.shape : ", y4.shape)
        y4 = self.m(y4)
        # print("y4.shape : ", y4.shape)

        # y4_ = self.four(features[0])
        # print("y4_.shape : ", y4_.shape)
        # y4 = y4 + y4_
        # print("y4.shape : ", y4.shape)

        y3 = self.conv_layer2(y4)
        # print("y3.shape : ", y3.shape)
        y3 = self.m(y3)
        # print("y3.shape : ", y3.shape)

        # y3_ = self.three(features[1])
        # print("y3_.shape : ", y3_.shape)
        # y3 = y3 + y3_
        # print("y3.shape : ", y3.shape)

        y2 = self.conv_layer3(y3)
        # print("y2.shape : ", y2.shape)
        y2 = self.m(y2)
        # print("y2.shape : ", y2.shape)

        # y2_ = self.two(features[2])
        # print("y2_.shape : ", y2_.shape)
        # y2 = y2 + y2_
        # print("y2.shape : ", y2.shape)

        y1 = self.conv_layer4(y2)
        # print("y1.shape : ", y1.shape)
        y1 = self.m(y1)
        # print("y1.shape : ", y1.shape)

        # y1_ = self.one(features[3])
        # print("y1_.shape : ", y1_.shape)
        # y1 = y1 + y1_

        map = F.interpolate(y1, (self.resolution, self.resolution), mode='bilinear')
        # print("map.shape : ", map.shape)



        return map
        



