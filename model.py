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

import torch
from torch import nn
from torch.nn import init
from prompt import PromptLearner


device = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # print("model device : ", device)
        self.image_encoder = nn.Sequential(*nn.ModuleList(clip_model.visual.children())[:-1]).to(device)
        self.text_encoder = TextEncoder(clip_model)

        self.prompt_learner = PromptLearner(clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.dtype = clip_model.dtype

    
    
    def forward(self, input_batch, classname, partname):

        image_features = self.image_encoder(input_batch.type(self.dtype)) 
        

        prompts = self.prompt_learner(classname, partname)
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        print("prompts :", prompts.shape)
        print("tokenized_prompts :", tokenized_prompts.shape)
        print(tokenized_prompts)


        text_features = self.text_encoder(prompts, tokenized_prompts)
        print("text_features : ", text_features.shape)
        print("image features : ", image_features.shape)

        # print("Concatenated features along the channels : ", features.size())
        return image_features, text_features

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
        print("x1 : ", x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        print("x2 : ", x.shape)
        x = self.transformer(x)
        print("x3 : ", x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        print("x4 : ", x.shape)
        x = self.ln_final(x).type(self.dtype)
        print("x5 : ", x.shape)

        print("self.text_projection : ", self.text_projection.shape)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        print("x : ", x.shape)

        return x

class Decoder(nn.Module):
    """
    Input:
        This component takes input the feature score map from text and image alignment

    Output:
        reconstructed image of with class as parts of the image

    Problem:
        - Alignment of textual features with respect to the image
        - Generation of finer features from score maps
        - Clear separation of parts in the object
    """
    def __init__(self, d_model: int, num_layer: int, n_head: int = 8) -> None:
        super().__init__()
        # Defining naive transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)


    def forward(self, features):
        pass
