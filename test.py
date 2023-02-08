import os
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
# from utils import calc_iou
from model import Encoder
import cv2
import wandb


from dataset import CustomDataset as CustomDataset

from args import get_parser
parser = get_parser()
args = parser.parse_args()
print("Arguments are : ", args)

wandb.login()
wandb.init(project="part_segmentation_test_files")

def color_map_test(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def convert_mask_to_image(mask):
    cmap = color_map_test()[:, np.newaxis, :]
    target = np.array(mask)[:, :, np.newaxis]
    new_im = np.dot(target == 0, cmap[0])
    for i in range(1, cmap.shape[0]):
        new_im += np.dot(target == i, cmap[i])
    # new_im = Image.fromarray(new_im.astype(np.uint8))
    return new_im.astype(np.uint8)

def inference(model, loader, loss_fn):
    model.eval()
    running_loss = 0.
    for i, batch in enumerate(loader):
        image = batch['image'].to(device)
        gt = batch['gt'].squeeze(1).type(torch.LongTensor).to(device)
        name = batch['name']
        

        with torch.no_grad():
            output = encoder(image)
            loss = loss_fn(output, gt)
            x = torch.nn.functional.softmax(output, dim = 1)
            pred = torch.argmax(x, dim=1)
            running_loss += loss.item()
            pred = pred.cpu().numpy()
            print(i, np.unique(pred, return_counts = True))

    

        for idx in range(len(name)):
            cv2.imwrite(os.path.join(args.result_dir, name[idx] + ".png"), convert_mask_to_image(pred[idx]))
            # plt.imshow(pred[idx])
            # plt.show()
    avg_loss = running_loss / (i + 1)
    print('Done testing')
    print("average loss : ", avg_loss)




device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.isdir(args.result_dir):
    print("Path to store results already exists")
else:
    os.mkdir(args.result_dir)
    print("Created path to store results.")


class_part_df = pd.read_csv(os.path.join(args.dataset_dir, "class_part_label.csv"))
names_df = pd.read_csv(os.path.join(args.dataset_dir, "names.csv"))
unique_part_names = list(class_part_df.part.unique())
    # print("unique_part_names : ", unique_part_names)


train, test = train_test_split(names_df, test_size=0.05, random_state = 42)
train = train.reset_index(drop = True)
test = test.reset_index(drop = True)


clip_model, preprocess = clip.load(args.clip_model, device=device)
encoder = Encoder(clip_model, unique_part_names)
encoder = encoder.to(device)


if os.path.isfile(args.model_name):
    print("Loading checkpoint '{}'".format(args.model_name))
    checkpoint = torch.load(args.model_name)
    encoder.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded model is for epoch {}".format(epoch))
    print("Loaded checkpoint '{}'".format(args.model_name))
else:
    print("Checkpoint load failed")


train_dataset = CustomDataset(class_part_csv_file = train, root_dir = args.dataset_dir,
                                     preprocess = preprocess)

test_dataset = CustomDataset(class_part_csv_file = test, root_dir = args.dataset_dir,
                                     preprocess = preprocess)

trainLoader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)
testLoader =  DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)


print("Length of Train dataset : {} and Test dataset : {}".format(len(train_dataset), len(test_dataset)))
print("Length of Train loader : {} and Test loader : {}".format(len(trainLoader), len(testLoader)))
loss_fn = nn.CrossEntropyLoss()
with torch.no_grad():
    inference(encoder, trainLoader, loss_fn)
    inference(encoder, testLoader, loss_fn)

wandb.save(os.path.join(args.result_dir, '*.png'), policy = 'now')

