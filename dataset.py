import os
import numpy as np
import cv2
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CustomDataset(Dataset):

    def __init__(self, class_part_csv_file, root_dir, preprocess):
        
        # self.classname = class_part_csv_file['class']
        # self.partname = class_part_csv_file['part']
        self.name = class_part_csv_file['name']
        self.root_dir = root_dir
        self.original_dir = os.path.join(self.root_dir, 'original')
        self.groundtruth_dir = os.path.join(self.root_dir, 'gt_mask')
        self.preprocess = preprocess
        self.transform = T.Compose([T.ToTensor()])
        self.vgg_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        
    def preprocess_name(self, name):
        # n = name.split("_")
        # original_name = n[0] + "_" + n[1]
        # groundtruth_name = name
        n = name.split(".")
        original_name = groundtruth_name = n[0]
        return original_name, groundtruth_name

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
#         print(idx, self.name[idx])
        
        original_name, groundtruth_name = self.preprocess_name(self.name[idx])
        original_path = os.path.join(self.original_dir, original_name + ".jpg")
        image = Image.open(original_path)
        # print("image : ", image.size)
        gt_path = os.path.join(self.groundtruth_dir, groundtruth_name + ".npy")
        gt = np.load(gt_path)
        
        # plt.imshow(gt)
        # plt.show()
        # gt = np.asarray(gt)
        
        if self.preprocess:
            img = self.preprocess(image)
            
        if self.transform:
            # print(np.unique(gt))
            gt = cv2.resize(gt, (224,224), interpolation = cv2.INTER_NEAREST)
            # print(np.unique(gt))
            # gt = self.transform(gt)
            gt = torch.from_numpy(gt)
            # print("np.unique(gt.numpy()) : ", np.unique(gt.numpy()))

        if self.vgg_transform:
            image_ = image.resize((224, 224))
            image_vgg = self.vgg_transform(image_)

            
        sample = {'name' : groundtruth_name, 'image': img, 'gt': gt, 'image_vgg': image_vgg }        

#         print(image.size(), gt.size(), text.shape)
        # sample = {'name' : groundtruth_name, 'image': image, 'gt': gt, 'classname' : self.classname[idx], 'partname' : self.partname[idx]}        
        return sample