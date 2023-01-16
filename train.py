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



from dataset import CustomDataset as CustomDataset
# from transformer import TransformerDecoder
# from PartCLIP import PartCLIP

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(model):
    PartCLIPmodel = PartCLIP(model).to(device)
    parameters = []
    named_parameters = []
    for name, param in PartCLIPmodel.named_parameters():
        if name.startswith('clip'):
            param.requires_grad = False
        else:
            named_parameters.append(param)
    
    # for name, param in PartCLIPmodel.named_parameters():
    #     print(name, param.requires_grad)
          
        
    for param in PartCLIPmodel.parameters():
        parameters.append(param)
    print(len(named_parameters), len(parameters))
    return PartCLIPmodel, named_parameters


def train_one_epoch(PartCLIPmodel, training_loader, optimizer, scaler, args):
    running_loss = 0.
    last_loss = 0.
    iou_list = []

    for idx, batch in enumerate(training_loader):
        
        text = batch['text'].squeeze(1).to(device)
        image = batch['image'].to(device)
        gt = batch['gt'].to(device)


        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = PartCLIPmodel(image, text)
            loss = F.binary_cross_entropy_with_logits(outputs, gt)
        
        scaler.scale(loss).backward()
#         loss.backward()

        # Adjust learning weights
        scaler.step(optimizer)
        scaler.update()
#         optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if (idx + 1) % 20 == 0:
            last_loss = running_loss / 20 # loss per batch
            print('  batch {} loss: {}'.format(idx + 1, last_loss))
            running_loss = 0.
            
        if args.calc_accuracy_training:
            gt = gt.detach().cpu().permute(0,2,3,1).squeeze(3).numpy()
            outputs = torch.sigmoid(outputs)
            pred = outputs.detach().cpu().permute(0,2,3,1).squeeze(3).numpy()
            iou = calc_iou(gt*255.0, pred*255.0)
            iou_list.append(iou)
    
    
    if iou_list:
        mean_acc = (sum(iou_list)/len(iou_list))
    else:
        mean_acc = -1
    return last_loss, mean_acc

def validation(PartCLIPmodel, loader, args):
    running_vloss = 0.0
    iou_list = []
    for i, batch in enumerate(loader):
        
        
        text = batch['text'].squeeze(1).to(device)
        image = batch['image'].to(device)
        gt = batch['gt'].to(device)

        outputs = PartCLIPmodel(image, text)
        
        vloss = F.binary_cross_entropy_with_logits(outputs, gt)
        running_vloss += vloss
        if i % 20 == 0:
            print("  val done : ", i)
        
        if args.calc_accuracy_training:
            gt = gt.detach().cpu().permute(0,2,3,1).squeeze(3).numpy()
            outputs = torch.sigmoid(outputs)
            pred = outputs.detach().cpu().permute(0,2,3,1).squeeze(3).numpy()
            iou = calc_iou(gt*255.0, pred*255.0)
            iou_list.append(iou)
        
    if iou_list:
        mean_acc = (sum(iou_list)/len(iou_list))
    else:
        mean_acc = -1
    avg_vloss = running_vloss / (i + 1)
    return avg_vloss, mean_acc

def main(args):
    print("Using device : ", device)
    # if args.wandb:
    #     import wandb
    #     wandb.login()
    #     wandb.init(project="referred_model_0.1_image_text")


    class_part_df = pd.read_csv(os.path.join(args.dataset_dir, "class_part_label.csv"))


    train, test = train_test_split(class_part_df, test_size=0.3, random_state = 42)
    train = train.reset_index(drop = True)
    test = test.reset_index(drop = True)


    clip_model, preprocess = clip.load(args.clip_model, device=device)
    # PartCLIPmodel, param_list = build_model(model)
    # best_vloss = 1_000_000.
    
    
    # optimizer = torch.optim.Adam(param_list,
    #                              lr = args.base_lr,
    #                              weight_decay = args.weight_decay)

    # if args.multi_step_scheduler:
    #     scheduler = MultiStepLR(optimizer,
    #                             milestones = args.milestones,
    #                             gamma = args.lr_decay)
    
    scaler = amp.GradScaler()
    
    if args.resume:
        if os.path.isfile(args.model_name):
                print("loading checkpoint '{}'".format(args.model_name))
                checkpoint = torch.load(args.model_name)
                PartCLIPmodel.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if args.multi_step_scheduler:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                best_vloss = checkpoint['best_vloss']
                epoch = checkpoint['epoch']
                print("Current best loss is {} and it is for epoch {}".format(best_vloss, epoch))
                print("loaded checkpoint '{}'".format(args.model_name))
        else:
            print("NO file to update checkpoint. Starting from fresh")
    else:
        print("Resume turned off. Starting from fresh")

    train_dataset = CustomDataset(class_part_csv_file = train, root_dir = args.dataset_dir,
                                     preprocess = preprocess)

    test_dataset = CustomDataset(class_part_csv_file = test, root_dir = args.dataset_dir,
                                     preprocess = preprocess)

    trainLoader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)
    testLoader =  DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)


    print("Length of Train dataset : {} and Test dataset : {}".format(len(train_dataset), len(test_dataset)))
    print("Length of Train loader : {} and Test loader : {}".format(len(trainLoader), len(testLoader)))
    
    from model import Encoder
    encoder = Encoder(clip_model)
    # print(encoder)
    print(clip_model.visual)
    for epoch in range(args.starting_epoch - 1, args.epochs):
        print('EPOCH {}:'.format(epoch + 1))
        for idx, batch in enumerate(trainLoader):
            image = batch['image'].to(device)
            gt = batch['gt'].to(device)
            name = batch['name']
            classname = batch['classname']
            partname = batch['partname']
            print(name, classname, partname, image.shape, gt.shape, np.unique(gt.numpy()))
            image_features, text_features = encoder(image, classname, partname)
            print("**************************************")
            break


        # Make sure gradient tracking is on, and do a pass over the data

        continue
        PartCLIPmodel.train()
        avg_loss, mean_train_acc = train_one_epoch(PartCLIPmodel, trainLoader, optimizer, scaler, args)
        # We don't need gradients on to do reporting
        
        PartCLIPmodel.eval()
        with torch.no_grad():
            avg_vloss, mean_vacc = validation(PartCLIPmodel, testLoader, args)
            
            
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        if args.calc_accuracy_training:
            print('Accuracy train {} valid {}'.format(mean_train_acc, mean_vacc))

        # Log the running loss averaged per batch

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            print("Saved best model. Old loss {} and new best loss {}".format(best_vloss, avg_vloss))
            best_vloss = avg_vloss
            if args.wandb:
                wandb.run.summary["best_val_loss"] = best_vloss
            model_path = os.path.join(args.model_dir, 'best_model')
            if args.multi_step_scheduler:
                torch.save({'state_dict': PartCLIPmodel.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_vloss': best_vloss,
                        'epoch': epoch + 1}, model_path)
            else:
                torch.save({'state_dict': PartCLIPmodel.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'best_vloss': best_vloss,
                       'epoch': epoch + 1}, model_path)

        if args.wandb:   
            wandb.log({"epoch": epoch + 1, "train loss": avg_loss,
                  "val loss": avg_vloss, "train accuracy": mean_train_acc,
                  "test accuracy": mean_vacc})
        
        model_path = os.path.join(args.model_dir, 'last_model')
        if args.multi_step_scheduler:
            torch.save({'state_dict': PartCLIPmodel.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(),
                       'best_vloss': best_vloss,
                       'epoch': epoch + 1}, model_path)
        else:
            torch.save({'state_dict': PartCLIPmodel.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'best_vloss': best_vloss,
                       'epoch': epoch + 1}, model_path)

        if args.multi_step_scheduler:
            scheduler.step()
        torch.cuda.empty_cache()
        


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print("Arguments are : ", args)
    main(args)

