import os
import einops
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
from vgg_extraction import FeatureExtraction, contrastive_loss

# from utils import calc_iou


from backbone import CLIPResNet
from dataset import CustomDataset as CustomDataset
# from transformer import TransformerDecoder
# from PartCLIP import PartCLIP

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_zoo_feat(images_zoo, zoo_feat_net):
    interp = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
    with torch.no_grad():
        zoo_feats = zoo_feat_net(images_zoo)
        zoo_feat = torch.cat([interp(zoo_feat) for zoo_feat in zoo_feats], dim=1)
    return zoo_feat

def train_one_epoch(encoder, zoo_feat_net, trainLoader, optimizer, loss_fn, args):
    total_running_loss = 0.
    total_running_loss_ce = 0.
    total_running_loss_contrastive = 0.
    running_loss = 0.
    last_loss = 0.
    iou_list = []

    for idx, batch in enumerate(trainLoader):
        image = batch['image'].to(device)
        images_vgg = batch['image_vgg'].to(device)
        gt = batch['gt'].squeeze(1).type(torch.LongTensor).to(device)
        name = batch['name']
            # classname = batch['classname']
            # partname = batch['partname']

        # print(np.unique(gt.cpu().numpy()))
        # print(np.unique(image.cpu().numpy()))
                    
        # print(name, image.shape, gt.shape, np.unique(gt.numpy()))
        optimizer.zero_grad()
        output = encoder(image)

        # this is contrastive code
        zoo_feat = get_zoo_feat(images_vgg, zoo_feat_net)
        # print("zoo_feat : ", zoo_feat.shape)
        basis = torch.einsum('brhw, bchw -> brc', output, zoo_feat)
        basis /= einops.reduce(output, 'b r h w -> b r 1', 'sum') + 1e-7
   
        loss_contrastive = contrastive_loss(basis[:, :, -args.layer_len:] if args.layer_len > 0 else basis, args.temperature)


        # print("torch.unique(gt) : ", torch.unique(gt))
        # print("pred = ", output.shape)
            # .type(torch.DoubleTensor)
        # print("output : ", output.shape, output.dtype, gt.shape, gt.dtype)
        # print("output : ", torch.unique(output))
        loss_ce = loss_fn(output, gt)
        total_loss = args.lamda_cross * loss_ce + args.lamda_contrastive * loss_contrastive

        total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)

        optimizer.step()
        print("Loss : ", loss_ce.item(), loss_contrastive.item(),  total_loss.item())
        x = torch.nn.functional.softmax(output, dim = 1)
        pred = torch.argmax(x, dim=1)
        
        # print("pred : ", torch.unique(pred))

        running_loss += total_loss.item()
        total_running_loss += total_loss.item()
        total_running_loss_ce += loss_ce.item()
        total_running_loss_contrastive += loss_contrastive.item()
        if (idx + 1) % 20 == 0:
            last_loss = running_loss / 20 # loss per batch
            print('  batch {} loss: {}, {}, {}'.format(idx + 1, loss_ce.item(), loss_contrastive.item(), total_loss.item(), last_loss))
            running_loss = 0.
            print("  torch.unique(gt) : ", torch.unique(gt))
            print("  pred unique : ", torch.unique(pred))
        

        
            
        
    
    avg_total_running_loss = total_running_loss/(idx + 1)
    avg_total_running_loss_ce = total_running_loss_ce/(idx + 1)
    avg_total_running_loss_contrastive = total_running_loss_contrastive/(idx + 1)
    
    return avg_total_running_loss, avg_total_running_loss_ce, avg_total_running_loss_contrastive

def validation(encoder, zoo_feat_net, loader, loss_fn, args):
    running_vloss = 0.0
    running_vloss_ce = 0.0
    running_vloss_contrast = 0.0

    iou_list = []
    for idx, batch in enumerate(loader):
        image = batch['image'].to(device)
        images_vgg = batch['image_vgg'].to(device)

        gt = batch['gt'].squeeze(1).type(torch.LongTensor).to(device)
        name = batch['name']

        output = encoder(image)
        
        # this is contrastive code
        zoo_feat = get_zoo_feat(images_vgg, zoo_feat_net)
        basis = torch.einsum('brhw, bchw -> brc', output, zoo_feat)
        basis /= einops.reduce(output, 'b r h w -> b r 1', 'sum') + 1e-7
        loss_contrastive = contrastive_loss(basis[:, :, -args.layer_len:] if args.layer_len > 0 else basis, args.temperature)


        # print("output : ", output.shape, output.dtype, gt.shape, gt.dtype)
        
        loss_ce = loss_fn(output, gt)
        total_loss = args.lamda_cross * loss_ce + args.lamda_contrastive * loss_contrastive
        running_vloss += total_loss.item()
        running_vloss_ce = loss_ce.item()
        running_vloss_contrast = loss_contrastive.item()

        print("Loss : ", loss_ce.item(), loss_contrastive.item(), total_loss.item())

        # print("Loss : ", loss.item())
        x = torch.nn.functional.softmax(output, dim = 1)
        pred = torch.argmax(x, dim=1)
        if idx % 20 == 0:
            print("  val done : ", idx)
            print("torch.unique(gt) : ", torch.unique(gt))
            print("  pred unique : ", torch.unique(pred))
        
        
        
    
    avg_vloss = running_vloss / (idx + 1)
    avg_vloss_ce = running_vloss_ce / (idx + 1)
    avg_vloss_cont = running_vloss_contrast / (idx + 1)
    return avg_vloss, avg_vloss_ce , avg_vloss_cont

def main(args):
    print("Using device : ", device)
    if args.wandb:
        import wandb
        wandb.login()
        wandb.init(project="part_segmentation")


    class_part_df = pd.read_csv(os.path.join(args.dataset_dir, "class_part_label.csv"))
    names_df = pd.read_csv(os.path.join(args.dataset_dir, "names.csv"))
    unique_part_names = list(class_part_df.part.unique())
    unique_part_names = ['head', 'neck', 'torso', 'tail', 'legs'] # hardcoding for now. Each part name index corresponds to gt index
    # print("unique_part_names : ", unique_part_names)


    train, test = train_test_split(names_df, test_size=0.05, random_state = 42)
    train = train.reset_index(drop = True)
    test = test.reset_index(drop = True)


    clip_model, preprocess = clip.load(args.clip_model, device=device)
    clip_visual = CLIPResNet([3, 4, 6, 3], pretrained= "pretrained/RN50.pt") # for ResNet50
    best_vloss = 1_000_000.
    
    


    
    
    # scaler = amp.GradScaler()
    
    # if args.resume:
    #     if os.path.isfile(args.model_name):
    #             print("loading checkpoint '{}'".format(args.model_name))
    #             checkpoint = torch.load(args.model_name)
    #             encoder.load_state_dict(checkpoint['state_dict'])
    #             optimizer.load_state_dict(checkpoint['optimizer'])
    #             if args.multi_step_scheduler:
    #                 scheduler.load_state_dict(checkpoint['scheduler'])
    #             best_vloss = checkpoint['best_vloss']
    #             epoch = checkpoint['epoch']
    #             print("Current best loss is {} and it is for epoch {}".format(best_vloss, epoch))
    #             print("loaded checkpoint '{}'".format(args.model_name))
    #     else:
    #         print("NO file to update checkpoint. Starting from fresh")
    # else:
    #     print("Resume turned off. Starting from fresh")

    train_dataset = CustomDataset(class_part_csv_file = train, root_dir = args.dataset_dir,
                                     preprocess = preprocess)

    test_dataset = CustomDataset(class_part_csv_file = test, root_dir = args.dataset_dir,
                                     preprocess = preprocess)

    trainLoader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)
    testLoader =  DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)


    print("Length of Train dataset : {} and Test dataset : {}".format(len(train_dataset), len(test_dataset)))
    print("Length of Train loader : {} and Test loader : {}".format(len(trainLoader), len(testLoader)))
    
    from model import Encoder
    encoder = Encoder(clip_model, clip_visual, unique_part_names, args.score_map_type)
    encoder = encoder.to(device)

    # This is taken from unsupervised contrastive paper for vgg feature extraction
    
    
    if int(args.ref_layer1[-3] + args.ref_layer1[-1]) <= int(args.ref_layer2[-3] + args.ref_layer2[-1]):
        last_layer = args.ref_layer1 + ',' + args.ref_layer2
        weights = [args.ref_weight1, args.ref_weight2]
    else:
        last_layer = args.ref_layer2 + ',' + args.ref_layer1
        weights = [args.ref_weight2, args.ref_weight1]
    zoo_feat_net = FeatureExtraction(
            feature_extraction_cnn='vgg19', normalization=False, last_layer=last_layer, weights=weights, gpu=0)
    zoo_feat_net.eval()



    
    named_parameters = []
    for name, param in encoder.named_parameters():
        if name.startswith('text_encoder'):
            param.requires_grad = False
        elif name.startswith('prompt_learner.token_embedding'):
            param.requires_grad = False
        elif name.startswith('image_encoder'):
            param.requires_grad = False
        else:
            named_parameters.append(param)

    print("Printing parameters and their gradient")
    for name, param in encoder.named_parameters():
        print(name, param.requires_grad)
    optimizer = torch.optim.Adam(named_parameters,
                                 lr = args.base_lr,
                                 weight_decay = args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    if args.multi_step_scheduler:
        scheduler = MultiStepLR(optimizer,
                                milestones = args.milestones,
                                gamma = args.lr_decay)


    # print(encoder)
    # print(clip_model.visual)
    print("Total epochs to be executed : ", args.epochs)
    for epoch in range(args.starting_epoch - 1, args.epochs):
        print('EPOCH {}:'.format(epoch + 1))


        encoder.train()
        avg_loss, avg_loss_ce, avg_loss_cont = train_one_epoch(encoder, zoo_feat_net, trainLoader, optimizer,loss_fn, args)
        
        # We don't need gradients on to do reporting
        
        encoder.eval()
        with torch.no_grad():
            avg_vloss, avg_vloss_ce, avg_vloss_cont  = validation(encoder, zoo_feat_net, testLoader, loss_fn, args)
            
            
        print('Total LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('CE LOSS train {} valid {}'.format(avg_loss_ce, avg_vloss_ce))
        print('Contrastive LOSS train {} valid {}'.format(avg_loss_cont, avg_vloss_cont))
        

        # Log the running loss averaged per batch

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            print("Saved best model. Old loss {} and new best loss {}".format(best_vloss, avg_vloss))
            best_vloss = avg_vloss
            if args.wandb:
                wandb.run.summary["best_val_loss"] = best_vloss
            model_path = os.path.join(args.model_dir, 'best_model_fpn_attention')
            if args.multi_step_scheduler:
                torch.save({'state_dict': encoder.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_vloss': best_vloss,
                        'epoch': epoch + 1}, model_path)
            else:
                torch.save({'state_dict': encoder.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'best_vloss': best_vloss,
                       'epoch': epoch + 1}, model_path)

        if args.wandb:   
            wandb.log({"epoch": epoch + 1, "train loss": avg_loss,
                  "val loss": avg_vloss, "lr": optimizer.param_groups[0]['lr'],
                  "train cross-entropy loss":avg_loss_ce, "val cross-entropy loss":avg_vloss_ce,
                  "train contrastive loss": avg_loss_cont, "val contrastive loss": avg_vloss_cont })
        
        model_path = os.path.join(args.model_dir, 'last_model_fpn_attention')
        if args.multi_step_scheduler:
            torch.save({'state_dict': encoder.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(),
                       'best_vloss': best_vloss,
                       'epoch': epoch + 1}, model_path)
        else:
            torch.save({'state_dict': encoder.state_dict(),
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

