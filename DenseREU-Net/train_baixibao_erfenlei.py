## 训练自己的数据集
import random
import torchvision
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import argparse
import os
from collections import OrderedDict
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from metrics1 import iou_score,dice_coef,compute_f1_score
from utils import AverageMeter, str2bool
from torch.utils.data import DataLoader
from skimage import io

#网络模型
from unety import UNett_batcnnorm
from Unet import My_Unet
from unety.SAR_UNet import Se_PPP_ResUNet
from Swin_Unet import vision_transformer
from VIT import ViT
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from unety import attention_unet
from ResUNet_family import res_unet_plus
from UNet3P.models import UNet_3Plus
import resunet_aspp_up_vit4
from UNet2P import UNet_2Plus
from UCTransNet.nets import UCTransNet


#改进模型
import resunet_aspp_up_vit3
import resunet_aspp_up_CAM
import resunet_aspp_up_CAM2
import resunet_aspp_up_CAM3
import ECA_UNet_plus
from unety.resunet import resunet
import resunet_vit
import resunet_aspp_up
import resunet_aspp_up_change
import resunet_aspp_up_change2
import resunet_traconv_aspp_up_CDA
import DepthWiseConv_resUnet
import DepthWiseConv_ResUNet_Tranconv_Aspp_up
import DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA
import DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change
import DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2
import DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change3
import ESPCN
import resunet_Depthwise_CDA


#my_model
from baixibao_my_model import UNet_dense1
from  baixibao_my_model import UNet_dense2
from baixibao_my_model import UNet_dense3
from baixibao_my_model import UNet_dense4
from baixibao_my_model import UNet_dense5
from baixibao_my_model import UNet_dense6

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--name', default="UNet_dense6",
                        help='model name: Modified_UNET',choices=['UNET', 'UNET1', 'Swin_Unet',"UCTransNet",'VIT','resunet_aspp_up_vit3','resunet_aspp_up_CAM',
                                                                  'resunet_aspp_up_CAM2','resunet_aspp_up_CAM3',"UNet3P","UNet2P"'SAR_UNet','TransUNet',
                                                                  'attention_unet',"resunet_plus","ECA_UNet_plus","resunet","resunet_vit","resunet_aspp_up","resunet_aspp_up_change"
                                                                  ,"resunet_aspp_up_change2","resunet_traconv_aspp_up_CDA","DepthWiseConv_resUnet","DepthWiseConv_ResUNet_Tranconv_Aspp_up",
                                                                  "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA","DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change","ESPCN",
                                                                  "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2","DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change3",
                                                                  "resunet_Depthwise_CDA","resunet_aspp_up_vit4","UNet_dense1","UNet_dense2","UNet_dense3","UNet_dense4","UNet_dense5","UNet_dense6"])
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 6)')
    parser.add_argument('--early_stopping', default=50, type=int,
                        metavar='N', help='early stopping (default: 50)')
    parser.add_argument('--num_workers', default=8, type=int)
    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    # data
    parser.add_argument('--augmentation', type=str2bool, default=False, choices=[True, False])
    config = parser.parse_args()

    return config


class MyData(Dataset):
    def __init__(self, root_dir, label_dir, transformers = None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.image_path = os.listdir(self.root_dir)
        self.label_path = os.listdir(self.label_dir)
        self.transformers = transformers
    def __getitem__(self, idx):  #如果想通过item去获取图片，就要先创建图片地址的一个列表
        img_name = self.image_path[idx]
        label_name = self.label_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)  # 每个图片的位置
        label_item_path = os.path.join(self.label_dir, label_name)
        image = io.imread(img_item_path)/255
        image = torch.from_numpy(image)
        label = io.imread(label_item_path)
        label = torch.from_numpy(label)
        return image,label
    def __len__(self):
        return len(self.image_path)


def train(train_loader, model, criterion, optimizer):
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'dice': AverageMeter(),
                      'f1-score': AverageMeter()
                      }
        model.train()
        pbar = tqdm(total=len(train_loader))
        for input, target in train_loader:
            input = input.float().cuda()
            target = target.float().cuda()
            target = torch.unsqueeze(target,dim=1)
            b,h,w,c = input.size()
            input = input.reshape(b,c,h,w)
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            f1_score = compute_f1_score(output, target)
            # compute gradient and do optimizing step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['f1-score'].update(f1_score, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('f1-score', avg_meters['f1-score'].avg)])

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('f1-score', avg_meters['f1-score'].avg)
                            ])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'f1-score': AverageMeter()
                  }
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.float().cuda()
            target = target.float().cuda()
            target = torch.unsqueeze(target,dim=1)
            b, h, w, c = input.size()
            input = input.reshape(b, c, h, w)
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            f1_score = compute_f1_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['f1-score'].update(f1_score, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('f1-score', avg_meters['f1-score'].avg)])

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('f1-score', avg_meters['f1-score'].avg)
                            ])


def main():
    """
    创建储存最好模型、xml文件
    """
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)

    # Get configuration
    config = vars(parse_args()) #将其数据类型变为字典类型
    # Make Model output directory
    if config['augmentation'] == True:
        file_name = config['name'] + '_with_augmentation'
    else:
        file_name = config['name'] + '_base'
    os.makedirs('checpoint/baixibao_erfenlei/{}'.format(file_name), exist_ok=True)
    print("Creating directory called",file_name)

    print('-' * 20)
    print("Configuration Setting as follow")
    for key in config:
        print('{}: {}'.format(key, config[key]))
    print('-' * 20)
    #save configuration
    with open('checpoint/baixibao_erfenlei/{}/config.yml'.format(file_name), 'w') as f:
        yaml.dump(config, f)

    """
    设置损失函数
    """
    # criterion = nn.CrossEntropyLoss(weight= torch.tensor([0,1,3,3,1.5,1,1]), reduction="mean").cuda()# 初始化损失函数
    criterion = nn.BCEWithLogitsLoss(weight=None, reduction="mean").cuda()  # 初始化损失函数
    # cudnn.benchmark = True
    """
    create model
    """
    print("=> creating model" )
    if config['name'] == "UNET":
        model = UNett_batcnnorm.Unet(3, 1)
    elif config['name'] == 'UNET1':
        model = My_Unet(1, 4)
    elif config['name'] == "Swin_Unet":
        model = vision_transformer.SwinUnet(img_size=256, num_classes=7)
    elif config['name'] == "UCTransNet":
        model = UCTransNet.UCTransNet(n_channels=3, n_classes=7,img_size=256,vis=False)
    elif config['name'] == "VIT":
        model = ViT(
                image_size=128,
                channels=1,
                patch_size=16,
                num_classes=4,
                dim=256,
                depth=10,
                heads=16,
                dim_head=16,
                mlp_dim=256,
                dropout=0.1,
                emb_dropout=0.1
             )
    elif config['name'] == "resunet_aspp_up_vit3":
        model = resunet_aspp_up_vit3.Unet(1, 4)
    elif config['name'] == "resunet_aspp_up_CAM":
        model = resunet_aspp_up_CAM.Unet(1, 4)
    elif config['name'] == "resunet_aspp_up_CAM2":
        model = resunet_aspp_up_CAM2.Unet(1, 4)
    elif config['name'] == "resunet_aspp_up_CAM3":
        model = resunet_aspp_up_CAM3.Unet(1, 4)
    elif config['name'] == "UNet3P":
        model = UNet_3Plus.UNet_3Plus(in_channels=3, n_classes=7)
    elif config['name'] == "UNet2P":
        model = UNet_2Plus.Unet2_Plus(num_classes=1)
    elif config['name'] == 'SAR_UNet':
        model = Se_PPP_ResUNet(3, 1, deep_supervision=False)
    elif config['name'] == 'TransUNet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        model = ViT_seg(config_vit, img_size=256, num_classes=1)
    elif config['name'] == 'attention_unet':
        model = attention_unet.AttU_Net(3, 1)
    elif config['name'] == "resunet_plus":
        model = res_unet_plus.ResUnetPlusPlus(3, 1)
    elif config['name'] == "ECA_UNet_plus":
        model = ECA_UNet_plus.Unet(1, 4)
    elif config['name'] == "resunet":
        model = resunet(1, 4)
    elif config['name'] == "resunet_vit":
        model = resunet_vit.Unet(1, 4)
    elif config['name'] == "resunet_aspp_up":
        model = resunet_aspp_up.resunet(1, 4)
    elif config['name'] == "resunet_aspp_up_change":
        model =resunet_aspp_up_change.resunet(1,4)
    elif config['name'] == "resunet_aspp_up_change2":
        model =resunet_aspp_up_change2.resunet(1,4)
    elif config['name'] == "resunet_traconv_aspp_up_CDA":
        model =resunet_traconv_aspp_up_CDA.resunet(1,4)
    elif config['name'] == "DepthWiseConv_resUnet":
        model =DepthWiseConv_resUnet.resunet(1,4)
    elif config['name'] == "DepthWiseConv_ResUNet_Tranconv_Aspp_up":
        model = DepthWiseConv_ResUNet_Tranconv_Aspp_up.resunet(1,4)
    elif config['name'] == "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA":
        model = DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA.resunet(1,4)
    elif config['name'] == "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change":
        model = DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change.resunet(1,4)
    elif config['name'] == "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2":
        model = DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2.resunet(1,4)
    elif config['name'] == "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change3":
        model = DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change3.resunet(1,4)
    elif config['name'] == "ESPCN":
        model = ESPCN.resunet(1, 4)
    elif config['name'] == "resunet_Depthwise_CDA":
        model = resunet_Depthwise_CDA.resunet(1,4)
    elif config['name'] == "resunet_aspp_up_vit4":
        model = resunet_aspp_up_vit4.Unet(1,4)
    elif config['name'] == "UNet_dense1":
        model = UNet_dense1.Unet(3,7)
    elif config['name'] == "UNet_dense2":
        model = UNet_dense2.Unet(3,7)
    elif config['name'] == "UNet_dense3":
        model = UNet_dense3.Unet(3,7)
    elif config['name'] == "UNet_dense4":
        model = UNet_dense4.Unet(3,7)
    elif config['name'] == "UNet_dense5":
        model = UNet_dense5.Unet(3,1)
    elif config['name'] == "UNet_dense6":
        model = UNet_dense6.Unet(3,1)
    else:
        raise ValueError("Wrong Parameters")

    model = model.cuda()
    params = filter(lambda p: p.requires_grad, model.parameters())#保存梯度信息
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) # 每经过step_size 个epoch，做一次学习率decay，以gamma值为缩小倍数。

###############加载数据路径
    train_img = 'baixobao_2fnelei/train/img'
    train_label = 'baixobao_2fnelei/train/label'
    val_img = 'baixobao_2fnelei/val/img'
    val_label = 'baixobao_2fnelei/val/label'

    train_dataset = MyData(train_img, train_label)
    val_dataset = MyData(val_img, val_label)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False)

    log= pd.DataFrame(index=[],columns= ['epoch','lr','loss','iou',"dice",'f1-score','val_loss','val_iou',"val_dice",'val_f1-score'])

    best_dice = 0
    trigger = 0

    for epoch in range(config['epochs']):

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        print('Training epoch [{}/{}], Training loss:{:.4f}, Training IOU:{:.4f}, Training DICE:{:.4f},Training f1-score:{:.4f}, Validation loss:{:.4f}, Validation IOU:{:.4f}, Validation DICE:{:.4f}, Validation f1-score:{:.4f}'.format(
            epoch + 1, config['epochs'], train_log['loss'], train_log['iou'],  train_log['dice'], train_log['f1-score'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['f1-score']))

        tmp = pd.Series([
            epoch,
            config['lr'],
            #train_log['lr_exp'],
            train_log['loss'],
            train_log['iou'],
            train_log['dice'],
            train_log['f1-score'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice'],
            val_log['f1-score']
        ], index=['epoch', 'lr', 'loss', 'iou','dice','f1-score', 'val_loss', 'val_iou',"val_dice",'val_f1-score'])

        log = log._append(tmp, ignore_index=True)
        log.to_csv('checpoint/baixibao_erfenlei/{}/log.csv'.format(file_name), index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'checpoint/baixibao_erfenlei/{}/bestmodel_baixibao_{}_final.pth'.format(file_name,config["lr"]))
            best_dice = val_log['dice']
            print("=> saved best model as validation DICE is greater than previous best DICE")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()