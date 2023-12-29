import PIL
import torch, torchvision
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torch import nn
from skimage import io
import os
import argparse
from PIL import Image
from utils import AverageMeter
from tqdm import tqdm
from metrics import iou_score2,dice_score,f1_scorex2
from collections import OrderedDict


#网络模型
from unety import UNett_batcnnorm
from Unet import My_Unet
from Swin_Unet import vision_transformer
from VIT import ViT
from UNet3P.models import UNet_3Plus
from unety.SAR_UNet import Se_PPP_ResUNet
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from unety import attention_unet
from ResUNet_family import res_unet_plus
import resunet_aspp_up_vit4
from UNet2P import UNet_2Plus


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
from UCTransNet.nets import UCTransNet

#my_model
from baixibao_my_model import UNet_dense1
from baixibao_my_model import UNet_dense2
from baixibao_my_model import UNet_dense3
from baixibao_my_model import UNet_dense4
from baixibao_my_model import UNet_dense5
from baixibao_my_model import UNet_dense6

predimg = []
predimg_color = []
labelimg = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="UNet_dense5",
                        help='model name: Modified_UNET', choices=['UNET', 'UNET1', 'Swin_Unet','UCTransNet','VIT','resunet_aspp_up_vit3','resunet_aspp_up_CAM',
                                                                   'resunet_aspp_up_CAM2','resunet_aspp_up_CAM3',"UNet3P","UNet2P","SAR_UNet",'TransUNet',
                                                                   'attention_unet','resunet_plus','ECA_UNet_plus',"resunet","resunet_vit","resunet_aspp_up","resunet_aspp_up_change"
                                                                   ,"resunet_aspp_up_change2","resunet_traconv_aspp_up_CDA","DepthWiseConv_resUnet","DepthWiseConv_ResUNet_Tranconv_Aspp_up",
                                                                   "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA","DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change","ESPCN",
                                                                   "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2","DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change3",
                                                                   "resunet_Depthwise_CDA","resunet_aspp_up_vit4","UNet_dense1","UNet_dense2","UNet_dense3","UNet_dense4","UNet_dense5","UNet_dense6"])
    config = parser.parse_args()
    return config


def add_alpha_channel(img,fac):
    img = Image.open(img)
    img = img.convert('RGBA')
    # 更改图像透明度
    factor = fac
    img_blender = Image.new('RGBA', img.size, (0, 0, 0, 0))
    img = Image.blend(img_blender, img, factor)
    return img


def image_together(image, layer, save_path, save_name):
    layer = layer
    base = image
    # bands = list(layer.split())
    heigh, width = layer.size
    for i in range(heigh):
        for j in range(width):
            r, g, b, a = layer.getpixel((i, j))
            if r == 0 and g == 0 and b == 0:
                layer.putpixel((i, j), (0, 0, 0, 0))
            if r == 255 and g == 0 and b == 0:
                layer.putpixel((i, j), (255, 0, 0, 0))
            if r == 0 and g == 255 and b == 0:
                layer.putpixel((i, j), (0, 255, 0, 0))
            if r == 0 and g == 0 and b == 255:
                layer.putpixel((i, j), (0, 0, 255, 0))
    base.paste(layer, (0, 0), layer)  # 贴图操作
    base.save(save_path + "/" + save_name + ".png")  # 图片保存

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


def testdate(test_loader, model):
    avg_meters = {#'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'f1-score':AverageMeter()
    }
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for input, target in test_loader:
            input = input.float().cuda()
            target = target.long().cuda()
            b, h, w, c = input.size()
            input = input.reshape(b, c, h, w)
            output = model(input)

            preds = torch.softmax(output, dim=1).cpu()
            preds = torch.argmax(preds.data, dim=1)
            predimg_color.append(preds)
            preds = torch.squeeze(preds)
            predimg.append(preds)
            labelimg.append(target.cpu())

            iou = iou_score2(output, target)
            dice = dice_score(output, target)
            f1_score = f1_scorex2(output, target)
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['f1-score'].update(f1_score, input.size(0))
            postfix = OrderedDict([
                #('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('f1-score', avg_meters['f1-score'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([#('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('f1-score',avg_meters['f1-score'].avg)])


colormap1 = [[0,0,0], [255,0,0],[0,255,0],[0,0,255],[255,0,255],[255,255,0],[0,255,255]]

def label2image(prelabel,colormap):
    #预测的标签转化为图像，针对一个标签图
    _,h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w,-1)
    image = np.zeros((h*w,3),dtype=np.uint8)
    for i in range(len(colormap)):
        index = np.where(prelabel == i)
        image[index,:] = colormap[i]
    return image.reshape(h, w, 3)

forecast_label = 'C:\\Users\\beautiful\\Desktop\\lung_keshihua\\forecast_label' #预测标签转换成图片的地址
forecast_label_npy = "C:\\Users\\beautiful\\Desktop\\lung_keshihua\\forecast_label_npy"
labelnpytoimg_dir = 'C:\\Users\\beautiful\\Desktop\\lung_keshihua\\labelnpytoimg_dir' #将labelnpy转换成图片保存的地址
imgnpytoimg_dir = "C:\\Users\\beautiful\\Desktop\\lung_keshihua\\imgnpytoimg_dir"
img_dir = 'baixibao/test/img'  #原始图片npy
label_dir = 'baixibao/test/label'  #原始标签npy
imgandlabel = "C:\\Users\\beautiful\\Desktop\\lung_keshihua\\imgandlabel"
imgandlabel2 = "C:\\Users\\beautiful\\Desktop\\lung_keshihua\\imgandlabel2"

img_read = os.listdir(img_dir)

dataset = MyData(img_dir, label_dir,transformers=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

config = vars(parse_args())
print("=> creating model" )
if config['name'] == 'UNET':
    net = UNett_batcnnorm.Unet(3, 7)
elif config['name'] == 'UNET1':
    net = My_Unet(1, 4)
elif config['name'] == "Swin_Unet":
    net = vision_transformer.SwinUnet(img_size=128, num_classes=4)
elif config['name'] == "VIT":
    net = ViT(
          image_size = 128,
          channels=1,
          patch_size = 16,
          num_classes = 4,
          dim = 256,
          depth = 10,
          heads = 16,
          dim_head=16,
          mlp_dim = 256,
          dropout = 0.1,
          emb_dropout = 0.1
        )
elif config['name'] == "resunet_aspp_up_vit3":
    net = resunet_aspp_up_vit3.Unet(1,4)
elif config['name'] == "resunet_aspp_up_CAM":
    net = resunet_aspp_up_CAM.Unet(1,4)
elif config['name'] == "resunet_aspp_up_CAM2":
    net = resunet_aspp_up_CAM2.Unet(1, 4)
elif config['name'] == "resunet_aspp_up_CAM3":
    net = resunet_aspp_up_CAM3.Unet(1,4)
elif config['name'] == "UNet3P":
    net = UNet_3Plus.UNet_3Plus(in_channels=1, n_classes=4)
elif config['name'] == "UNet2P":
    net = UNet_2Plus.Unet2_Plus(num_classes=7)
elif config['name'] == "UCTransNet":
    net = UCTransNet.UCTransNet(n_channels=3, n_classes=7,img_size=256,vis=False)
elif config['name'] == 'SAR_UNet':
    net = Se_PPP_ResUNet(3, 7, deep_supervision=False)
elif config['name'] == 'TransUNet':
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    net = ViT_seg(config_vit, img_size=256, num_classes=7)
elif config['name'] == 'attention_unet':
    net = attention_unet.AttU_Net(3, 7)
elif config['name'] == "resunet_plus":
    net = res_unet_plus.ResUnetPlusPlus(3, 7)
elif config['name'] == 'ECA_UNet_plus':
    net = ECA_UNet_plus.Unet(1, 4)
elif config['name'] == "resunet":
    net = resunet(1, 4)
elif config['name'] == "resunet_vit":
    net = resunet_vit.Unet(1, 4)
elif config['name'] == "resunet_aspp_up":
    net = resunet_aspp_up.resunet(1, 4)
elif config['name'] == "resunet_aspp_up_change":
    net =resunet_aspp_up_change.resunet(1,4)
elif config['name'] == "resunet_aspp_up_change2":
    net =resunet_aspp_up_change2.resunet(1,4)
elif config['name'] == "resunet_traconv_aspp_up_CDA":
    net =resunet_traconv_aspp_up_CDA.resunet(1,4)
elif config['name'] == "DepthWiseConv_resUnet":
    net =DepthWiseConv_resUnet.resunet(1,4)
elif config['name'] == "DepthWiseConv_ResUNet_Tranconv_Aspp_up":
    net = DepthWiseConv_ResUNet_Tranconv_Aspp_up.resunet(1,4)
elif config['name'] == "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA":
    net = DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA.resunet(1,4)
elif config['name'] == "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change":
    net = DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change.resunet(1,4)
elif config['name'] == "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2":
    net = DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change2.resunet(1,4)
elif config['name'] == "DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change3":
    net = DepthWiseConv_ResUNet_Tranconv_Aspp_up_CDA_change3.resunet(1,4)
elif config['name'] == "ESPCN":
    net = ESPCN.resunet(1, 4)
elif config['name'] == "resunet_Depthwise_CDA":
    net = resunet_Depthwise_CDA.resunet(1,4)
elif config['name'] == "resunet_aspp_up_vit4":
    net = resunet_aspp_up_vit4.Unet(1,4)
elif config['name'] == "UNet_dense1":
    net = UNet_dense1.Unet(3,7)
elif config['name'] == "UNet_dense2":
    net = UNet_dense2.Unet(3,7)
elif config['name'] == "UNet_dense3":
    net = UNet_dense3.Unet(3,7)
elif config['name'] == "UNet_dense4":
    net = UNet_dense4.Unet(3,7)
elif config['name'] == "UNet_dense5":
    net = UNet_dense5.Unet(3,7)
elif config['name'] == "UNet_dense6":
    net = UNet_dense6.Unet(3,7)
else:
    raise ValueError("Wrong Parameters")

net.load_state_dict(torch.load(('E:\\Medical_imageseg_yes_baixibao\\checpoint\\baixibao\\{}_base\\bestmodel_baixibao_0.0001_final2.pth').format(config["name"])))
net.cuda()

if __name__ == "__main__":
    test_log = testdate(test_loader,net)
    print('testdata IOU:{:.4f}, testdata dice:{:.4f}, testdata f1-score:{:.4f}'.format(test_log['iou'],
                                                        test_log['dice'],test_log['f1-score']))
    for i in range(len(predimg)):
        pre = predimg[i]#预测npy
        preimg2 = label2image(predimg_color[i],colormap=colormap1)#预测label图片
        label = label2image(labelimg[i],colormap=colormap1)#原始label图片
        x = io.imread(img_dir+"\\"+img_read[i])
        if i < 10:
            test_pre_name = "000{}.png".format(i)
            test_pre_np = "000{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2,x)
            imgx = add_alpha_channel(clip_image_path2,0.85)
            imgx2 = add_alpha_channel(clip_image_path2,0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path,0.85)

            image_together(imgx,labelx,imgandlabel,test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)

        if i >= 10 and i < 100:
            test_pre_name = "00{}.png".format(i)
            test_pre_np = "00{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2, x)
            imgx = add_alpha_channel(clip_image_path2, 0.85)
            imgx2 = add_alpha_channel(clip_image_path2, 0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path, 0.85)

            image_together(imgx, labelx, imgandlabel, test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)

        if i>=100 and i < 1000:
            test_pre_name = "0{}.png".format(i)
            test_pre_np = "0{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2, x)
            imgx = add_alpha_channel(clip_image_path2, 0.85)
            imgx2 = add_alpha_channel(clip_image_path2, 0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path, 0.85)

            image_together(imgx, labelx, imgandlabel, test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)

        if i>=1000:
            test_pre_name = "{}.png".format(i)
            test_pre_np = "{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2, x)
            imgx = add_alpha_channel(clip_image_path2, 0.85)
            imgx2 = add_alpha_channel(clip_image_path2, 0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path, 0.85)

            image_together(imgx, labelx, imgandlabel, test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)



