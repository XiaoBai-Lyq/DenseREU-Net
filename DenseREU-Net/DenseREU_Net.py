from thop import profile
from torch import nn
import torch
from CBAM import CBAMLayer
import torch.nn.functional as F

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channel, output):
        super(ASPP, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=3, dilation=3)
        self.atrous_block12 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block18 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=9, dilation=9)
        self.avgpool = ASPPPooling(in_channel,in_channel)
        self.conv_1x1_output = nn.Conv2d((in_channel) * 6, output, 1, 1)
        self.batchnorm = nn.GroupNorm(8, output)
        self.action = nn.ReLU(inplace=True)

    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        aspppool = self.avgpool(x)
        cat = torch.cat([x, atrous_block1, atrous_block6, atrous_block12, atrous_block18, aspppool], dim=1)
        net = self.conv_1x1_output(cat)
        net = self.batchnorm(net)
        net = self.action(net)
        return net

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))

        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate)
        # 在通道维上将输入和输出连结
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1), layer)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.resconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch)
        )
        self.action = nn.ReLU(inplace=True)
        self.cbam = CBAMLayer(out_ch)
    def forward(self, input):
        x1 = self.conv(input)
        x2 = self.resconv(input)
        x3 = x1 + x2
        x3 = self.action(x3)
        x4 = self.cbam(x3)
        return x3, x4



class TransConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(TransConv, self).__init__()
        self.tranconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,input):
        return self.tranconv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = _DenseBlock(4, 64, bn_size=4, growth_rate=16, drop_rate=0)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = _DenseBlock(4, 128, bn_size=4, growth_rate=32, drop_rate=0)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = _DenseBlock(4, 256, bn_size=4, growth_rate=64, drop_rate=0)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        # self.conv5 = DoubleConv(512, 1024)
        self.up6 = TransConv(1024, 512)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = TransConv(512, 256)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = TransConv(256, 128)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = TransConv(128, 64)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.softmax = nn.Softmax()

        self.pool1_1 = nn.AvgPool2d(2, stride=2)
        # self.pool2_1 = nn.AvgPool2d(2, stride=2)
        # self.pool3_1 = nn.AvgPool2d(2, stride=2)
        # self.pool4_1 = nn.AvgPool2d(2, stride=2)
        self.aspp = ASPP(512,1024)

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c1, c1_1 = self.conv1(x)
        p1 = self.pool1(c1)
        p1_1 = self.pool1_1(c1_1)
        P1 = p1*p1_1
        c2= self.conv2(P1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        mid1 = self.dropout(c4)
        p4 = self.pool4(mid1)
        # c5, c5_1 = self.conv5(p4)
        c5_1 = self.aspp(p4)
        mid2 = self.dropout(c5_1)
        up_6 = self.up6(mid2)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6, c6_1 = self.conv6(merge6)
        up_7 = self.up7(c6_1)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7, c7_1 = self.conv7(merge7)
        up_8 = self.up8(c7_1)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8, c8_1 = self.conv8(merge8)
        up_9 = self.up9(c8_1)
        merge9 = torch.cat([up_9, c1_1], dim=1)
        c9, c9_1 = self.conv9(merge9)
        c10 = self.conv10(c9_1)
        return c10


# ==================================================================================
if __name__ == '__main__':
    unet = Unet(3, 7)
    unet.eval()
    rgb = torch.randn([1, 3, 256, 256])
    out1 = unet(rgb).size()

    flops, params = profile(unet, inputs=(rgb,))
    flop_g = flops / (10 ** 9)
    param_mb = params / (1024 * 1024)  # 转换为MB

    print(f"模型的FLOP数量：{flop_g}G")
    print(f"参数数量: {param_mb}MB")
    print(out1)




