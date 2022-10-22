# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         unet
# Description:  Res50Unet +global pooling
# Author:       Boliu.Kelvin
# Date:         2020/4/12
#-------------------------------------------------------------------------------


import torchvision
resnet = torchvision.models.resnet.resnet50(pretrained=True)
import torch.nn as nn
import torch
import torch.functional as F

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super(Bridge,self).__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super(UpBlockForUNetWithResNet50,self).__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super(UNetWithResnet50Encoder,self).__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3] # conv2d + bn + relu
        self.input_pool = list(resnet.children())[3]   #maxpool
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)        # 4 layers contain {bottlenecks:3,4,6,3}
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

class Resnet50Encoder(nn.Module):
    def __init__(self,):
        super(Resnet50Encoder,self).__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:4]  # conv2d + bn + relu+maxpool
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)  # 4 layers contain {bottlenecks:3,4,6,3}
        self.down_blocks = nn.ModuleList(down_blocks)
        self.pool = AdaptivePool('average')

    def forward(self, x):
        outputs = []
        pools = []
        output = self.input_block(x)
        pools.append(self.pool(output))
        outputs.append(output)
        for i,layer in enumerate(self.down_blocks):
            output = layer(output)
            pools.append(self.pool(output))
            outputs.append(output)

        final_out = torch.cat(pools,1)
        return final_out
    
class VGG16Encoder(nn.Module):
    def __init__(self):
        super(VGG16Encoder,self).__init__()
        vgg = torchvision.models.vgg16_bn(pretrained=True)
        gap = [2,2,3,3,3]
        blocks = []
        blocks.append(list(vgg.children())[0][0:6])
        blocks.append(list(vgg.children())[0][6:13])
        blocks.append(list(vgg.children())[0][13:23])
        blocks.append(list(vgg.children())[0][23:33])
        blocks.append(list(vgg.children())[0][33:43])
        blocks.append(list(vgg.children())[0][43:44])
        self.blocks = nn.ModuleList(blocks)
        self.pool = AdaptivePool('average')


    def forward(self, x):

        pools=[]
        outputs=[]
        for layer in (self.blocks):
            x = layer(x)
            pools.append(self.pool(x))
            outputs.append(x)
        final_out = torch.cat(pools, 1)
        return final_out
class AdaptivePool(nn.Module):
    def __init__(self,way='average'):
        super(AdaptivePool,self).__init__()
        self.way = way


    def forward(self, x): #x.shape :[b,c,h,w]
        h = x.size()[2]
        w = x.size()[3]
        out = None
        if self.way =='average':
            pool = nn.AvgPool2d(kernel_size=[h,w])
        elif self.way=='max':
            pool = nn.MaxPool2d(kernel_size=[h,w])
        out = pool(x)
        return out
# model = UNetWithResnet50Encoder().cuda()
# inp = torch.rand((2, 3, 512, 512)).cuda()
# out = model(inp)