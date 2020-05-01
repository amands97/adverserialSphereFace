import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math


import torch.nn.functional as F
# from .unet_parts import *
from unetparts import *


class MaskMan(nn.Module):
    """ Full assembly of the parts to form the complete network """


# class UNet(nn.Module):
    def __init__(self, n_channels=512, n_classes=1, bilinear=True):
        super(MaskMan, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.outc1 = OutConv(128, n_classes)
        self.upsample = nn.Upsample(scale_factor = 4, mode = 'nearest')
        self.outc2 = OutConv(512, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.down4(x4)
        # x5 = self.outc2(x5)
        # return x5
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        # x = self.outc1(x)
        # return x

        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

class MaskMan2(nn.Module):
    def __init__(self,  in_features):
        super(MaskMan, self).__init__()
        self.in_features = in_features
        self.conv1_1 = nn.Conv2d(in_features,128,3,1,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(128)
        self.conv1_2 = nn.Conv2d(128,16,3,1,1)
        self.relu1_2 = nn.PReLU(16)
        self.conv1_3 = nn.Conv2d(16,1,3,1,1)
        self.sigmoid = nn.Sigmoid()

# a = torch.rand((3,2)).type(torch.DoubleTensor)
# print(a)
# b = torch.randint(2, (3, 2)).type(torch.DoubleTensor)
# print(b)
# print(torch.mul(a, b))
# TODO: Check thether this is in_features or in_channels
    # Returns the mask. Use torch.mul(a,b) to get the resultant image. a and b are of same dimension
    # change: Return the masked image(not the mask)
    def forward(self, x):
        # x_ = x
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.sigmoid(self.conv1_3(x))
        #  TODO: change this sigmoid
        # do the multilpication in the main part of the function
        # x = torch.mul(x, x_) 
        return x

