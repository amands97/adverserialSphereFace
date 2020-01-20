import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

class MaskMan(nn.Module):
    def __init__(self,  in_features):
        super(MaskMan, self).__init__()
        self.in_features = in_features
        self.conv1_1 = nn.Conv2d(in_features,in_features,3,1,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(in_features)
        self.conv1_2 = nn.Conv2d(in_features,in_features,3,1,1)
        self.relu1_2 = nn.PReLU(in_features)
        self.conv1_3 = nn.Conv2d(in_features,in_features,3,1,1)
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
        # do the multilpication in the main part of the function
        # x = torch.mul(x, x_) 
        return x

