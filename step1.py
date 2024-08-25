import numpy as np
import numpy.matlib
from scipy.io import loadmat
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import time
import os

    

class LCNN(nn.Module):
    def __init__(self):
        super(LCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(2, 2), stride=(1, 1)),
                                   nn.PReLU(8),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 2), stride=(1, 1)),
                                   nn.PReLU(16),)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), stride=(1, 1)),
                                   nn.PReLU(32),)
        self.conv_3x3_2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
                                        nn.PReLU(32),)
        self.conv_3x3_1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
                                        nn.PReLU(16),)
        self.conv_in = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1)),
                                     nn.PReLU(32),)


        self.denses = nn.Sequential(
            # nn.Conv2d(in_channels=520, out_channels=520, kernel_size=(1, 1)),
            # nn.PReLU(520),
            # nn.Conv2d(in_channels=520, out_channels=1024, kernel_size=(1, 1)),
            # nn.PReLU(1024),
            # nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1, 1)),
            # nn.PReLU(1024),
            # nn.Conv2d(in_channels=1024, out_channels=1600, kernel_size=(1, 1)),
            # nn.PReLU(1600),
            nn.Linear(in_features=520, out_features=520),
            nn.PReLU(520),
            nn.Linear(in_features=520, out_features=1024),
            nn.PReLU(1024),
            nn.Linear(in_features=1024, out_features=1024),
            nn.PReLU(1024),
            nn.Linear(in_features=1024, out_features=1600),
            nn.PReLU(1600),
        )


    def forward(self, x):
        # out_3x3_1= F.leaky_relu_(self.conv_3x3_1(x))
        # out1 = F.leaky_relu_(self.conv1(x))
        # out_3x3_2 = F.leaky_relu_(self.conv_3x3_2(out1))
        # out2 = F.leaky_relu_(self.conv2(out1))
        # out_add1 = out_3x3_1+out2
        # out3 = F.leaky_relu_(self.conv3(out_add1))
        # out_add2 = out3+out_3x3_2
        # out3 = F.leaky_relu_(self.conv_in(out_add2))

        out_3x3_1= self.conv_3x3_1(x)
        out1 = self.conv1(x)
        out_3x3_2 = self.conv_3x3_2(out1)
        out2 = self.conv2(out1)
        out_add1 = out_3x3_1+out2
        out3 = self.conv3(out_add1)
        out_add2 = out3+out_3x3_2
        out3 = self.conv_in(out_add2)
        #自动调整维度上的元素个数
        out1 = out1.view(out1.size(0), -1)  # 8*3*7=168
        out2 = out2.view(out2.size(0), -1)  # 16*2*6=192
        out3 = out3.view(out3.size(0), -1)  # 32*1*5=160

        out = torch.cat([ out1, out2, out3 ], 1)  # 64+168+192+160=584-64=520
        # b_s,c = out.shape
        # out = out.reshape(b_s,c,1,1)

        out = self.denses(out)
        # out = out.view(out.size(0), -1)
        return out

