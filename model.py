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
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from math import sqrt
import torch
from torch import nn
import ops
import math


def extract_center_submatrix(feature_map, submatrix_size=30):
    # Calculate the starting index for the submatrix
    start_index = (feature_map.shape[2] - submatrix_size) // 2

    # Extract and return the submatrices
    end_index = start_index + submatrix_size
    return feature_map[:, :, start_index:end_index, start_index:end_index]

def set_submatrix_zero(feature_map, submatrix_size=60):
    # Calculate the starting index for the submatrix
    start_index = (feature_map.shape[2] - submatrix_size) // 2

    # Extract and return the submatrices
    end_index = start_index + submatrix_size
    feature_map[:, :, start_index:end_index, start_index:end_index]=0
    return feature_map

def pad_to_larger_matrix(tensor, target_shape=(64, 1, 80, 80), padding_value=0):

    matrix = tensor.detach().cpu().numpy()
    padding_size = (target_shape[2] - matrix.shape[2]) // 2
    padding = ((0, 0), (0, 0), (padding_size, padding_size), (padding_size, padding_size))

    # Pad the matrix and return
    padded_matrix=np.pad(matrix, pad_width=padding, mode='constant', constant_values=padding_value)

    padded_tensor = torch.tensor(padded_matrix).cuda()
    return padded_tensor

class Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.ResidualBlock(in_channels, out_channels)
        self.b2 = ops.ResidualBlock(in_channels, out_channels)
        self.b3 = ops.ResidualBlock(in_channels, out_channels)
        self.c1 = ops.BasicBlock(in_channels * 2, out_channels, 1, 1, 0)
        self.c2 = ops.BasicBlock(in_channels * 3, out_channels, 1, 1, 0)
        self.c3 = ops.BasicBlock(in_channels * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class CARN(nn.Module):
    def __init__(self,d=64):
        super(CARN, self).__init__()

        scale = 2
        multi_scale = True
        group = 1

        # self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        # self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(1, d, 3, 1, 1)

        self.b1 = Block(d, d)
        self.b2 = Block(d, d)
        self.b3 = Block(d, d)
        self.c1 = ops.BasicBlock(d * 2, d, 1, 1, 0)
        self.c2 = ops.BasicBlock(d * 3, d, 1, 1, 0)
        self.c3 = ops.BasicBlock(d * 4, d, 1, 1, 0)

        self.upsample = ops.UpsampleBlock(d, scale=scale,
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(d, 1, 3, 1, 1)

    def forward(self, x, scale):
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=scale)
        out = self.exit(out)
        return out
    
class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=32, s=12, m=1):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.LeakyReLU()
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), 
                        nn.LeakyReLU()]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.LeakyReLU()])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.LeakyReLU()])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

    def forward(self, x):
        
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
    
class FSRCNN_change(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=32, s=12, m=2):
        super(FSRCNN_change, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=3, padding=3//2),
            nn.LeakyReLU()
        )
        self.mid_part = nn.Sequential(nn.Conv2d(d, num_channels, kernel_size=1), 
                        #  nn.PReLU(num_channels))
                        nn.LeakyReLU())
        # self.last_part = nn.ConvTranspose2d(s, num_channels, kernel_size=17, stride=scale_factor, padding=17//2,
        #                                     output_padding=scale_factor-1)

    def forward(self, x):
        
        x = self.first_part(x)
        x = self.mid_part(x)
        # x = self.last_part(x)
        x=F.interpolate(x, scale_factor=2, mode='bicubic')
        # x = x.view(x.size(0),-1)
        return x


class Bicubic_plus_plus(nn.Module):
    def __init__(self, sr_rate=2):
        super(Bicubic_plus_plus, self).__init__()
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv_out = nn.Conv2d(32, (2*sr_rate)**2 * 1, kernel_size=3, padding=1, bias=False)
        self.Depth2Space = nn.PixelShuffle(2*sr_rate)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.1)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.act(x0)
        x1 = self.conv1(x0)
        x1 = self.act(x1)
        x2 = self.conv2(x1)
        x2 = self.act(x2) + x0
        y = self.conv_out(x2)
        y = self.Depth2Space(y)
        return y    
    
    
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.carn = CARN()
        self.fsrcnn=FSRCNN_change(scale_factor=2)
        self.bicubic_plus=Bicubic_plus_plus(sr_rate=2)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=32,kernel_size=1,padding=0,groups=1,bias=False))
    def forward(self, output1,x):
        b_sz1, len = output1.shape
        roi=int(30)
        input2=output1[:,len//2-(roi*roi)//2:len//2+(roi*roi)//2]
        input1 = output1.reshape(b_sz1, 1, 40, 40)
        input2=input2.reshape(b_sz1,1,roi,roi)
        
        out_roi=self.carn(input2,2)
        
        # out_all=F.interpolate(input1, scale_factor=2, mode='bicubic')
        out_all=self.bicubic_plus(input1)
        out_roi_copy=out_roi.clone()
        out_roi_pad=pad_to_larger_matrix(out_roi_copy)
        out_all_copy=out_all.clone()
        out_all_set=set_submatrix_zero(out_all_copy)
        out=out_roi_pad+out_all_set
        out_roi = out_roi.view(out_roi.size(0), -1)
        out_all = out_all.view(out_all.size(0), -1)
        out = out.view(out.size(0), -1)
        return out_roi,out_all,out

class CombinedModel(nn.Module):
        def __init__(self, model1, model2):
            super(CombinedModel, self).__init__()
            self.model1 = model1
            self.model2 = model2


        def forward(self, x):
            output1 = self.model1(x)
            output2 = self.model2(output1,x)
            return output1, output2
        
