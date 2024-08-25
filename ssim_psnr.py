import torch
import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np
from math import exp



def cal_ssim(im1,im2):
      assert len(im1.shape) == 2 and len(im2.shape) == 2
      assert im1.shape == im2.shape
      mu1 = im1.mean()
      mu2 = im2.mean()
      sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
      sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
      sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
      k1, k2, L = 0.01, 0.03, 255
      C1 = (k1*L) ** 2
      C2 = (k2*L) ** 2
      C3 = C2/2
      l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
      c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
      s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
      ssim = l12 * c12 * s12
      return ssim


def cal_psnr(im1, im2):
    mse = (np.abs(im1 - im2) ** 2).mean()
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr