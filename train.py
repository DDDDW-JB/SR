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
from step1 import LCNN
# from model import ResBlock
from model import Net
from model import CombinedModel
import ssim_psnr
import matplotlib as mpl
import matplotlib.cm as cm
# from thop import profile

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
SNR = 35
num_epoch = 150
batch_size = 64
lr1 = 0.00065
lr2 = 0.0005
# lr1=0.001
# lr2=0.001

# #################### input #######################
mat = loadmat(r'/home/river/workplace/h_carn/fuyalei_60x60_roi.mat')
# A1_set = mat['A1_set']   # 32*11714  输入！！
# A2_set = mat['A2_set']   # 32*11714  输入！！
T_multirate_set_cu = mat['T_Rotation']  # 1600*11714 标签！！
# ##################################################

# #################### label #######################
mat1 = loadmat(r'/home/river/workplace/SR-OURS-M-COPY/fuyalei_80x80.mat')
A1_set = mat1['A1_set']   # 32*11714  输入！！
A2_set = mat1['A2_set']   # 32*11714  输入！！
T_multirate_set_xi = mat1['T_multirate_set']  # 6400*11714  标签！！
# ##################################################
#加噪
for i in range(0, 11714):
    M = 32
    nuw = (np.linalg.norm(A1_set[:, i]) ** 2) / M * (10 ** (-SNR / 10))
    A1_set[:, i] = A1_set[:, i] + np.sqrt(nuw) * np.matlib.randn(A1_set[:, i].shape)

for i in range(0, 11714):
    M = 32
    nuw = (np.linalg.norm(A2_set[:, i]) ** 2) / M * (10 ** (-SNR / 10))
    A2_set[:, i] = A2_set[:, i] + np.sqrt(nuw) * np.matlib.randn(A2_set[:, i].shape)

temp = np.random.permutation(11714)
# temp = loadmat(r'/home/river/workplace/lesrcnn/shunxu_200_35_best.mat')
# temp = temp['shunxu']
# temp = temp[0]

mdic3 = {"shunxu": temp, "label": "experment"}
savemat("shunxu_200_35_best.mat", mdic3)
t_train = 11200
t_test = 514
XTrain = np.zeros(shape=(t_train, 2, 4, 8))
XValidation = np.zeros(shape=(t_test, 2, 4, 8))

YTrain_cu = np.zeros(shape=(t_train, 1600))
YValidation_cu = np.zeros(shape=(t_test, 1600))
YTrain_xi = np.zeros(shape=(t_train, 6400))
YValidation_xi = np.zeros(shape=(t_test, 6400))

max_T_xi = np.max(T_multirate_set_xi)
min_T_xi = np.min(T_multirate_set_xi)
max_T_cu = np.max(T_multirate_set_cu)
min_T_cu = np.min(T_multirate_set_cu)

# 训练集 11125个
A1_train = A1_set[:, temp[0:t_train]]
# Normalized
mean_A1_train = np.mean(A1_train, axis=0)
std_A1_train = np.std(A1_train, axis=0)
A1_train = (A1_train - mean_A1_train) / std_A1_train

A2_train = A2_set[:, temp[0:t_train]]
mean_A2_train = np.mean(A2_train, axis=0)
std_A2_train = np.std(A2_train, axis=0)
A2_train = (A2_train - mean_A2_train) / std_A2_train

T_train_cu = T_multirate_set_cu[:, temp[0:t_train]]
T_train_cu = (T_train_cu - min_T_cu) / (max_T_cu - min_T_cu)

T_train_xi = T_multirate_set_xi[:, temp[0:t_train]]
T_train_xi = (T_train_xi - min_T_xi) / (max_T_xi - min_T_xi)


# 测试集――589个样本
A1_test = A1_set[:, temp[t_train:]]
# Normalized
mean_A1_test = np.mean(A1_test, axis=0)
std_A1_test = np.std(A1_test, axis=0)
A1_test = (A1_test - mean_A1_test) / std_A1_test

A2_test = A2_set[:, temp[t_train:]]
mean_A2_test = np.mean(A2_test, axis=0)
std_A2_test = np.std(A2_test, axis=0)
A2_test = (A2_test - mean_A2_test) / std_A2_test

T_test_cu = T_multirate_set_cu[:, temp[t_train:]]  # 测试集不进行归一化！

T_test_xi = T_multirate_set_xi[:, temp[t_train:]]


for i in range(0, t_train):
    # reshape 8 * 4 A1_test to 4 * 8 double_column_matrix_A1
    sample_matrix_A1 = (A1_train[:, i])
    sample_matrix_A1 = np.reshape(sample_matrix_A1, (8, 4), order="F")
    odd_line_A1 = sample_matrix_A1[0:7:2, :]
    even_line_A1 = sample_matrix_A1[1:8:2, :]
    double_column_matrix_A1 = np.zeros(shape=(4, 8))
    double_column_matrix_A1 = np.array(double_column_matrix_A1)
    double_column_matrix_A1[:, 0:7:2] = odd_line_A1
    double_column_matrix_A1[:, 1:8:2] = even_line_A1

    # reshape 8*4 A2_train to 4*8 double_column_matrix_A2
    sample_matrix_A2 = (A2_train[:, i])
    sample_matrix_A2 = np.reshape(sample_matrix_A2, (8, 4), order="F")
    odd_line_A2 = sample_matrix_A2[0:7:2, :]
    even_line_A2 = sample_matrix_A2[1:8:2, :]
    double_column_matrix_A2 = np.zeros(shape=(4, 8))
    double_column_matrix_A2 = np.array(double_column_matrix_A2)
    double_column_matrix_A2[:, 0:7:2] = odd_line_A2
    double_column_matrix_A2[:, 1:8:2] = even_line_A2

    # generate 4D array XTrain, 11125*2*4*8
    XTrain[i, 0, :, :] = double_column_matrix_A1
    XTrain[i, 1, :, :] = double_column_matrix_A2


YTrain_cu = np.transpose(T_train_cu)
YTrain_xi = np.transpose(T_train_xi)

for i in range(0, t_test):
    # reshape 8 * 4 A1_test to 4 * 8 double_column_matrix_A1
    sample_matrix_A1 = (A1_test[:, i])
    sample_matrix_A1 = np.reshape(sample_matrix_A1, (8, 4), order="F")
    odd_line_A1 = sample_matrix_A1[0:7:2, :]
    even_line_A1 = sample_matrix_A1[1:8:2, :]
    double_column_matrix_A1 = np.zeros(shape=(4, 8))
    double_column_matrix_A1 = np.array(double_column_matrix_A1)
    double_column_matrix_A1[:, 0:7:2] = odd_line_A1
    double_column_matrix_A1[:, 1:8:2] = even_line_A1

    # reshape 8 * 4 A2_test to 4 * 8 double_column_matrix_A2
    sample_matrix_A2 = (A2_test[:, i])
    sample_matrix_A2 = np.reshape(sample_matrix_A2, (8, 4), order="F")
    odd_line_A2 = sample_matrix_A2[0:7:2, :]
    even_line_A2 = sample_matrix_A2[1:8:2, :]
    double_column_matrix_A2 = np.zeros(shape=(4, 8))
    double_column_matrix_A2 = np.array(double_column_matrix_A2)
    double_column_matrix_A2[:, 0:7:2] = odd_line_A2
    double_column_matrix_A2[:, 1:8:2] = even_line_A2

    # generate 4 D array XValidation, 589*2*4*8
    XValidation[i, 0, :, :] = double_column_matrix_A1
    XValidation[i, 1, :, :] = double_column_matrix_A2


YValidation_cu = np.transpose(T_test_cu)
YValidation_xi = np.transpose(T_test_xi)

# torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
XTrain = torch.from_numpy(XTrain)
YTrain_cu = torch.from_numpy(YTrain_cu)
YTrain_xi = torch.from_numpy(YTrain_xi)
XValidation = torch.from_numpy(XValidation)
YValidation_cu = torch.from_numpy(YValidation_cu)
YValidation_xi = torch.from_numpy(YValidation_xi)

torch_dataset = Data.TensorDataset(XTrain, YTrain_cu, YTrain_xi)
# torch_dataset = Data.TensorDataset(XTrain, YTrain_xi)
Train_loader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

test_dataset = Data.TensorDataset(XValidation, YValidation_cu, YValidation_xi)
# test_dataset = Data.TensorDataset(XValidation,  YValidation_xi)
test_loader = Data.DataLoader(test_dataset, batch_size=t_test, shuffle=False)


# ########################train##########################


net1 = LCNN()
net1.cuda()
net2 = Net()
# net = CombinedModel(net1, net2)
# net = torch.load('OURS_200_35_no40*40.pkl')
net2.cuda()

# print("trainable params:{}".format(sum(x.numel() for x in net.parameters())))

optimizer = torch.optim.Adam( 
    [   {"params": net1.parameters(), "lr": lr1},
        {"params": net2.parameters(), "lr": lr2},  
    ], 
)

# optimizer = torch.optim.Adam(params=net1.parameters(), lr=0.00065)
# loss_func2 = nn.MSELoss().cuda()
loss_func2 = nn.L1Loss().cuda()

total_step = len(Train_loader)
torch.cuda.synchronize()
b = time.time()
for epoch in range(num_epoch):
    for step, (x,y_cu,y_xi) in enumerate(Train_loader):  # gives batch data, normalize x when iterate train_loader
        x = torch.tensor(x, dtype=torch.float32)
        y_cu = torch.tensor(y_cu, dtype=torch.float32)
        y_xi = torch.tensor(y_xi, dtype=torch.float32)
        x = x.clone().detach().float()
        y_cu = y_cu.clone().detach().float()
        y_xi = y_xi.clone().detach().float()
        x = x.cuda()
        y_cu = y_cu.cuda()
        y_xi = y_xi.cuda()

        # output_step1, output_step2 = net(x)
        output_step1 = net1(x)
        output_roi,_,output_step2 = net2(output_step1,x)
        b_s,_ = output_step1.shape
        output_step1 = output_step1.reshape(b_s, 1, 40, 40)
        output_step1_up = F.interpolate(output_step1, scale_factor=2, mode='bicubic')    # 双三次插值倍上采样
        output_step1_up = output_step1_up.view(output_step1_up.size(0), -1)
        loss1 = loss_func2(output_step1_up, y_xi)
        loss2 = loss_func2(output_roi,y_cu)
        loss3 = loss_func2(output_step2, y_xi)
        loss = 0.8*loss2+0.2*loss3
        optimizer.zero_grad()  # clear gradients for this training step

        loss1.backward(retain_graph=True)  # backpropagation, compute gradients
        # loss2.backward()
        # loss3.backward()
        loss.backward()
        optimizer.step()  # apply gradients

        if (step + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss1: {:.4f}, Loss2: {:.4f}'
                  .format(epoch + 1, num_epoch, step + 1, total_step, loss1.item(), loss2.item()))
torch.cuda.synchronize()
e = time.time()
train_time = e - b
net1.eval()
net2.eval()
# torch.save(net,'OURS_200_35_no40*40_1.0.pkl')
with torch.no_grad():
    for (XValidation, YValidation_cu ,YValidation_xi) in test_loader:
        XValidation = torch.tensor(XValidation, dtype=torch.float32)
        YValidation_xi = torch.tensor(YValidation_xi, dtype=torch.float32)
        XValidation = XValidation.clone().detach().float()
        YValidation_xi = YValidation_xi.clone().detach().float()
        XValidation = XValidation.cuda()
        YValidation_xi = YValidation_xi.cuda()

        torch.cuda.synchronize()
        begin_time1 = time.time()

        YPredicted1 = net1(XValidation)
        end_time1 = time.time()
        begin_time2 = time.time()
        YPredicted2_roi,YPredicted2_all,YPredicted2 = net2(YPredicted1,XValidation)

        torch.cuda.synchronize()
        end_time2 = time.time()
        run_time_all1 = end_time1 - begin_time1
        run_time_all2 = end_time2 - begin_time2
        print('test_time_all1', run_time_all1)
        print('test_time_all2', run_time_all2)


b_s_test,_ = YPredicted1.shape
YPredicted1 = YPredicted1.reshape(b_s_test, 1, 40, 40)
YPredicted1_up = F.interpolate(YPredicted1, scale_factor=2, mode='bicubic')    # 双三次插值倍上采样
YPredicted1_up = YPredicted1_up.view(YPredicted1_up.size(0), -1)
YPredicted1_up = YPredicted1_up.cuda().data.cpu().numpy()
YPredicted1 = YPredicted1.cuda().data.cpu().numpy()
YPredicted2_roi = YPredicted2_roi.cuda().data.cpu().numpy()
YPredicted2_all = YPredicted2_all.cuda().data.cpu().numpy()
YPredicted2 = YPredicted2.cuda().data.cpu().numpy()
# YValidation_cu = YValidation_cu.cuda().data.cpu().numpy()  # t_test*1600
YValidation_xi = YValidation_xi.cuda().data.cpu().numpy()  # t_test*6400

YPredicted1 = np.maximum(YPredicted1, 0)
YPredicted1 = YPredicted1 * (max_T_xi - min_T_xi) + min_T_xi  # t_test*1600

YPredicted1_up = np.maximum(YPredicted1_up, 0)
YPredicted1_up = YPredicted1_up * (max_T_xi - min_T_xi) + min_T_xi  # t_test*1600

YPredicted2_roi = np.maximum(YPredicted2_roi, 0)
YPredicted2_roi = YPredicted2_roi * (max_T_cu - min_T_cu) + min_T_xi  # t_test*6400

YPredicted2_all = np.maximum(YPredicted2_all, 0)
YPredicted2_all = YPredicted2_all * (max_T_xi - min_T_xi) + min_T_xi  # t_test*6400

YPredicted2 = np.maximum(YPredicted2, 0)
YPredicted2 = YPredicted2 * (max_T_xi - min_T_xi) + min_T_xi  # t_test*6400

CNN_sample_error_0 = 0
CNN_sample_error_1 = 0
CNN_sample_error_2 = 0
CNN_ssim_L = 0
CNN_psnr_L = 0
CNN_ssim_H = 0
CNN_psnr_H = 0
with torch.no_grad():
    for i in range(0, t_test):

        # maxs_cu = YValidation_cu[i, :].max()
        # mins_cu = YValidation_cu[i, :].min()
        maxs_Predicted1_up = YPredicted1_up[i, :].max()
        mins_Predicted1_up = YPredicted1_up[i, :].min()
        # data_n_copy_cu=np.copy(YValidation_cu[i, :]).astype(np.uint16)
        # gray_img_cu = torch.tensor(data_n_copy_cu/(maxs_cu-mins_cu)*255).unsqueeze_(dim=1)

        data_n_copy_Predicted1_up=np.copy(YPredicted1_up[i, :]).astype(np.uint16)
        gray_img_Predicted1_up = torch.tensor(data_n_copy_Predicted1_up/(maxs_Predicted1_up-mins_Predicted1_up)*255).unsqueeze_(dim=1)

        maxs_xi = YValidation_xi[i, :].max()
        mins_xi = YValidation_xi[i, :].min()
        maxs_Predicted2 = YPredicted2[i, :].max()
        mins_Predicted2 = YPredicted2[i, :].min()
        data_n_copy_xi=np.copy(YValidation_xi[i, :]).astype(np.uint16)
        gray_img_xi = torch.tensor(data_n_copy_xi/(maxs_xi-mins_xi)*255).unsqueeze_(dim=1)

        data_n_copy_Predicted2=np.copy(YPredicted2[i, :]).astype(np.uint16)
        gray_img_Predicted2 = torch.tensor(data_n_copy_Predicted2/(maxs_Predicted2-mins_Predicted2)*255).unsqueeze_(dim=1)

        # sample_loss_0 = np.linalg.norm(YPredicted1[i, :] - YValidation_cu[i, :], ord=2) / np.linalg.norm(YValidation_cu[i, :], ord=2)
        # CNN_sample_error_0 = CNN_sample_error_0 + sample_loss_0

        sample_loss_1 = np.linalg.norm(YPredicted1_up[i, :] - YValidation_xi[i, :], ord=2) / np.linalg.norm(YValidation_xi[i, :], ord=2)
        CNN_sample_error_1 = CNN_sample_error_1 + sample_loss_1            # np.linalg.norm()用于求范数，ord=2表示求2范数

        sample_loss_2 = np.linalg.norm(YPredicted2[i, :] - YValidation_xi[i, :], ord=2) / np.linalg.norm(YValidation_xi[i, :], ord=2)
        CNN_sample_error_2 = CNN_sample_error_2 + sample_loss_2

        ssim_L = ssim_psnr.cal_ssim(gray_img_Predicted1_up,gray_img_xi)
        CNN_ssim_L = CNN_ssim_L+ssim_L
        
        psnr_L = ssim_psnr.cal_psnr(gray_img_Predicted1_up,gray_img_xi)
        CNN_psnr_L = CNN_psnr_L+psnr_L

        ssim_H = ssim_psnr.cal_ssim(gray_img_Predicted2,gray_img_xi)
        CNN_ssim_H = CNN_ssim_H+ssim_H

        psnr_H = ssim_psnr.cal_psnr(gray_img_Predicted2,gray_img_xi)
        CNN_psnr_H = CNN_psnr_H+psnr_H

    

Mean_sample_error_0 = CNN_sample_error_0 / t_test
Mean_sample_error_1 = CNN_sample_error_1 / t_test  # 测试集平均归一化重建误差
Mean_sample_error_2 = CNN_sample_error_2 / t_test  # 测试集平均归一化重建误差
Mean_ssim_L = CNN_ssim_L / t_test
Mean_psnr_L = CNN_psnr_L / t_test
Mean_ssim_H = CNN_ssim_H / t_test
Mean_psnr_H = CNN_psnr_H / t_test
print('train_time', train_time)
print('测试集平均归一化重建误差-step1-up', Mean_sample_error_1)  # 0.05583231054137302
print('测试集平均ssim_L-up', Mean_ssim_L)
print('测试集平均psnr_L-up', Mean_psnr_L)
print('测试集平均归一化重建误差-step2', Mean_sample_error_2)  # 0.049486306680964084
print('测试集平均ssim_H', Mean_ssim_H)
print('测试集平均psnr_H', Mean_psnr_H)

result = np.array(YPredicted2_roi)  # 第二步roi超分辨率预测，大小为40x40
mdic1 = {"d": result, "label": "experment"}
savemat("step2_rec_40x40_SNR35_200_R.mat", mdic1)
result = np.array(YPredicted2_all)  # 第二步粗全域超分辨率预测，大小为80x80
mdic1 = {"d": result, "label": "experment"}
savemat("step2_rec_80x80_SNR35_200_A.mat", mdic1)
result = np.array(YPredicted2)  # 第二步融合精细超分辨率预测，大小为80x80
mdic1 = {"d": result, "label": "experment"}
savemat("step2_rec_80x80_SNR35_200_F.mat", mdic1)
