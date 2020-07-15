import torch
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.io as scio
from PIL import Image
from sklearn.decomposition import PCA
from sklearn import preprocessing

device=torch.device('cuda:0')

def pca(input_data):
    pca = PCA(n_components=0.9)
    pca = pca.fit(input_data)
    out_pca= pca.transform(input_data)
    return out_pca

def compute_des_cos_dis(des1,des2):
    #des1=preprocessing.normalize(des1,norm='l2')
    #des2=preprocessing.normalize(des2,norm='l2')
    #des1=pca(des1)
    #des2=pca(des2)
    des1=torch.from_numpy(des1).cuda(device) ####change numpy.array to torch.tensor and change cpu data to gpu data by use .cuda()
    des2=torch.from_numpy(des2).cuda(device)
    des1_T=torch.transpose(des1,0,1) #### 转置
    des2_T=torch.transpose(des2,0,1)
    temp_1=torch.mm(des1,des2_T).cpu().numpy() #### torch.mm 矩阵的点乘 ，data.cpu().numpy() 将gpu的tensor 转化为cpu的numpy
    temp_2=torch.pow(torch.mm(des1,des1_T),0.5).cpu().numpy()
    temp_3=torch.pow(torch.mm(des2,des2_T),0.5).cpu().numpy()
    temp_matrix=np.zeros((temp_2.shape[0],temp_3.shape[0])) #####初始化矩阵
    for i in range(temp_2.shape[0]):
        for j in range(temp_3.shape[0]):
            temp_matrix[i,j]=temp_2[i,i]*temp_3[j,j] ####取对角线元素相乘
    cos_dis_matrix=temp_1/temp_matrix     #####cosine distance
    return cos_dis_matrix



def compute_des_L2_dis(des1,des2):
    #des1=preprocessing.normalize(des1,norm='l2')
    #des2=preprocessing.normalize(des2,norm='l2')
    #des1=pca(des1) 
    #des2=pca(des2)
    des1=torch.from_numpy(des1).cuda(device) ####change numpy.array to torch.tensor and change cpu data to gpu data by use .cuda()
    des2=torch.from_numpy(des2).cuda(device)
    pdist=torch.nn.PairwiseDistance(2)
    dis_matrix=np.zeros((len(des1),len(des2)))
    dim=des1.shape[1]
    for i in range(len(des1)):
        dis_matrix[i]=pdist(des1[i].view(1,dim),des2)
    return dis_matrix

