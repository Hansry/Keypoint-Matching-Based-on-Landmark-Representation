import torch.nn as nn
import torchvision.models as models  
import torch
import torchvision.transforms as transforms
from PIL import Image
from generate_subimg import generate_subimage 
import numpy as np
from forward import *
import time
from compute_distance import *
from kp_match_same_kp import SIFT,drawMatch,Fundamental,ORB,RootSIFT,Homography
from inlier_pr import plot_pr
from kp_match_same_kp import subimg_match_same_kp, detectPointV2
import cv2
import os
import argparse

cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser(description='Landmark-based Keypoint Matching')
parser.add_argument('-nt', '--network_type', type=str, default="alexnet", choices=["alexnet", "vgg16", "vgg19", "inception_v3", "squeezenet1_0", "resnet"],
                    help = 'The choices of network type to generate the landmark descriptor')
parser.add_argument('-nb', "--network_batch_size", type=int, default=8, help='Network mini-batch size')
parser.add_argument('-db', "--data_batch_size", type=int, default=32, help='Data mini-batch size')
parser.add_argument('-mis', "--model_img_size", type=int, default=224, help='The input img size of network')
parser.add_argument('-st', "--shape_threshold", type=float, default=1.3, help='Shape similarity to match the landmark pair')
parser.add_argument('-dt', "--dataset_type", type=str, default="UAcampus", choices=["UAcampus_gist", "UAcampus", "Mapillary", "Mapillary_gist"], help='Potential loop closure dataset type')
parser.add_argument('-mt', "--match_type", type=str, default="cc", choices=["cc", "dr", "nn"], help='Pruning technique')
parser.add_argument('-wkn', "--whole_kp_num", type=int, default=500, help='Total keypoint num on the whole img')
parser.add_argument('-ln', "--landmark_number", type=int, default = 100, help='Extract landmark number from image')
parser.add_argument('-lmc', "--landmark_constraint", type=float, default=0.6, help='Using a constraint on the size of the bounding box to limit landmark pairs to be matched')
parser.add_argument('-ks', "--keypoint_strategy", type=str, default="low", choices=["low", "normal"], help="The number of the proposed algorithm to extract different keypoints on whole image")
parser.add_argument('-sm', "--show_match", type=bool, default=False, help='Whether to show subimg match')
parser.add_argument('-tp', "--true_positive_number", type=int, default=630, help='True positive image index of potential loop closure dataset')
parser.add_argument('-fp', "--false_positive_number", type=int, default=10, help='False positive image index of potential loop closure dataset')
parser.add_argument('-sfr', "--show_final_result", type=bool, default=False, help='Whether show final result')
parser.add_argument('-tsfr', "--save_final_result", type=bool, default=True, help='Whether to save final result')


args = parser.parse_args()
device=torch.device('cuda:0') #调用gpu:0
mynet = models.alexnet(pretrained=True).cuda(device) #default network
if args.network_type == "vgg16":
   mynet = models.vgg16(pretrained=True).cuda(device)
elif args.network_type == "vgg19":
   mynet = models.vgg19(pretrained=True).cuda(device)
elif args.network_type == "inception_v3":
   mynet = models.inception_v3(pretrained=True).cuda(device)
elif args.network_type == "squeezenet1_0":
   mynet = models.squeezenet1_0(pretrained=True).cuda(device)
elif args.network_type == "resnet":
   mynet = models.resnet18(pretrained=True).cuda(device)
mynet.eval()

######class#########
class generate_des:
    def __init__(self, net, img_tensor, mini_batch_size=args.network_batch_size, net_type=args.network_type):
        self.descriptor = self.extract_batch_conv_features(net, img_tensor, mini_batch_size, net_type)

    #####extract batch conv features#####
    def extract_batch_conv_features(self, net, input_data, mini_batch_size, net_type):
        batch_number = int(len(input_data)/mini_batch_size)
        descriptor_init = self.extract_conv_features(net, input_data[:mini_batch_size], net_type).cpu().detach().numpy()
        #start=time.time()
        for i in range(1, batch_number):
            mini_batch = input_data[mini_batch_size*i:mini_batch_size*(i+1)]
            temp_descriptor = self.extract_conv_features(net, mini_batch, net_type).cpu().detach().numpy()
            descriptor_init = np.vstack((descriptor_init, temp_descriptor))
        #end=time.time()
        #print('加载数据耗时:'+str(end-start))
        #####avoid the last mini_batch is NULL######
        if (len(input_data) % mini_batch_size == 0):
            return descriptor_init 
        descriptor = self.extract_conv_features(net, input_data[mini_batch_size*batch_number:len(input_data)+1], net_type).cpu().detach().numpy()
        #####aviod the batch_number=0######
        if batch_number > 0:
            descriptor = np.vstack((descriptor_init, descriptor))
        return descriptor

    #####extract conv features#####
    def extract_conv_features(self, net, input_data,net_type):
        if net_type.startswith('alexnet'):####alexnet is in forward.py
            x = alexnet(net, input_data)
        elif net_type.startswith('vgg16'):
            x = vgg16(net, input_data)
        elif net_type.startswith('vgg19'):
            x = vgg19(net, input_data)
        elif net_type.startswith('inception_v3'):
            x = inception_v3(net, input_data)
        elif net_type.startswith('squeezenet1_0'):
            x = squeezenet1(net, input_data)
        elif net_type.startswith('resnet'):
            x = resnet(net, input_data)
        return x

#####分批进行计算.如果一起读入，数据太大，读取时间太长，不能充分利用GPU.
def compute_batch_descriptor(net,input_data, mini_batch_size):
    batch_number = int(len(input_data)/mini_batch_size)  #####计算将数据分成了多少块
    descriptor_init = generate_des(mynet, input_data[:mini_batch_size].cuda(device)).descriptor ####第一个mini_batch
    start=time.time()
    ######循环读入mini_batch#######
    for i in range(1, batch_number):
        mini_batch = input_data[mini_batch_size*i:mini_batch_size*(i+1)]
        temp_descriptor = generate_des(mynet, mini_batch.cuda(device)).descriptor
        descriptor_init = np.vstack((descriptor_init, temp_descriptor)) ####将描述符叠起来
    end = time.time()
    #print("计算描述符共耗时:"+str(end-start))
    #####avoid the last mini_batch is NULL######
    if (len(input_data)%mini_batch_size==0):
        return descriptor_init
    #####计算最后一块，可能为0，数据块小于mini_batch的大小
    descriptor=generate_des(net, input_data[mini_batch_size*batch_number:len(input_data)+1].cuda(device)).descriptor
    #####aviod the batch_number=0######
    if batch_number > 0:
        descriptor = np.vstack((descriptor_init, descriptor))
    return descriptor    

def compute_max_xy(dis_matrix,img_left_txt,img_right_txt):

    Max_Matches = np.dtype({'names': ['MAX', 'i', 'j'], 'formats': ['f', 'i', 'i']})
    M1 = np.array([(0, -1, -1)]*args.landmark_number, dtype=Max_Matches)
    M2 = np.array([(0, -1, -1)]*args.landmark_number, dtype=Max_Matches)

    for i in range(args.landmark_number):
        MAX_COS_DIS = np.max(dis_matrix[i, 0:args.landmark_number-1])
        M1[i]['MAX'] = MAX_COS_DIS
        M1[i]['i'] = i
        M1[i]['j'] = np.argmax(dis_matrix[i, 0:args.landmark_number-1])

    for j in range(args.landmark_number):
        MAX_COS_DIS = np.max(dis_matrix[0:args.landmark_number-1, j])
        M2[i]['MAX'] = MAX_COS_DIS
        M2[j]['i'] = j
        M2[j]['j'] = np.argmax(dis_matrix[0:args.landmark_number-1, j])
    
    MAX_XY=[]
    box_a=np.loadtxt(img_left_txt, skiprows=1, dtype=np.float, delimiter=',')
    box_b=np.loadtxt(img_right_txt, skiprows=1, dtype=np.float, delimiter=',')
    for i in range(args.landmark_number):
      if M2[M1[i]['j']]['j']==i:
         width_a=box_a[i][3]-box_a[i][1]
         lenth_a=box_a[i][4]-box_a[i][2]
         width_b=box_b[M1[i]['j']][3]-box_b[M1[i]['j']][1]
         lenth_b=box_b[M1[i]['j']][4]-box_b[M1[i]['j']][2]
         max_width=max(width_a,width_b)
         max_lenth=max(lenth_a,lenth_b)
         min_width=min(width_a,width_b)
         min_lenth=min(lenth_a,lenth_b)
         if max_width <= args.shape_threshold*min_width and max_lenth <= args.shape_threshold*min_lenth:
         #   print("the left image {} best matches is right image {},cosine value is {}".format(M1[i]['i'],M1[i]['j'],M1[i]['MAX']))
            MAX_XY.append([M1[i]['i'], M1[i]['j'], M1[i]['MAX']])
    return MAX_XY

def del_repeat(total_left_kp, total_right_kp):
    unique = []
    if len(total_left_kp) == 0:
       return 0
    total_kp=np.append(total_left_kp,total_right_kp, axis=1)
    total=total_kp.tolist()
    for i in range(len(total)):
        if total[i] not in unique:
           unique.append(total[i])
    return len(unique)

#先匹配路标上，然后在路标的基础上进行特征点的匹配
def Landmark_match_same_kp(img_left, img_right, img_left_txt, img_right_txt, keypoint_type, img_num):

    img_tensor_left=generate_subimage(img_left_txt, img_left, args.landmark_number, args.model_img_size)
    img_tensor_right=generate_subimage(img_right_txt, img_right, args.landmark_number, args.model_img_size)
    desc_left=compute_batch_descriptor(mynet, img_tensor_left, args.data_batch_size)
    desc_right=compute_batch_descriptor(mynet, img_tensor_right, args.data_batch_size)
    dis_matrix=compute_des_cos_dis(desc_left, desc_right)  #matrx of distance
    MAX_XY=compute_max_xy(dis_matrix, img_left_txt, img_right_txt)

    #def subimg_match_same_kp(img_left, img_right, img_left_txt, img_right_txt, MAX_XY, keypoint_type,match_type,kpNum,subimg_show)
    ########### match with same keypoint ###########
    total_left_kp, total_right_kp, OriginKpNum = subimg_match_same_kp(img_left, img_right, img_left_txt, img_right_txt, MAX_XY, keypoint_type, args)

    ##############################################
    total_left_kp, total_right_kp = Fundamental(total_left_kp, total_right_kp)
    img_left = cv2.imread(img_left)
    img_right = cv2.imread(img_right)

    drawMatch(img_left, img_right, total_left_kp, total_right_kp, str(str(img_num)+"LM: "+keypoint_type), args)
    inliers = del_repeat(total_left_kp, total_right_kp)  # 去掉重复的点
    return inliers, total_left_kp, total_right_kp, OriginKpNum

def Origin_match(img_left,img_right,keypoint_type,img_num):
    img_left = cv2.imread(img_left)
    img_right = cv2.imread(img_right)

    if keypoint_type=="orb":
       kp_query_orb, kp_train_orb, _, _ = ORB(img_left, img_right, args, args.whole_kp_num)
       kp_query_orb, kp_train_orb = Fundamental(kp_query_orb, kp_train_orb)
       drawMatch(img_left, img_right, kp_query_orb, kp_train_orb, str(img_num)+keypoint_type, args)
       inliers=len(kp_query_orb)
    elif keypoint_type=="sift":
       kp_query_sift, kp_train_sift, _, _ = SIFT(img_left, img_right, args, args.whole_kp_num)
       kp_query_sift, kp_train_sift = Fundamental(kp_query_sift, kp_train_sift)
       drawMatch(img_left, img_right, kp_query_sift, kp_train_sift, str(img_num)+keypoint_type, args)
       inliers=len(kp_query_sift)
    elif keypoint_type=="rootsift":
       kp_query_rootsift, kp_train_rootsift, _, _ = RootSIFT(img_left, img_right, args, args.whole_kp_num)
       kp_query_rootsift, kp_train_rootsift=Fundamental(kp_query_rootsift, kp_train_rootsift)
       drawMatch(img_left, img_right, kp_query_rootsift, kp_train_rootsift, str(img_num)+keypoint_type, args)
       inliers=len(kp_query_rootsift)
    return inliers

def calculate_inliers(img_number,A,B):
    ORB_LM=[]
    ORB=[]
    SIFT_LM=[]
    SIFT=[]
    RootSIFT_LM=[]
    RootSIFT=[]

    totalOriginKpNumORB = 0
    totalOriginKpNumSIFT = 0
    count = 0
    for index in range(img_number):
   # for index in [127, 28]:
        count += 1
        print("process img:"+str(index))
        img_left = './dataset/'+str(args.dataset_type)+'/AB/'+A+'/'+str(index)+'.jpg'
        img_right = './dataset/'+str(args.dataset_type)+'/AB/'+B+'/'+str(index)+'.jpg'
        img_left_txt = './dataset/'+str(args.dataset_type)+'/AB_txt/'+A+'/'+str(index)+'.txt'
        img_right_txt = './dataset/'+str(args.dataset_type)+'/AB_txt/'+B+'/'+str(index)+'.txt'

        if os.access(img_left,os.F_OK):
              #def Landmark_match_same_kp(img_left,img_right,img_left_txt,img_right_txt,keypoint_type,img_num):

              LM_ORB_inliers, _, _, OriginKpNumORB = Landmark_match_same_kp(img_left, img_right, img_left_txt, img_right_txt, "orb", index)
              print('LM_ORB_inliers:' + str(LM_ORB_inliers))
              ORB_LM.append(LM_ORB_inliers)
              LM_SIFT_inliers, _, _, OriginKpNumSIFT = Landmark_match_same_kp(img_left, img_right, img_left_txt, img_right_txt, "sift", index)
              SIFT_LM.append(LM_SIFT_inliers)
              print('LM_SIFT_inliers:' + str(LM_SIFT_inliers))
              LM_RootSIFT_inliers, _, _, OriginKpNumRootSIFT = Landmark_match_same_kp(img_left, img_right, img_left_txt, img_right_txt, "rootsift", index)
              print('RootSIFT_LM_inliers:' + str(LM_RootSIFT_inliers))
              RootSIFT_LM.append(LM_RootSIFT_inliers)


              orb_cc_inliers = Origin_match(img_left, img_right, "orb", index)
              ORB.append(orb_cc_inliers)
              print('ORB_inliers:'+str(orb_cc_inliers))
              sift_dr_inliers = Origin_match(img_left, img_right, "sift", index)
              SIFT.append(sift_dr_inliers)
              print('SIFT_inliers:'+str(sift_dr_inliers))
              RootSIFT_CC_inliers = Origin_match(img_left, img_right, "rootsift", index)
              print('RootSIFT_inliers:'+str(RootSIFT_CC_inliers))
              RootSIFT.append(RootSIFT_CC_inliers)

              totalOriginKpNumORB += OriginKpNumORB
              totalOriginKpNumSIFT += OriginKpNumSIFT

              print("Average ORB extraction: ", float(totalOriginKpNumORB)/float(count))
              print("Average SIFT extraction: ", float(totalOriginKpNumSIFT)/float(count))

        else:
           continue
    return ORB_LM,ORB,SIFT_LM,SIFT,RootSIFT_LM,RootSIFT

if __name__=="__main__":
    imgNumber1 = args.true_positive_number #正样本数据集
    imgNumber2 = args.false_positive_number #负样本数据集

    True_ORB_LM, True_ORB,True_SIFT_LM, True_SIFT, True_RootSIFT_LM, True_RootSIFT = calculate_inliers(imgNumber1, 'AY', 'BY')
    args.show_final_result = False
    args.save_final_result = False
    False_ORB_LM, False_ORB, False_SIFT_LM, False_SIFT, False_RootSIFT_LM, False_RootSIFT = calculate_inliers(imgNumber2, 'AN', 'BN')


    print("True positive inliers")
    print("True_ORB_LM"+str(True_ORB_LM))
    print("True_ORB"+str(True_ORB))
    print("True_SIFT_LM"+str(True_SIFT_LM))
    print("True_SIFT"+str(True_SIFT))
    print("True_RootSIFT_LM"+str(True_RootSIFT_LM))
    print("True_RootSIFT"+str(True_RootSIFT))


    print("False positive inliers")
    print("False_ORB_LM"+str(False_ORB_LM))
    print("False_ORB"+str(False_ORB))
    print("False_SIFT_LM"+str(False_SIFT_LM))
    print("False_SIFT"+str(False_SIFT))
    print("False_RootSIFT_LM"+str(False_RootSIFT_LM))
    print("False_RootSIFT"+str(False_RootSIFT))


    plot_pr(True_ORB_LM, False_ORB_LM, 200, 'ORB_LM', 'b', '-')
    plot_pr(True_ORB, False_ORB, 200, 'ORB_CC', 'b', '--')
    plot_pr(True_SIFT_LM, False_SIFT_LM, 200, 'SIFT_LM', 'black', '-')
    plot_pr(True_SIFT, False_SIFT, 200, 'SIFT', 'black', '--')
    plot_pr(True_RootSIFT_LM, False_RootSIFT_LM, 200, 'RootSIFT_LM', 'y', '-')
    plot_pr(True_RootSIFT, False_RootSIFT, 200, 'RootSIFT', 'y', '--')

    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.01)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ## 设置坐标标签字体大小
    plt.xlabel('Recall'+str(args.dataset_type), fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.grid(linestyle=':', alpha=1, linewidth=0.8)
    plt.legend(fontsize=11)
    plt.show()
    
 
