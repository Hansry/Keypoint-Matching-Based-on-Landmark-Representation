import torch.nn as nn
import torchvision.models as models  
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from forward import *
import time
from compute_distance import *
import scipy.io as scio
import matplotlib.pyplot as plt
import os
import cv2
import argparse


parser = argparse.ArgumentParser(description='Landmark-based Keypoint Matching')
parser.add_argument('-nt', '--network_type', type=str, default="alexnet", choices=["alexnet", "vgg16", "vgg19", "inception_v3", "squeezenet1_0", "resnet"],
                     help='The choices of network type to generate the landmark descriptor')
parser.add_argument('-mis', "--model_img_size", type=int, default=224, help='The input img size of network')
parser.add_argument('-dt', "--dataset_type", type=str, default="UAcampus", choices=["UAcampus", "Mapillary"], help='Potential loop closure dataset generation')
parser.add_argument('-in', "--img_number", type=int, default=647, choices=[647, 1301], help='Image number of dataset')
parser.add_argument('-dth', "--distance_threshold", type=float, default=0.7, choices=[0.7, 0.78], help='The distance threshold to accept the potential loop closure')
parser.add_argument('-is', "--img_suffix", type=str, default=".jpg", choices=[".jpg", ".png"], help='The choices of image suffix')


args = parser.parse_args()
device=torch.device('cuda:0')
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
img_to_tensor = transforms.ToTensor()


#ConvNet feature extraction class
class generate_des:
    def __init__(self, net, img_tensor, mini_batch_size=8, net_type=args.network_type):
        self.descriptor = self.extract_batch_conv_features(net, img_tensor, mini_batch_size, net_type)

    #extract batch ConvNet features
    def extract_batch_conv_features(self, net, input_data, mini_batch_size, net_type):

        batch_number = int(len(input_data)/mini_batch_size)
        descriptor_init = self.extract_conv_features(net, input_data[:mini_batch_size], net_type).cpu().detach().numpy()

        for i in range(1, batch_number):
            mini_batch = input_data[mini_batch_size*i:mini_batch_size*(i+1)]
            temp_descriptor = self.extract_conv_features(net, mini_batch, net_type).cpu().detach().numpy()
            descriptor_init = np.vstack((descriptor_init, temp_descriptor))

        #avoid the last mini_batch is NULL
        if (len(input_data) % mini_batch_size == 0):
            return descriptor_init 
        descriptor = self.extract_conv_features(net, input_data[mini_batch_size*batch_number:len(input_data)+1], net_type).cpu().detach().numpy()

        #aviod the batch_number equal to zero
        if batch_number > 0:
            descriptor = np.vstack((descriptor_init, descriptor))
        return descriptor

    #extract ConvNet features
    def extract_conv_features(self, net, input_data, net_type):

        if net_type.startswith('alexnet'):
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

def read_image(Img_path):
    img = Image.open(Img_path)
    img = img.resize((args.model_img_size, args.model_img_size))
    img = img_to_tensor(img)
    img = img.numpy()
    img = img.reshape(3, args.model_img_size, args.model_img_size)
    return img

def change_images_to_tensor(Img_path):
    img_list = []
    start = time.time()
    for i in range(args.img_number):
        _Img_path = Img_path+str(i)+args.img_suffix
        img = read_image(_Img_path)
        img_list.append(img)
    img_array = np.array(img_list)
    img_tensor = torch.from_numpy(img_array)
    end = time.time()
    print('Loading image time:'+str(end-start))
    return img_tensor

#compute batch ConvNet descriptor for potential loop closure pair generation
def compute_batch_descriptor(net,input_data,mini_batch_size):
    batch_number = int(len(input_data)/mini_batch_size)
    descriptor_init = generate_des(mynet, input_data[:mini_batch_size].cuda(device)).descriptor

    start = time.time()
    for i in range(1, batch_number):
        mini_batch = input_data[mini_batch_size*i:mini_batch_size*(i+1)]
        temp_descriptor = generate_des(mynet,mini_batch.cuda(device)).descriptor
        descriptor_init = np.vstack((descriptor_init, temp_descriptor))
    end = time.time()
    print("Calculating the descriptor time:"+str(end-start))

    if (len(input_data) % mini_batch_size == 0):
        return descriptor_init

    descriptor = generate_des(net, input_data[mini_batch_size*batch_number:len(input_data)+1].cuda(device)).descriptor

    if batch_number > 0:
        descriptor = np.vstack((descriptor_init, descriptor))

    return descriptor    

def compute_ground_true():
    groundTruth_path = './dataset/'+args.dataset_type+'/'+args.dataset_type+'GroundTruth.mat'
    data = scio.loadmat(groundTruth_path)
    ground_true = data[args.dataset_type]
    ground_true = np.array(ground_true, dtype=np.float32)
    return ground_true

def compute_PR_v2(dis_matrix):
    ground_true = compute_ground_true()
    predict_match_num = 0
    actual_match_num = 0
    for i in range(0, args.img_number):
        row_vector = dis_matrix[i]
        row_max = np.max(row_vector)
        index_row_max = np.argmax(row_vector)
        if row_max >= args.distance_threshold:
            predict_match_num = predict_match_num+1
            if ground_true[i, index_row_max]==1:
                actual_match_num = actual_match_num + 1
    if predict_match_num == 0:
        return 0, 0
    
    print("The number of True Positive："+str(actual_match_num))
    print("The number of the predicted loop closure："+str(predict_match_num))
    P = float(actual_match_num/predict_match_num)
    R = float(predict_match_num/args.img_number)
    print('Precision:'+str(P))
    print('Recall:'+str(R))
    return P, R

def generate_dataset(Img_path_left,Img_path_right,dis_matrix):
    ground_true=compute_ground_true()
    predict_match_num = 0
    actual_match_num = 0
    actual_num_order = 0
    wrong_predict_num = 0
    wrong_num_order = 0
    for i in range(0, args.img_number):
        row_vector = dis_matrix[i]
        row_max = np.max(row_vector)
        index_row_max = np.argmax(row_vector)
        if row_max >= args.distance_threshold:
           predict_match_num = predict_match_num+1
           if ground_true[i, index_row_max] == 1:
              img_AY = cv2.imread(Img_path_left+str(i) + args.img_suffix)
              img_BY = cv2.imread(Img_path_right+str(index_row_max) + args.img_suffix)
              save_dir_AY = './dataset/' + args.dataset_type + '/AB/AY'
              save_dir_BY = './dataset/' + args.dataset_type + '/AB/BY'
              if os.path.exists(save_dir_AY) == False:
                 os.makedirs(save_dir_AY)
              if os.path.exists(save_dir_BY) == False:
                 os.makedirs(save_dir_BY)
              cv2.imwrite(save_dir_AY + '/' + str(actual_num_order) + args.img_suffix, img_AY)
              cv2.imwrite(save_dir_BY + '/' + str(actual_num_order) + args.img_suffix, img_BY)
              actual_match_num = actual_match_num+1
              actual_num_order = actual_num_order+1
           else:
              img_AN = cv2.imread(Img_path_left+str(i) + args.img_suffix)
              img_BN = cv2.imread(Img_path_right+str(index_row_max) + args.img_suffix)
              save_dir_AN = './dataset/' + args.dataset_type + '/AB/AN'
              save_dir_BN = './dataset/' + args.dataset_type + '/AB/BN'
              if os.path.exists(save_dir_AN) == False:
                  os.makedirs(save_dir_AN)
              if os.path.exists(save_dir_BN) == False:
                  os.makedirs(save_dir_BN)
              cv2.imwrite(save_dir_AN + '/' + str(wrong_num_order) + args.img_suffix, img_AN)
              cv2.imwrite(save_dir_BN + '/' + str(wrong_num_order) + args.img_suffix, img_BN)
              wrong_predict_num = wrong_predict_num+1
              wrong_num_order = wrong_num_order+1

    if predict_match_num == 0:
        return 0, 0
    P = float(actual_match_num/predict_match_num)
    R = float(predict_match_num/args.img_number)
    print('Precision:'+str(P))
    print('Recall:'+str(R))
    return P, R

if __name__=="__main__":
    Img_path_left = './dataset/' + args.dataset_type + '/' + args.dataset_type + '_query/'
    Img_path_right = './dataset/' + args.dataset_type + '/' + args.dataset_type + '_train/'

    img_tensor_left = change_images_to_tensor(Img_path_left)
    img_tensor_right = change_images_to_tensor(Img_path_right)

    desc_left = compute_batch_descriptor(mynet, img_tensor_left, 128)
    desc_right = compute_batch_descriptor(mynet, img_tensor_right, 128)
    dis_matrix = compute_des_cos_dis(desc_left, desc_right)
    max_number = np.max(dis_matrix)
    min_number = np.min(dis_matrix)
    
    #generate dataset
    print("generate dataset")
    P, R = generate_dataset(Img_path_left, Img_path_right, dis_matrix)
    print("finish generate dataset")

    """
    step_number=100
    P_list=[]
    R_list=[]
    AP=[]
    threshold_list=np.linspace(min_number,max_number,step_number)
    threshold_list=sorted(threshold_list)
    print("threshold_list"+str(threshold_list))
    for i in range(len(threshold_list)):
        threshold=threshold_list[i]
        P,R=compute_PR_v2(Img_num,dis_matrix)
        P_list.append(P)
        R_list.append(R)
    #P_list.reverse() 
    #R_list.reverse()
    print("Precision:",str(P_list))
    print("Recall:",str(R_list))
    AP.append(P_list[0]*R_list[0])
    for i in range(1,step_number):
        AP.append(P_list[i]*(R_list[i]-R_list[i-1]))
    mAP=np.sum(AP)
    print("mAP:"+str(mAP))
    recall_100=[]
    for i in range(len(P_list)):
        if P_list[i]==1.0:
            recall_100.append(R_list[i])
    #print("Recall at 100% precision："+str(recall_100))
    #Draw the pr curve
    P_array=np.array(P_list)
    print(P_array)
    R_array=np.array(R_list)
    print(R_array)
    threshold_array=np.array(threshold_list)
#    P_array=P_array[P_array>0]
#    R_array=R_array[R_array>0]
#    R_array.tolist()
#    P_array.tolist()
#    print('R_array'+str(R_array))
#    print('P_array'+str(P_array))
    plt.grid(True)
    plt.plot(threshold_array,P_array,label='Distance-Precision')
    plt.plot(threshold_array,R_array,label='Distance-Recall')
    plt.show()
    """