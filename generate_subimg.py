# generate_subimg.py 

import cv2
import numpy as np
import torchvision.transforms as transforms
import torch

img_to_tensor=transforms.ToTensor()

#generate subimage
def process_image(img,model_img_size):
    img=cv2.resize(img,(model_img_size,model_img_size))
    img=img_to_tensor(img)
    img=img.numpy()
    img=img.reshape(3,model_img_size,model_img_size)
    return img


def generate_subimage(image_box_txt,image,landmark_number,model_img_size):
  box_coord=np.loadtxt(image_box_txt, skiprows=1,dtype=np.float, delimiter=',')
  img=cv2.imread(image)
  subimg_list=[]
  for i in range(landmark_number):
      box_coord[i][4]=box_coord[i][4]-1
      imgcrop=img[int(box_coord[i][2]):int(box_coord[i][4]),int(box_coord[i][1]):int(box_coord[i][3])]
      imgcrop=process_image(imgcrop,model_img_size)
      subimg_list.append(imgcrop)
  subimg_array=np.array(subimg_list)
  subimg_tensor=torch.from_numpy(subimg_array)
  return subimg_tensor


if __name__=="__main__":
  image_box_txt="./dataset/1.txt"
  imageA="./dataset/1.jpg"
  landmark_number=100
  model_imge_size=120
  subimg_list=generate_subimage(image_box_txt,imageA,landmark_number,120)


