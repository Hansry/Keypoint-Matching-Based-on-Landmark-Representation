import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import copy
import os


def ORB(img1,kp_num, mask):  #ORB, NO RANSAC

    orb=cv2.ORB_create(kp_num)
    kp1, _ = orb.detectAndCompute(img1, mask)
    return kp1

def SIFTKeyPoint(img1, kp_num, mask):
    sift = cv2.xfeatures2d.SIFT_create(kp_num)
    kp1, _ = sift.detectAndCompute(img1, mask)
    return kp1

if __name__=='__main__':
   img = cv2.imread('/home/hansry/1.png')
   img1 = cv2.imread('/home/hansry/1.png', cv2.COLOR_BGR2GRAY)
   kp1 = ORB(img, 500, None)
   mask = np.ones([img.shape[0], img.shape[1]], dtype = np.uint8)
   mask[0:240, 320:640] = 0

   cv2.imshow("mask", mask)
   cv2.waitKey(0)
   kp2 = ORB(img1, 500, mask)

   print("kp num: ", len(kp2))
   for i in range(len(kp1)):
       (x1, y1) = kp1[i].pt
       cv2.circle(img, (int(np.round(x1)), int(np.round(y1))), 1, (0, 255, 255), 2)

   for j in range(len(kp2)):
       (x2, y2) = kp2[j].pt
       cv2.circle(img1, (int(np.round(x2)), int(np.round(y2))), 1, (0, 255, 255), 2)

   cv2.imshow("img1", img)
   cv2.imshow("img2", img1)
   cv2.waitKey(0)
