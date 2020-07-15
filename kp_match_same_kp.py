#kp_match_same_kp.py

import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import copy
import os

MIN_MATCH_COUNT = 0
cv2.ocl.setUseOpenCL(False)
ratio=0.8

def combineImage(img1,img2):  #Easy to visulize
    rows1=img1.shape[0]
    rows2=img2.shape[0]
    if rows1<rows2:   #Judge the size of two image
        concat=np.zeros((rows2-rows1,img1.shape[1],3),dtype=np.uint8)
        img1=np.concatenate((img1,concat),axis=0) #padding
    if rows1>rows2:
        concat=np.zeros((rows1-rows2,img2.shape[1],3),dtype=np.uint8)
        img2=np.concatenate((img2,concat),axis=0)
    combine_img=np.concatenate((img1,img2), axis=1) #padding
    return combine_img

def SIFTKeyPoint(img1, img2, kp_num):
    sift = cv2.xfeatures2d.SIFT_create(kp_num)
    kp1, _ = sift.detectAndCompute(img1, None)
    kp2, _ = sift.detectAndCompute(img2, None)
    return kp1, kp2

def ORBKeyPoint(img1,img2,kp_num):  #ORB, NO RANSAC

    orb=cv2.ORB_create(kp_num)
    kp1,_=orb.detectAndCompute(img1,None)
    kp2,_=orb.detectAndCompute(img2,None)
    return kp1, kp2

def RootSIFT(img1,img2,args,kp_num):

    sift = cv2.xfeatures2d.SIFT_create(kp_num)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    returnKp1 = kp1
    returnKp2 = kp2

    eps=1e-7
    if len(kp1)==0:
        return [],[],[],[]
    des1/=(des1.sum(axis=1,keepdims=True)+eps)
    des1=np.sqrt(des1)
    if len(kp2)==0:
        return [],[],[],[]
    des2/=(des2.sum(axis=1,keepdims=True)+eps)
    des2=np.sqrt(des2)

    if args.match_type == "nn":
       bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
       if len(kp1) >= 2 and len(kp2) >= 2:
           matches = bf.match(des1, des2)
       else:
           return [], [], [],[]
    if args.match_type == "cc":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        if len(kp1) >= 2 and len(kp2) >= 2:
            matches = bf.match(des1, des2)
        else:
            return [],[],[],[]
    if args.match_type == "dr":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        if len(kp1) >= 2 and len(kp2) >= 2:
            matches = bf.knnMatch(des1, des2, k=2)
        else:
            return [],[],[],[]
            # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:  # DR
                good.append(m)
        matches = good

    querypoint=[]
    trainpoint=[]
    if len(matches) >= MIN_MATCH_COUNT:
       for j in range(len(matches)):
           querypoint.append(kp1[matches[j].queryIdx].pt)
           trainpoint.append(kp2[matches[j].trainIdx].pt)
       kp_query=np.array(querypoint)
       kp_train=np.array(trainpoint)

       if args.show_match==True:
          cols1=img1.shape[1]
          combine_img=combineImage(img1,img2)
          if len(kp_query)==0:
             return [],[],returnKp1,returnKp2
          for i in range(len(kp_query)):
             (x1,y1)=kp_query[i]
             (x2,y2)=kp_train[i]
             cv2.circle(combine_img,(int(np.round(x1)),int(np.round(y1))),1,(0,255,255),2)
             cv2.circle(combine_img,(int(np.round(x2)+cols1),int(np.round(y2))),1,(0,0,255),2)
             cv2.line(combine_img, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (255, 0, 0), 1, lineType=cv2.LINE_AA, shift=0)
          cv2.imshow('RooSIFT', combine_img)
          cv2.waitKey(0)

       return kp_query,kp_train,returnKp1,returnKp2
    else:
       print ("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT))
       return [], [],returnKp1,returnKp2

def ORB(img1,img2,args,kp_num):  #ORB, NO RANSAC

    orb=cv2.ORB_create(kp_num)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    returnKp1 = kp1
    returnKp2 = kp2

    if args.match_type == "nn":
       bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
       if len(kp1) >= 2 and len(kp2) >= 2:
           matches = bf.match(des1, des2)
       else:
           return [], [], [],[]
    elif args.match_type == "cc":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if len(kp1) >= 2 and len(kp2) >= 2:
            matches = bf.match(des1, des2)
        else:
            return [],[],[],[]
    elif args.match_type == "dr":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        if len(kp1) >= 2 and len(kp2) >= 2:
            matches = bf.knnMatch(des1, des2, k=2)
        else:
            return [],[],[],[]
            # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:  # DR
                good.append(m)
        matches = good
    querypoint=[]
    trainpoint=[]
    if len(matches) >= MIN_MATCH_COUNT: #Judge the number of matches
        for j in range(len(matches)):
            querypoint.append(kp1[matches[j].queryIdx].pt)
            trainpoint.append(kp2[matches[j].trainIdx].pt)
        kp_query=np.array(querypoint)
        kp_train=np.array(trainpoint)

        if args.show_match == True:
           cols1=img1.shape[1]
           combine_img=combineImage(img1,img2)
           for i in range(len(kp_query)):
              (x1,y1)=kp_query[i]
              (x2,y2)=kp_train[i]
              cv2.circle(combine_img,(int(np.round(x1)),int(np.round(y1))),1,(0,255,255),2)
              cv2.circle(combine_img,(int(np.round(x2)+cols1),int(np.round(y2))),1,(0,0,255),2)
              cv2.line(combine_img, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (255, 0, 0), 1, lineType=cv2.LINE_AA, shift=0)
           cv2.imshow('ORB_CC',combine_img)
           cv2.waitKey(0)

        return kp_query,kp_train,returnKp1,returnKp2
    else:
        print ("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT))
        return [], [], returnKp1,returnKp2

def SIFT(img1,img2,args,kp_num):  #SIFT NO RANSAC

  # Initiate SIFT detector

  sift = cv2.xfeatures2d.SIFT_create(kp_num)
  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)

  returnKp1 = kp1
  returnKp2 = kp2

  is_show=False

  ##show keypoint##
  if args.match_type == "nn":
     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
     if len(kp1) >= 2 and len(kp2) >= 2:
         matches = bf.match(des1, des2)
     else:
         return [], [], [],[]
  if args.match_type == "cc":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        if len(kp1) >= 2 and len(kp2) >= 2:
            matches = bf.match(des1, des2)
        else:
            return [],[],[],[]
  if args.match_type == "dr":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        if len(kp1) >= 2 and len(kp2) >= 2:
            matches = bf.knnMatch(des1, des2, k=2)
        else:
            return [],[],[],[]
            # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:  # DR
                good.append(m)
        matches = good
  querypoint=[]
  trainpoint=[]
  if len(matches) >= MIN_MATCH_COUNT:
      for j in range(len(matches)):
         querypoint.append(kp1[matches[j].queryIdx].pt)
         trainpoint.append(kp2[matches[j].trainIdx].pt)
      kp_query=np.array(querypoint)
      kp_train=np.array(trainpoint)

      if args.show_match == True:
         cols1=img1.shape[1]
         combine_img=combineImage(img1,img2)
         for i in range(len(kp_query)):
            (x1,y1)=kp_query[i]
            (x2,y2)=kp_train[i]
            cv2.circle(combine_img,(int(np.round(x1)),int(np.round(y1))),1,(0,255,255),2)
            cv2.circle(combine_img,(int(np.round(x2)+cols1),int(np.round(y2))),1,(0,0,255),2)
            cv2.line(combine_img, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (255, 0, 0), 1, lineType=cv2.LINE_AA, shift=0)
         cv2.imshow('SIFT_DR',combine_img)
         cv2.waitKey(0)

      return kp_query,kp_train,returnKp1,returnKp2
  else:
    print ("Not enough matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT))
    return [],[],returnKp1,returnKp2

def del_repeat(total_left_kp, total_right_kp):
    unique = []
    if len(total_left_kp) == 0:
        return 0
    total_kp = np.append(total_left_kp, total_right_kp, axis=1)
    total = total_kp.tolist()
    for i in range(len(total)):
        if total[i] not in unique:
            unique.append(total[i])
    total_left_kp=[]
    total_right_kp=[]
    for j in range(len(unique)):
        total_left_kp.append((unique[j][0],unique[j][1]))
        total_right_kp.append((unique[j][2],unique[j][3]))
    return total_left_kp,total_right_kp

#删除重复检测特征点
def detectPoint(detectKp):
    uniquePoint = []
    overlapKp = 0
    for i in range(len(detectKp)):
        if(detectKp[i]==0):
            continue
        for j in range(len(detectKp[i])):
            kpPt = detectKp[i][j].pt
            if kpPt not in uniquePoint:
               uniquePoint.append(kpPt)
    return len(uniquePoint)

def detectPointV2(detectKp):
    uniquePoint = []
    for i in range(len(detectKp)):
        kpPt = detectKp[i].pt
        if kpPt not in uniquePoint:
            uniquePoint.append(kpPt)
    return len(uniquePoint)

#在子图中进行sift特征点匹配并返回总的特征点个数
def subimg_match_same_kp(img_left,img_right,img_left_txt,img_right_txt,MAX_XY,keypoint_type,args):
    box_coord_left=np.loadtxt(img_left_txt, skiprows=1, delimiter=',')
    box_coord_right=np.loadtxt(img_right_txt, skiprows=1, delimiter=',')

    img_left=cv2.imread(img_left)
    img_right=cv2.imread(img_right)

    #约束sub_img的大小
    MAX_XY_temp=[]
    for i in range(len(MAX_XY)):
        MAX_XY[i][0] = int(MAX_XY[i][0])
        MAX_XY[i][1] = int(MAX_XY[i][1])
        imgcrop_left = img_left[int(box_coord_left[MAX_XY[i][0]][2]):int(box_coord_left[MAX_XY[i][0]][4]), int(box_coord_left[MAX_XY[i][0]][1]):int(box_coord_left[MAX_XY[i][0]][3])]
        imgcrop_right = img_right[int(box_coord_right[MAX_XY[i][1]][2]):int(box_coord_right[MAX_XY[i][1]][4]), int(box_coord_right[MAX_XY[i][1]][1]):int(box_coord_right[MAX_XY[i][1]][3])]
        if imgcrop_left.shape[0] <= args.landmark_constraint*img_left.shape[0] and imgcrop_left.shape[1] <= args.landmark_constraint*img_left.shape[1]:#约束路标的大小
           MAX_XY_temp.append(MAX_XY[i])
    MAX_XY=MAX_XY_temp

    DetectTotalKpLeft = []
    DetectTotalKpRight = []

    wholeNum = args.whole_kp_num

    for i in range(len(MAX_XY)):
        MAX_XY[i][0] = int(MAX_XY[i][0])
        MAX_XY[i][1] = int(MAX_XY[i][1])
        imgcrop_left = img_left[int(box_coord_left[MAX_XY[i][0]][2]):int(box_coord_left[MAX_XY[i][0]][4]),int(box_coord_left[MAX_XY[i][0]][1]):int(box_coord_left[MAX_XY[i][0]][3])]
        imgcrop_right = img_right[int(box_coord_right[MAX_XY[i][1]][2]):int(box_coord_right[MAX_XY[i][1]][4]),int(box_coord_right[MAX_XY[i][1]][1]):int(box_coord_right[MAX_XY[i][1]][3])]
        kp_num = int(float(wholeNum)/len(MAX_XY)+0.5)  #keep the same keypoint number

        if keypoint_type == "orb":
           returnKpLeft, returnKpRight = ORBKeyPoint(imgcrop_left,imgcrop_right,kp_num)
        elif keypoint_type=="sift":
           returnKpLeft, returnKpRight = SIFTKeyPoint(imgcrop_left,imgcrop_right,kp_num)
        elif keypoint_type=="rootsift":
           returnKpLeft, returnKpRight = SIFTKeyPoint(imgcrop_left,imgcrop_right,kp_num)

        for order in range(len(returnKpLeft)):
            (x, y) = np.array(returnKpLeft[order].pt) + [int(box_coord_left[MAX_XY[i][0]][1]), int(box_coord_left[MAX_XY[i][0]][2])]
            returnKpLeft[order].pt = (x, y)

        for order1 in range(len(returnKpRight)):
            (x1, y1) = np.array(returnKpRight[order1].pt) + (int(box_coord_right[MAX_XY[i][1]][1]), int(box_coord_right[MAX_XY[i][1]][2]))
            returnKpRight[order1].pt = (x1, y1)

        DetectTotalKpLeft.append(returnKpLeft)
        DetectTotalKpRight.append(returnKpRight)

    ##统计单帧提取的特征点个数
    DetectTotalKpRight = detectPoint(DetectTotalKpRight)
    DetectTotalKpLeft = detectPoint(DetectTotalKpLeft)

    overLapRateRight = float(wholeNum-DetectTotalKpRight)/wholeNum
    overLapRateLeft = float(wholeNum-DetectTotalKpLeft)/wholeNum

   # print("The overLap Rate After Detect Right KeyPoint:"+str(overLapRateRight))
   # print("The OverLap Rate After Detect Left KeyPoint:"+str(overLapRateLeft))

    #这里是第二次提取特征点
    ##########################################################################################################
    overLapRate = float(overLapRateLeft+overLapRateRight)/2

    print("overLapRate: ", overLapRate)
    if args.keypoint_strategy == "low":
        if keypoint_type == "orb":
           if args.dataset_type == "Mapillary_gist" or args.dataset_type == "Mapillary":
               secondNum = int(wholeNum * (1.2 + overLapRate))
           else:
               secondNum = int(wholeNum * (1.15 + overLapRate))
        elif keypoint_type == "sift":
           if args.dataset_type == "Mapillary_gist" or args.dataset_type == "Mapillary":
              secondNum = int(wholeNum * (1.15 + overLapRate))
           else:
              secondNum = int(wholeNum * (1.0 + overLapRate))
        elif keypoint_type == "rootsift":
           if args.dataset_type == "Mapillary_gist" or args.dataset_type == "Mapillary":
               secondNum = int(wholeNum * (1.15 + overLapRate))   #0.9
           else:
               overLapRateRoot = 0.9 + overLapRate
               if overLapRateRoot < 1.0:
                  overLapRateRoot = 1.0
               secondNum = int(wholeNum * overLapRateRoot)
    elif args.keypoint_strategy == "normal":
        if keypoint_type == "orb":
           secondNum = int(wholeNum * (1.0 + 2.0 * overLapRate))
        else:
           secondNum = int(wholeNum * (1.0 + 2.2 * overLapRate))

    DetectTotalKpLeft = []
    DetectTotalKpRight = []
    total_left_kp=np.array([[0,0]])
    total_right_kp=np.array([[0,0]])

    for i in range(len(MAX_XY)):
        MAX_XY[i][0] = int(MAX_XY[i][0])
        MAX_XY[i][1] = int(MAX_XY[i][1])
        imgcrop_left = img_left[int(box_coord_left[MAX_XY[i][0]][2]):int(box_coord_left[MAX_XY[i][0]][4]),
                       int(box_coord_left[MAX_XY[i][0]][1]):int(box_coord_left[MAX_XY[i][0]][3])]
        imgcrop_right = img_right[int(box_coord_right[MAX_XY[i][1]][2]):int(box_coord_right[MAX_XY[i][1]][4]),
                        int(box_coord_right[MAX_XY[i][1]][1]):int(box_coord_right[MAX_XY[i][1]][3])]

        wholeNum = secondNum
        kp_num = int(float(wholeNum) / len(MAX_XY) + 0.5)  # keep the same keypoint number
        if keypoint_type == "orb":
            kp_left, kp_right, returnKpLeft, returnKpRight = ORB(imgcrop_left, imgcrop_right, args,kp_num)
        if keypoint_type == "sift":
            kp_left, kp_right, returnKpLeft, returnKpRight = SIFT(imgcrop_left, imgcrop_right, args,kp_num)
        if keypoint_type == "rootsift":
            kp_left, kp_right, returnKpLeft, returnKpRight = RootSIFT(imgcrop_left, imgcrop_right, args, kp_num)

        for order in range(len(returnKpLeft)):
            (x, y) = np.array(returnKpLeft[order].pt) + [int(box_coord_left[MAX_XY[i][0]][1]), int(box_coord_left[MAX_XY[i][0]][2])]
            returnKpLeft[order].pt = (x, y)

        for order1 in range(len(returnKpRight)):
            (x1, y1) = np.array(returnKpRight[order1].pt) + (int(box_coord_right[MAX_XY[i][1]][1]), int(box_coord_right[MAX_XY[i][1]][2]))
            returnKpRight[order1].pt = (x1, y1)

        DetectTotalKpLeft.append(returnKpLeft)
        DetectTotalKpRight.append(returnKpRight)

        if len(kp_left) != 0 and len(kp_right) != 0:
            kp_left = np.array(kp_left)
            kp_right = np.array(kp_right)
            kp_left = kp_left + [int(box_coord_left[MAX_XY[i][0]][1]), int(box_coord_left[MAX_XY[i][0]][2])]
            kp_right = kp_right + [int(box_coord_right[MAX_XY[i][1]][1]), int(box_coord_right[MAX_XY[i][1]][2])]
            total_left_kp = np.concatenate([total_left_kp, kp_left], axis=0)
            total_right_kp = np.concatenate([total_right_kp, kp_right], axis=0)

    total_left_kp = np.delete(total_left_kp, 0, axis=0)
    total_right_kp = np.delete(total_right_kp, 0, axis=0)

    ##统计单帧提取的特征点个数
    DetectTotalKpRight = detectPoint(DetectTotalKpRight)
    DetectTotalKpLeft = detectPoint(DetectTotalKpLeft)

    print("after detect Right KeyPoint 2:" + str(DetectTotalKpRight))
    print("after detect Left KeyPoint 2:" + str(DetectTotalKpLeft))
    OriginKpNum = int((DetectTotalKpLeft + DetectTotalKpRight) / 2)

    ##########################################################################################################################
    if len(total_left_kp)==0:
        return [],[],OriginKpNum
    if len(total_left_kp) <= MIN_MATCH_COUNT:
        print ("Not enough matches are found - %d/%d" % (len(total_left_kp), 8))
        return [],[],OriginKpNum
    return total_left_kp,total_right_kp,OriginKpNum

def drawMatch(img1, img2, kp_query, kp_train, name, args):
    cols1 = img1.shape[1]
    combine_img = combineImage(img1, img2)
    if len(kp_query) == 0:
        return
    for i in range(len(kp_query)):
        (x1, y1) = kp_query[i]
        (x2, y2) = kp_train[i]
        cv2.circle(combine_img, (int(np.round(x1)), int(np.round(y1))), 1, (0, 255, 255),2)
        cv2.circle(combine_img, (int(np.round(x2)+cols1), int(np.round(y2))), 1, (0, 0, 255), 2)
        cv2.line(combine_img, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2)+cols1), int(np.round(y2))), (255, 255, 0), 1, lineType=cv2.LINE_AA, shift=0)

    if args.show_final_result:
        cv2.imshow(str(name), combine_img)
        cv2.waitKey(0)

    if args.save_final_result:
       save_dir = "./" + str(args.dataset_type) + "_result"
       if os.path.exists(save_dir) == False:
           os.makedirs(save_dir)
       cv2.imwrite(save_dir + "/" + str(name) + ".jpg", combine_img)

def Homography(kp1,kp2):
    kp1=np.array(kp1)
    kp2=np.array(kp2)
    if(len(kp1)>=4):
       H,mask=cv2.findHomography(kp1,kp2,cv2.RANSAC)
       if H is None:
          print('H matrix is None.')
          return [],[]
       else:
          kp1=kp1[mask.ravel()==1]
          kp2=kp2[mask.ravel()==1]
    return kp1,kp2

def Fundamental(kp1,kp2):
    kp1=np.array(kp1)
    kp2=np.array(kp2)
    if(len(kp1)>=4):
       F, mask = cv2.findFundamentalMat(kp1,kp2,cv2.FM_RANSAC,2,0.99)
       if F is None:
#          print('F matrix is None.')
          return [],[]
       else:
          kp1=kp1[mask.ravel()==1]
          kp2=kp2[mask.ravel()==1]
    return kp1,kp2


if __name__=='__main__':
   img1_path='./dataset/1.jpg'
   img2_path='./dataset/2.jpg'
   img1 = cv2.imread(img1_path)   # queryImage
   img2 = cv2.imread(img2_path)   # trainImage
   is_show=True
   kp1,kp2=SIFT(img1,img2,is_show)
   drawMatch(img1,img2,kp1,kp2,'result')
   
