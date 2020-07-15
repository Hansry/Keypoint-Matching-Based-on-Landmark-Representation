import cv2
import os
def order(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            L.append(file)
            L.sort()
    return L

if __name__=='__main__':      
   #L=order('/home/hansry/append/code/landmark_keypoint_matching/dataset/mapillary/AB_temp/BY_txt')      
   #print(L)
   #count=172           
   #path='/home/hansry/append/code/landmark_keypoint_matching/dataset/mapillary/AB_temp/BY_txt/'
   #for i in range(len(L)):
       #os.rename(path+str(L[i]),path+str(count)+'.txt')
       #count=count+1
   for i in range(648):
       img=cv2.imread("/home/hansry/append/code/landmark_keypoint_matching/dataset/UAcampus/UAcampus/"+str(i+647)+'.jpg')
       cv2.imwrite("/home/hansry/append/code/landmark_keypoint_matching/dataset/UAcampus/UAcampus_right/"+str(i)+'.jpg',img)
       
       
