import os      
import cv2
from shutil import copyfile
   
def order(file_dir,type):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:
            if type=='.jpg':
               file=file.strip('.jpg')
               file=int(file)
            if type=='.txt':
               file=file.strip('.txt')
            L.append(file)
            L.sort()
    return L  

def folder(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in dirs:
            L.append(file)
            L.sort()
    return L

def dataset_to_bing():  #jpg
    bing_result_path='/home/hansry/append/code/bing/datasets/VOC2007/Results/BBoxesB2W8MAXBGR/'
    dataset_read='./dataset/UAcampus/UAcampus_right/'
    dataset_write='./dataset/UAcampus/UAcampus_right_copy/'
    L_bing=order(bing_result_path,'.txt')
    L_dataset=order(dataset_read,'.jpg')
    for i in range(647):
        img= cv2.imread(dataset_read+str(i)+'.jpg')
        cv2.imwrite(dataset_write+L_bing[i]+'.jpg',img)

def bing_to_dataset(): #txt
    bing_txt_path='./dataset/UAcampus/UAcampus_right_txt'
    bing_txt='./dataset/UAcampus/UAcampus_right_txt'
    #L_dataset=order(dataset_read,'.jpg')
    #print(L_dataset)
    L_bing_txt=order(bing_txt_path,'.txt')
    print(L_bing_txt)
    for i in range(1300):
        os.rename(bing_txt_path+'/'+str(L_bing_txt[i])+'.txt',bing_txt+'/'+str(i)+'.txt')

if __name__=='__main__':
   #dataset_to_bing()
   
   bing_to_dataset()
   '''  
   #L=folder('/home/hansry/append/code/landmark_keypoint_matching/dataset/hpatches/viewpoint')
   #bing_result_path='/home/hansry/append/code/bing/datasets/VOC2007/Results/BBoxesB2W8MAXBGR/'
   #L_bing=order(bing_result_path,'.txt')
   #for i in range(len(L)):
       #for j in range(6):
       # img=cv2.imread('./dataset/hpatches/viewpoint/'+L[i]+'/'+str(j+1)+'.ppm')
        #cv2.imwrite('./dataset/hpatches/viewpoint_copy/'+L_bing[i*6+j]+'.jpg',img)
   bing_txt_path='./dataset/hpatches/viewpoint_txt'
   L_bing_txt=order(bing_txt_path,'.txt')
   print(L_bing_txt)
   L=folder('./dataset/hpatches/viewpoint')
   print(L)
   count=1
   j=0
   for i in range(len(L_bing_txt)):
       os.rename(bing_txt_path+'/'+str(L_bing_txt[i])+'.txt','./dataset/hpatches/viewpoint/'+L[j]+'/'+str(count)+'.txt')
       count=count+1
       if count==7:
          count=1
          j=j+1
   '''        
