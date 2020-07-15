#accuracy rate.py
import numpy as np

def compute_corresponse(pt1,H_path):
    H=np.loadtxt(H_path,dtype=np.float32)
    pt2_list=[]
    for i in range(len(pt1)):
        (x1,y1)=pt1[i]
        x2=(H[0][0]*x1+H[0][1]*y1+H[0][2])/(H[2][0]*x1+H[2][1]*y1+H[2][2])
        y2=(H[1][0]*x1+H[1][1]*y1+H[1][2])/(H[2][0]*x1+H[2][1]*y1+H[2][2])
        pt2=(x2,y2)
        pt2_list.append(pt2)
    pt2_array=np.array(pt2_list)
    return pt2_array

def success_rate(pt1,pt2,H_path):
    #####计算图1进过H变换后在图2中的估计位置
    pt1=compute_corresponse(pt1,H_path)
    pt1=np.array(pt1)
    pt2=np.array(pt2)
    valid_match=0.0
    if len(pt2)==0:
       return 0,0
    for i in range(len(pt2)):
        dist=np.linalg.norm(pt1[i]-pt2[i])
        if dist<=3:
            valid_match=valid_match+1.0
    accuracy_rate=float(valid_match)/float(len(pt2))
    print(valid_match)
    return accuracy_rate,valid_match
