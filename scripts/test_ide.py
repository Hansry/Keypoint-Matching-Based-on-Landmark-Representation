#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:51:47 2018

@author: hansry
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import numpy as np
import scipy.io
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import random
## 从.txt文件中读取数据

######################################
params={
    'axes.labelsize': '20',
    'xtick.labelsize':'10',
    'ytick.labelsize':'10',
    'xtick.direction':'in',
    'ytick.direction':'in',
    'lines.linewidth':'4' ,
#    'legend.fontsize': '2',
    'figure.figsize'   : '5, 4'
}
pylab.rcParams.update(params)
######################################

'''
A=[]
B=[]
for i in range(0,300):
    A.append(random.randint(100,250))
for j in range(0,300):
    B.append(random.randint(1,150))
'''
def floatrange(start,stop,steps):
    return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]

A=[182, 349, 387, 421, 37, 295, 442, 455, 38, 353, 369, 366, 141, 257, 85, 31, 84, 28, 62, 363, 82, 796, 244, 339, 110, 148, 368, 154, 224, 471, 212, 156, 352, 103, 40, 114, 138, 130, 96, 219, 219, 174, 293, 133, 212, 62, 67, 56, 19, 16, 48, 82, 92, 129, 19, 15, 126, 76, 26, 98, 281, 515, 1228, 604, 163, 644, 138, 169, 284, 135, 89, 4, 28, 0, 146, 101, 44, 7, 228, 57, 43, 36, 233, 105, 60, 113, 135, 156, 170, 173, 17, 36, 210, 86, 272, 170, 384, 116, 284, 19, 145, 25, 355, 1232, 619, 462, 530, 107, 2599, 421, 357, 376, 30, 189, 33, 717, 401, 358, 566, 807, 564, 341, 70, 365, 521, 491, 461, 98, 30, 147, 15, 825, 15, 46, 29, 161, 10, 10, 165, 30, 629, 19, 0, 114, 172, 152, 258, 9, 1252, 1240, 111, 729, 584, 652, 604, 289, 1011, 1131, 20, 221, 474, 942, 311, 307, 127, 680, 532, 37, 135, 194, 67, 401]


B=[15, 9, 4, 4, 6, 8, 0, 7, 6, 0, 5, 6, 0, 6, 5, 20, 40, 8, 8, 9, 20, 9, 10, 0, 7, 0, 0, 47, 6, 0, 13, 5, 10, 8, 9, 6, 8, 7, 10, 5, 10, 32, 14, 5, 5, 5, 5, 9, 13, 12, 0, 8, 6, 9, 10, 44, 44, 6, 8, 4, 0, 6, 0, 9, 7, 0, 8, 0, 6, 0, 0, 0, 6, 11, 9, 12, 19, 33, 16, 5, 16, 8, 10, 6, 8, 8, 10, 0, 8, 8, 5, 18, 0, 5, 8, 375, 310, 5, 5, 7, 0, 6, 5, 7, 5, 0, 0, 17, 6, 5, 0, 7, 0, 5, 0, 5, 4, 5, 0, 6, 0, 0, 6, 0, 0, 0, 0, 4, 5, 7, 4, 10, 0, 0, 18, 10, 7, 0, 11, 12, 6, 6, 5, 17, 18, 0, 0, 6, 0, 5, 6, 6, 0, 8, 4, 0, 5, 8, 6, 5, 9, 0, 8, 5, 0, 9, 12, 29, 6, 50, 9, 0, 131, 200, 723, 6, 297, 133, 34, 10, 10, 7, 4, 7, 5, 14, 9, 10, 0, 5, 4, 0, 7, 6, 5, 5, 12, 0, 5, 0, 20, 7, 223, 652, 158, 380, 373, 731, 263, 0, 0, 0, 0, 0, 0, 12, 5, 5, 7, 4, 7, 0, 5, 10, 122, 11, 0, 0, 8, 15, 11, 15, 6, 7, 9, 16, 5, 6, 6, 15, 6, 7, 8, 7, 5, 0, 5, 39, 0, 6, 0, 0, 5, 6, 8, 0, 6, 189, 12, 506, 11, 0, 563, 4, 136, 12, 34, 7, 6, 117, 96, 4, 80, 70, 20, 63, 0, 0, 0, 6, 9, 0, 14, 0, 23, 11, 5, 13, 7, 8, 10, 12, 6, 23, 19, 9, 6, 12, 12, 33, 25, 36, 8, 22, 5]


A.sort()
for i in range(10):
    del A[-1]

#define the max number
A_max=max(A)
B_max=max(B)
max_number=max(A_max,B_max)

interval=150 #0~1
frequency_A=[]
frequency_B=[]
x_axis=[]
for i in range(interval):
    frequency_A.append(0)
    frequency_B.append(0)
for j in floatrange(0,1,interval):
    x_axis.append(j)
    
for k in range(len(A)):
    Normalized_inlier_count=float(A[k])/float(max_number)  
    for l in range(interval-1):
        if Normalized_inlier_count<=x_axis[l+1] and x_axis[l]<Normalized_inlier_count:
            frequency_A[l]=frequency_A[l]+1
for m in range(len(B)):
    Normalized_inlier_count=float(B[m])/float(max_number)  
    for n in range(interval-1):
        if Normalized_inlier_count<=x_axis[n+1] and x_axis[n]<Normalized_inlier_count:
            frequency_B[n]=frequency_B[n]+1

width=float(1)/(interval)
plt.ylim(0,80)
plt.xlim(0,1)
j=floatrange(0,1,11)
plt.xticks(j,('0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'))
plt.bar(x_axis,frequency_A,width,color = 'black',label='Loop-closure',align='center')  
plt.bar(x_axis,frequency_B,width,color ='r',label='Non Loop-closure',align='center') 
plt.xlabel('Normalized inlier counter')
plt.ylabel('Frequency')
plt.legend(loc="upper right")
plt.show()
        
