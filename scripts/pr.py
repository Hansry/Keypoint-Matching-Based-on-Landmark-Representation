#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:12:25 2018

@author: hansry
"""

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io
import numpy as np

#################################################

params={
    'xtick.direction':'in',
    'ytick.direction':'in',
    'axes.labelsize': '10',
    'xtick.labelsize':'10',
    'ytick.labelsize':'10',
    'lines.linewidth':'1.4',
    'legend.fontsize': '11',
    'figure.figsize'   : '4.5, 4.5'    # set figure size
}
pylab.rcParams.update(params)            #set figure parameter

#################################################
def floatrange(start,stop,steps):
    return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]

'''
#UAcampus_gist
SIFT_LM_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.99744,0.970183,0.804577]
SIFT_LM_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.79212,0.85339,0.925601,1.0]
SIFT_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.973372,0.87919,0.82376,0.804577]
SIFT_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.62144,0.71991,0.8599,0.9102,1.0]

ORB_LM_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.99736,0.922,0.8045]
ORB_LM_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.76586,0.827133,0.9868,1.0]
ORB_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.976744,0.87934,0.80457]
ORB_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.663019,0.735229,0.9409,1.0]

RootSIFT_LM_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.99475,0.985507,0.89151,0.8045774]
RootSIFT_LM_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.783369,0.82932,0.89277,0.98905,1.0]
RootSIFT_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.91400,0.834914,0.8045774]
RootSIFT_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.70459,0.81400,0.96280,1.0]

j=floatrange(0.6,1,9)
#i=floatrange(0.0,1,10)
i=floatrange(0.2,1,9)
plt.yticks(j, ('0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95','1.0'))
#plt.xticks(i, ('0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'))
#plt.xticks(i, ('0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'))
plt.xlim(0.2,1.01)
plt.ylim(0.6,1.01)
'''


#mapillary_gist
SIFT_LM_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.959183,0.88333,0.688524,0.44099,0.237560]
SIFT_LM_R=[0,0.1,0.2,0.3,0.4,0.5,0.58108,0.635135,0.7162,0.851351, 0.95945,1.0]
SIFT_P=[1.0,1.0,1.0,1.0,1.0,0.97297,0.92682,0.777777,0.6136,0.380804, 0.31566,0.237560]
SIFT_R=[0,0.1,0.2,0.30,0.37162,0.48648,0.5135,0.61486,0.7297,0.83108,0.9256,1.0]

ORB_LM_P=[1.0,1.0,1.0,1.0,1.0,0.98780,  0.96842,0.7065,0.5219, 0.389355,0.237560192]
ORB_LM_R=[0,0.1,0.2,0.30,0.4459,0.547,  0.62162,0.7972, 0.88513,0.939,1.0]
ORB_P=[1.0,1.0,1.0,1.0,0.91358,0.8214,0.73426,0.32346241,0.237560192]
ORB_R=[0,0.1,0.2,0.35135,0.5,0.621621,0.70945,0.959459,1.0]

RootSIFT_LM_P=[1.0,1.0,1.0,1.0,1.0,1.0,0.9736,0.98924,0.912,0.54732,0.29072,0.23756]
RootSIFT_LM_R=[0,0.1,0.2,0.30,0.4,0.5,0.560810,0.6216,0.77027,0.898,0.95270,1.0]
RootSIFT_P=[1.0,1.0,1.0,1.0,1.0,0.92708,0.8046,0.48616,0.293248]
RootSIFT_R=[0,0.1,0.2,0.30,0.41891,0.6013,0.695,0.831081,0.93918]

#j=floatrange(0,1,10)
#i=floatrange(0.0,1,10)
#i=floatrange(0.2,1,9)
#plt.yticks(j, ('0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'))
#plt.xticks(i, ('0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'))
#plt.xticks(i, ('0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'))
plt.xlim(0.0,1.01)
plt.ylim(0.0,1.02)
'''

#mapillary
SIFT_LM_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.8666,0.7753,0.443]
SIFT_LM_R=[0,0.1,0.2,0.30,0.40,0.5,0.6,0.7,0.8036,0.894,0.9497,0.977,1.0]
SIFT_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9568,0.8716,0.6153,0.443]
SIFT_R=[0,0.1,0.2,0.30,0.40,0.5,0.6,0.689,0.8082,0.899,0.9497,1.0]

ORB_LM_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9712,0.8484,0.6656,0.443]
ORB_LM_R=[0,0.1,0.2,0.30,0.40,0.5,0.6,0.689,0.771,0.8949,0.9543,1.0]
ORB_P=[1.0,1.0,1.0,1.0,1.0,1.0,0.992,0.956,0.8669,0.8371,0.5208,0.4433]
ORB_R=[0,0.1,0.2,0.30,0.40,0.516,0.598,0.703,0.80365,0.844,0.96803,1.0]

RootSIFT_LM_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.97073,0.8631,0.7535,0.443]
RootSIFT_LM_R=[0,0.1,0.2,0.30,0.40,0.5,0.6,0.7,0.8356,0.9086,0.9497,0.977,1.0]
RootSIFT_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9726,0.8510,0.55555,0.443]
RootSIFT_R=[0,0.1,0.2,0.30,0.40,0.5,0.6,0.7488,0.8127,0.91324,0.9589,1.0]

j=floatrange(0.3,1,8)
#i=floatrange(0.0,1,10)
i=floatrange(0.2,1,9)
#plt.yticks(j, ('0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95','1'))
plt.xticks(j, ('0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'))
plt.yticks(i, ('0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'))
plt.xlim(0.3,1.01)
plt.ylim(0.2,1.02)
'''


'''
#UAcampus

SIFT_LM_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.99829,0.97962]
SIFT_LM_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.864,0.9344,1.0]
SIFT_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9943, 0.98799313,0.97962]
SIFT_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.7504,0.8384,0.9216,1.0]

ORB_LM_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.995090,0.979623]
ORB_LM_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.824,0.9728,1.0]
ORB_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.981937602,0.97962]
ORB_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.7408,0.9568,1.0]

RootSIFT_LM_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9983,0.97962]
RootSIFT_LM_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8768,0.9536,1.0]
RootSIFT_P=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9905482,0.98546042,0.97962]
RootSIFT_R=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.7616,0.8384,0.976,1.0]

j=floatrange(0.9,1,6)
#i=floatrange(0.0,1,10)
i=floatrange(0.2,1,9)
plt.yticks(j, ('0.90','0.92','0.94','0.96','0.98','1.0'))
plt.xlim(0.4,1.01)
plt.ylim(0.9,1.002)

'''

plt.plot(ORB_LM_R,ORB_LM_P,'g*-',markerfacecolor='none',markersize=12)
plt.plot(SIFT_LM_R,SIFT_LM_P,'rs-',markerfacecolor='none',markersize=10) # in 'bo-', b is blue, o is O marker, - is solid line and so on
plt.plot(RootSIFT_LM_R,RootSIFT_LM_P,'h-',color='greenyellow',markerfacecolor='none',markersize=10)
plt.plot(ORB_R,ORB_P,'gx-.',markerfacecolor='none',markersize=10)
plt.plot(SIFT_R,SIFT_P,'ro-.',markerfacecolor='none',markersize=10)
plt.plot(RootSIFT_R,RootSIFT_P,'v-.',color='greenyellow',markerfacecolor='none',markersize=10) # in 'bo-', b is blue, o is O marker, - is solid line and so on
#plt.plot(sub_dr_recall,sub_dr_precision,'rs--',markerfacecolor='none',label='MLP-DR(0.8)+RANSAC',markersize=10)

'''''
plt.plot(ORB_LM_R,ORB_LM_P,'g*-',markerfacecolor='none',label='LM-ORB-RANSAC',markersize=12)
plt.plot(SIFT_LM_R,SIFT_LM_P,'rs-',markerfacecolor='none',label='LM-SIFT-RANSAC',markersize=10) # in 'bo-', b is blue, o is O marker, - is solid line and so on
plt.plot(RootSIFT_LM_R,RootSIFT_LM_P,'h-',color='greenyellow',markerfacecolor='none',label='LM-RootSIFT-RANSAC',markersize=10)
plt.plot(ORB_R,ORB_P,'gx-.',markerfacecolor='none',label='ORB-RANSAC',markersize=10)
plt.plot(SIFT_R,SIFT_P,'ro-.',markerfacecolor='none',label='SIFT-RANSAC',markersize=10)
plt.plot(RootSIFT_R,RootSIFT_P,'v-.',color='greenyellow',markerfacecolor='none',label='RootSIFT-RANSAC',markersize=10) # in 'bo-', b is blue, o is O marker, - is solid line and so on
#plt.plot(sub_dr_recall,sub_dr_precision,'rs--',markerfacecolor='none',label='MLP-DR(0.8)+RANSAC',markersize=10)
plt.legend(loc="lower left")  #set legend location
'''''

fig1 = plt.figure(1)
axes = plt.subplot(111)
axes = plt.gca()
#axes.set_yticks([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0])
#axes.grid(True)  # add grid




plt.ylabel('Precision')   # set ystick label
plt.xlabel('Recall')  # set xstck label
plt.grid(linestyle=':',alpha=1,linewidth=0.8)
plt.savefig('/home/hansry/gist_mapillary.png',dpi = 400,bbox_inches='tight')
plt.show()