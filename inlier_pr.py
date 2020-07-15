import numpy as np
import matplotlib.pyplot as plt

def floatrange(start,stop,steps):
    return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]

def plot_pr(True_positive_inliers,False_positive_inliers,step_number,label_name,line_color,line_style):
    True_positive_inliers=sorted(True_positive_inliers,reverse=True)
    False_positive_inliers=sorted(False_positive_inliers,reverse=True)
    del False_positive_inliers[0]
    del True_positive_inliers[-1]
    True_max_number=True_positive_inliers[0]
    True_min_number=True_positive_inliers[-1]
    False_max_number=False_positive_inliers[0]
    False_min_number=False_positive_inliers[-1]
    max_number=max(True_max_number,False_max_number)
    min_number=min(True_min_number,False_min_number)

    P_list=[]
    R_list=[]
    P_list_select = []
    R_list_select = []
    threshold=[]
    threshold_set=np.linspace(min_number,max_number,step_number)
    threshold_set=sorted(threshold_set,reverse=True)
    for i in range(len(threshold_set)):
        threshold=threshold_set[i]
        true_positive=0
        flase_positive=0
        for j in range(len(True_positive_inliers)):
            if True_positive_inliers[j]>=threshold:
               true_positive=true_positive+1
        for k in range(len(False_positive_inliers)):
            if False_positive_inliers[k]>=threshold:
               flase_positive=flase_positive+1
        precision=float(true_positive)/float(true_positive+flase_positive)
        recall=float(true_positive)/float(len(True_positive_inliers))
        P_list.append(precision)
        R_list.append(recall)
        if recall>=0.8:
            P_list_select.append(precision)
            R_list_select.append(recall)

    print(label_name)
    print(P_list_select)
    print(R_list_select)
    AP=[]
    for i in range(1, step_number):
        AP.append(P_list[i] * (R_list[i] - R_list[i - 1]))
    AP = np.sum(AP)
    print('AP IS: ' + str(AP))
   # print(P_list)
   # print(R_list)
    plt.plot(R_list,P_list,label=label_name,linewidth=2,color=line_color,linestyle=line_style)