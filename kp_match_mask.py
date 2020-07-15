
def subimg_match_mask(img_left, img_right, img_left_txt, img_right_txt, MAX_XY, keypoint_type, args):
    box_coord_left = np.loadtxt(img_left_txt, skiprows=1, delimiter=',')
    box_coord_right = np.loadtxt(img_right_txt, skiprows=1, delimiter=',')

    img_left = cv2.imread(img_left)
    img_right = cv2.imread(img_right)

    # 约束sub_img的大小
    MAX_XY_temp = []
    for i in range(len(MAX_XY)):
        MAX_XY[i][0] = int(MAX_XY[i][0])
        MAX_XY[i][1] = int(MAX_XY[i][1])
        imgcrop_left = img_left[int(box_coord_left[MAX_XY[i][0]][2]):int(box_coord_left[MAX_XY[i][0]][4]), int(box_coord_left[MAX_XY[i][0]][1]):int(box_coord_left[MAX_XY[i][0]][3])]
        imgcrop_right = img_right[int(box_coord_right[MAX_XY[i][1]][2]):int(box_coord_right[MAX_XY[i][1]][4]), int(box_coord_right[MAX_XY[i][1]][1]):int(box_coord_right[MAX_XY[i][1]][3])]
        if imgcrop_left.shape[0] <= args.landmark_constraint * img_left.shape[0] and imgcrop_left.shape[1] <= args.landmark_constraint * img_left.shape[1]:  # 约束路标的大小
            MAX_XY_temp.append(MAX_XY[i])
    MAX_XY = MAX_XY_temp

