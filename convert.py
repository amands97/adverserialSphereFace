import cv2
import numpy as np
import os

idx = 0
for names in os.listdir('lfw_masked/'):
    print(names, idx)
    idx += 1
    if names.endswith(".zip"):
        continue
    for files in os.listdir('lfw_masked/' + names):
        if not files.startswith("mask"):
            continue
        img = cv2.imread("lfw_masked/" + names + "/" + files)
        img = np.mean(img, axis=2)
        mask = img < 80
        # print(mask)
        # img_orig_name = files.strip("mask_")
        # img2 = cv2.imread("lfw_masked/" + names + "/" + img_orig_name)
        # img_save = mask * 
        # mask = np.zeros((500, 500))
        # mask = [[np.array_equal(img[i,j], np.array([[83,  0, 68]])) for j in range(500)] for i in range(500)]
        np.save('lfw_masked/' + names + "/np_" + files,  mask)
        # break
    # break