import os
import sys
import numpy as np
import nibabel
import nibabel as nib
from skimage.filters import threshold_multiotsu  # updated
from nibabel.testing import data_path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
plt.interactive(False)
from utils import *
import cv2
from skimage.morphology import convex_hull_image
from segment import segLV
# from scipy.spatial import ConvexHull
# from skimage import feature
import skimage



dataset_path = '/home/dj/git/MIA/dataset/training/'
# dataset_path = r'D:\ds\training\\'
dir_list = os.listdir(dataset_path)

file_list = os.listdir(dataset_path)
for i in file_list:
    if 'frame01.' in i:
        fileRaw = dataset_path + i
    elif 'frame01_' in i:
        fileGT = dataset_path + i

raw_data = nib.load(fileRaw)
gt_data = nib.load(fileGT)
gt = gt_data.get_fdata()
# img.shape[2]
info = raw_data.header
img = raw_data.get_fdata()
shape = img.shape
raw_info = info.structarr
xspacing = raw_info['pixdim'][1]
yspacing = raw_info['pixdim'][2]
xRadius = int(110/xspacing/2)
yRadius = int(110/yspacing/2)
width = xRadius
sliceNum = shape[2]
diceError = np.zeros(sliceNum)
diceDiff = np.zeros(sliceNum)


for s in range(sliceNum):
    tempGt = gt[:,:,s]
    tempRaw = img[:,:,s]

    # gt preprocessing
    tempGt[tempGt!=3] = 0
    
    tempGt[tempGt==3] = 1
    chull, x, y, sx, sy, roi, cx, cy = segLV(tempRaw, xRadius)
    diff = (chull - tempGt)
    plt.subplot(1, 2, 1)

    plt.imshow(roi,cmap='gray')
    plt.plot(sx-cx,sy-cy,'-o')

    plt.subplot(1, 2, 2)
    plt.imshow(diff)
    l = np.sum(diff)
    diceError[s]= findDiceError(chull, tempGt)
    d,_ = np.where(diff !=0)
    diceDiff[s] = len(d)
    
    plt.show()


    continue

# plt.bar(range(len(diceDiff)), diceDiff)
# plt.show()
# print('total diff error for segmentation is: ', str(total))
plt.bar(range(len(diceError)), diceError)
plt.show()
avgDice = np.sum(diceError)/sliceNum
print('average diceError is: ', str(avgDice))


