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
from segment import *
# from scipy.spatial import ConvexHull
# from skimage import feature
import skimage
from evalUtils import *



# dataset_path = '/home/dj/git/MIA/dataset/training/'
dataset_path = '/media/dj/6078A03871D31A90/MIA/dataset/training/'
dir_list = os.listdir(dataset_path)
studyNum = len(dir_list)
edDice = np.zeros(studyNum)
esDice = np.zeros(studyNum)
edDiceDetail = [0] * studyNum
esDiceDetail = [0] * studyNum

# # ===================================================================
# # single study run
# # ===================================================================
file_path = dataset_path + 'patient046'
f_list = os.listdir(file_path)
# read config
esSlice, edSlice = getEDES(f_list, file_path)
for i in f_list:
    print(i)
    if 'frame'+edSlice in i:
        if 'gt' in i:
            edFileGTName = file_path + '/' + i
        else:
            edFileName = file_path + '/' + i
    elif 'frame'+esSlice in i:
        if 'gt' in i:
            esFileGTName = file_path + '/' + i
        else:
            esFileName = file_path + '/' + i
edRaw = nib.load(edFileName)
edGT = nib.load(edFileGTName)
esRaw = nib.load(esFileName)
esGT = nib.load(esFileGTName)
                
segLV3DEval(edRaw, edGT, verbose=True)
segLV3DEval(esRaw, esGT, verbose=True)




# # loop over all the training data
# totalStudy = len(dir_list)
# preName = 'patient'


# for d in range(len(dir_list)):
# # d = dir_list[0]
#     file_path = dataset_path + preName + str(d+1).zfill(3)
#     f_list = os.listdir(file_path)
#     # read config
#     esSlice, edSlice = getEDES(f_list, file_path)
#     for i in f_list:
#         print(i)
#         if 'frame'+edSlice in i:
#             if 'gt' in i:
#                 edFileGTName = file_path + '/' + i
#             else:
#                 edFileName = file_path + '/' + i
#         elif 'frame'+esSlice in i:
#             if 'gt' in i:
#                 esFileGTName = file_path + '/' + i
#             else:
#                 esFileName = file_path + '/' + i
#     edRaw = nib.load(edFileName)
#     edGT = nib.load(edFileGTName)
#     esRaw = nib.load(esFileName)
#     esGT = nib.load(esFileGTName)
                    
#     edDice[d], edDiceDetail[d] = segLV3DEval(edRaw, edGT, verbose=False)
#     esDice[d], esDiceDetail[d] = segLV3DEval(esRaw, esGT, verbose=False)

# np.save('edDiceErrorDetail.npy',edDiceDetail)
# np.save('esDiceErrorDetail.npy',esDiceDetail)
