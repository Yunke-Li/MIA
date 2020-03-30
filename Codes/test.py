import os
import sys
import numpy as np
import nibabel
import nibabel as nib
from nibabel.testing import data_path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from utils import init_ROI, region_growing, get_convex_hull_centroid



dataset_path = '../dataset/training/'
patient_list = os.listdir(dataset_path)

file_path = dataset_path + patient_list[0] + '/'
file_list = os.listdir(file_path)
for i in file_list:
    if '4d' in i:
        file = file_path + i

img = nib.load(file)
img.shape[2]
img_data = img.get_fdata()
shape = img.shape
width = 30
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# prep
init_center, center_tl, threshold = init_ROI(img_data[:, :, 3, 0])
img = np.digitize(img_data[:,:,3,0], bins=threshold)
scan_map = region_growing(img, init_center, center_tl, threshold=1)

plt.imshow(img)
plt.show()
plt.imshow(scan_map)
plt.show()

cx, cy = get_convex_hull_centroid(scan_map)

print(cx, cy)



