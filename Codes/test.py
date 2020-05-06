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

"""
============================================================================
Li's code, organized by Deng
"""
# ================================
# required modification:
# put the self-defined function in utils.py
# ================================


roi = i[cx - 30:cx + 30, cy - 30:cy + 30]
plt.figure()
plt.imshow(i, cmap='gray')
plt.show()



# ==========================================================
T1, T2 = ExOtsu(roi)
# change to function in skimage
# from skimage.filters import threshold_multiotsu
# check official document for detail
# https://scikit-image.org/
# ==========================================================


# polar transformation
source = roi
img64_float = source.astype(np.float64)

Mvalue = 30  # radius np.sqrt(((img64_float.shape[0] / 2.0) ** 2.0) + ((img64_float.shape[1] / 2.0) ** 2.0))

polar_image = cv2.linearPolar(img64_float, (img64_float.shape[0] / 2, img64_float.shape[1] / 2), Mvalue,
                              cv2.WARP_FILL_OUTLIERS)  # 中心位置待定


polar_image = polar_image / 255
cv2.imshow("polar1", polar_image)


# ==========================================================
# same for otsu
ret, img = cv2.threshold(i, T1, T2, cv2.THRESH_BINARY)  # 阈值要改
# ==========================================================



# ==========================================================
# changed all center_x and center_y to cx cy
seed = [cx, cy]  # center要改
#
region = region_growing(img, seed)
plt.figure()
plt.imshow(region, cmap='gray')
plt.show()

edges = feature.canny(polar_image, sigma=11)
edges64_float = edges.astype(np.float64)
cartesian_edges = cv2.linearPolar(edges64_float, (img64_float.shape[0] / 2, img64_float.shape[1] / 2), Mvalue,
                                  cv2.WARP_INVERSE_MAP)  # 中心位置待定
kernel1 = skimage.morphology.disk(5)
kernel2 = skimage.morphology.disk(3)
img_dilation = skimage.morphology.dilation(cartesian_edges, kernel1)
img_erosion = skimage.morphology.erosion(img_dilation, kernel2)
finaledge = skimage.morphology.skeletonize(img_erosion)
edge_padd = np.zeros(i.shape)
edge_padd[cx - 30:cx + 30, cy - 30:cy + 30] = finaledge
contour = ((edge_padd + region) >= 1)

# convex hull
convex_hull = convex_hull_image(contour)
plt.figure()
plt.imshow(convex_hull, cmap='gray')
plt.show()
# low-pass filter of FFT (smoothing)





