import os
import sys
import numpy as np
import nibabel
import nibabel as nib
from skimage.filters import threshold_multiotsu  # updated
from nibabel.testing import data_path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from utils import *
import cv2
from skimage.morphology import convex_hull_image

from scipy.spatial import ConvexHull
from skimage import feature
import skimage



dataset_path = '/home/dj/git/MIA/dataset/training/'
# dataset_path = r'D:\ds\training\\'


patient_list = os.listdir(dataset_path)

file_path = dataset_path + patient_list[0] + '/'
file_list = os.listdir(file_path)
for i in file_list:
    if 'frame.' in i:
        file = file_path + i

img = nib.load(file)
img.shape[2]
info = img.header
img_data = img.get_fdata()
shape = img.shape
raw = info.structarr
xspacing = raw['pixdim'][1]
yspacing = raw['pixdim'][2]
xRadius = int(110/xspacing/2)
yRadius = int(110/yspacing/2)
width = xRadius
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
rawImg = img_data[:, :, 0]
# prep
init_center, center_tl, threshold = init_ROI(rawImg)
img = np.digitize(rawImg, bins=threshold)
scan_map = region_growing(img, init_center, center_tl, threshold=1)

plt.imshow(rawImg)
plt.show()
plt.imshow(scan_map)
plt.show()

cx, cy = get_convex_hull_centroid(scan_map)
cx = int(cx)
cy = int(cy)

print(str(cx), ' and ', str(cy))

# Li's code
i = rawImg

# edited by DJ
# img normalize
i = (i - np.min(i))/(np.max(i) - np.min(i)) * 255

roi = i[cx - xRadius:cx + xRadius+1, cy - yRadius:cy + yRadius+1]

#
thresholds = threshold_multiotsu(roi)   # updated
# T1 = thresholds[0]   # updated
TBlood = np.array([thresholds[1]])
diff = thresholds[1]-thresholds[0]

bloodRegion = np.digitize(roi, bins=TBlood)

# modified by Deng
plt.imshow(bloodRegion)

# modified canny edge
roiPolar = cv2.linearPolar(roi, (xRadius, yRadius), xRadius, cv2.WARP_FILL_OUTLIERS )
plt.imshow(roiPolar)
plt.show()
rawEdge = modifiedCanny(roiPolar, sigma=8)
TD, BU = edgeCandidate(rawEdge)
edgeX, edgeY = getEdgeCoordinate(TD, BU)
edgeX = np.concatenate((edgeX, edgeX[0:1]))
edgeY = np.concatenate((edgeY, edgeY[0:1]))
plt.imshow(roi,cmap='gray')
plt.plot(edgeX, edgeY, 'k-')
plt.show()


# region growing base on the thresholding result 
regionBloodPloar = cv2.linearPolar(bloodRegion, (xRadius, yRadius), xRadius, cv2.WARP_FILL_OUTLIERS )
yPos = 0
while regionBloodPloar[0,yPos]!=1:
    yPos = np.random.randint(yRadius)
seed = [0, yPos]
polarGrowing = region_growing(regionBloodPloar,regionBloodPloar,[0,0],seed=seed, threshold=1)
plt.imshow(polarGrowing)
growingCart = cv2.linearPolar(polarGrowing.astype(np.float), (xRadius, yRadius), xRadius, cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS)


chull = convex_hull_image(growingCart)
plt.imshow(chull)
x,y = np.where(chull==1)
newP = clockwise(x,y)
hullP = newP.T
hull = ConvexHull(hullP)

x, y = getConvexPoint(hull, hullP)
plt.plot(x, y, 'k-')
for simplex in hull.simplices:
    plt.plot(hullP[simplex, 1], hullP[simplex, 0], 'k-')
plt.imshow(roi, cmap='gray')
plt.show()


# FFT smoothing

sx,sy = fftSmooth(0.05, x,y)
sx = np.concatenate((sx,sx[0:1]))
sy = np.concatenate((sy,sy[0:1]))

plt.plot(sx,sy,'-')
plt.plot
plt.imshow(roi,cmap='gray')