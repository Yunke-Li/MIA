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



# dataset_path = '../dataset/training/'
# # dataset_path = r'D:\ds\training\\'


# patient_list = os.listdir(dataset_path)

# file_path = dataset_path + patient_list[0] + '/'
# file_list = os.listdir(file_path)
# for i in file_list:
#     if '4d' in i:
#         file = file_path + i

# img = nib.load(file)
# img.shape[2]
# info = img.header
# img_data = img.get_fdata()
# shape = img.shape
# raw = info.structarr
# xspacing = raw['pixdim'][1]
# yspacing = raw['pixdim'][2]
# xRadius = int(110/xspacing/2)
# yRadius = int(110/yspacing/2)
# width = xRadius
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# prep

# init_center, center_tl, threshold = init_ROI(img_data[:, :, 3, 0], cwidth=width)
def segLV(imgRaw, width):
    init_center, center_tl, threshold = init_ROI(imgRaw)
    img = np.digitize(imgRaw, bins=threshold)
    scan_map = region_growing(img, init_center, center_tl, threshold=1)
    #
    # plt.imshow(imgRaw)
    # plt.show()
    # plt.imshow(scan_map)
    # plt.show()

    cx, cy = get_convex_hull_centroid(scan_map)
    cx = int(cx)
    cy = int(cy)

    # Li's code
    i = imgRaw

    # edited by DJ
    # img normalize
    i = (i - np.min(i))/(np.max(i) - np.min(i)) * 255

    roi = i[cx - width:cx + width+1, cy - width:cy + width+1]

    #
    thresholds = threshold_multiotsu(roi)   # updated
    # T1 = thresholds[0]   # updated
    TBlood = np.array([thresholds[1]])
    threDiff = thresholds[1]-thresholds[0]

    bloodRegion = np.digitize(roi, bins=TBlood-0.3*threDiff)

    # modified by Deng

    # modified canny edge
    roiPolar = cv2.linearPolar(roi, (width, width), width, cv2.WARP_FILL_OUTLIERS )
    # plt.imshow(roiPolar)
    # plt.show()
    rawEdge = modifiedCanny(roiPolar, sigma=6)
    # plt.imshow(rawEdge)
    # plt.show()
    TD, BU = edgeCandidate(rawEdge)
    edgeX, edgeY, cEdge = getEdgeCoordinate(TD, BU, width)
    edgeX = np.concatenate((edgeX, edgeX[0:1]))
    edgeY = np.concatenate((edgeY, edgeY[0:1]))
    edgeX, edgeY = expension(edgeX, edgeY, pixNum=3)
    plt.imshow(roi, cmap='gray')
    plt.plot(edgeX, edgeY, 'o-')
    plt.show()


    # region growing base on the thresholding result 
    regionBloodPloar = cv2.linearPolar(bloodRegion, (width, width), width, cv2.WARP_FILL_OUTLIERS )



    yPos = 0
    while regionBloodPloar[0,yPos]!=1:
        yPos = np.random.randint(width)
    seed = [0, yPos]
    polarGrowing = region_growing(regionBloodPloar,regionBloodPloar,[0,0],seed=seed, threshold=1)
    # plt.imshow(polarGrowing)
    growingCart = cv2.linearPolar(polarGrowing.astype(np.float), (width, width), width, cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS)


    chull = convex_hull_image(growingCart)
    # plt.imshow(chull)
    x,y = np.where(chull==1)
    newP = clockwise(x,y)
    hullP = newP.T
    hull = ConvexHull(hullP)


    x, y = getConvexPoint(hull, hullP)

    # combine result with canny edge
    # canny > growing whithin threshold, pick growing, otherwise, canny
    # canny < growing whithin threshold, pick growing
    # canny < growing with diff larger than threshold
    #   check mean(intensity)
    #   mean(intensity) > TBlook + 0.5*diff, pick canny, otherwise growing

    diff = cEdge.astype(np.float) - chull.astype(np.float)
    nonOverlap = abs(diff)
    nonOverlapArea = np.count_nonzero(nonOverlap)
    cannyArea = np.count_nonzero(cEdge)
    if np.sum(diff) > 0: # canny > growing
        # expand canny area and do one more threshold to obtain the
        edgeX, edgeY = expension(edgeX, edgeY)
        rr, cc = skimage.draw.polygon(edgeX, edgeY, cEdge.shape)
        cannyMask = np.zeros_like(cEdge)
        cannyMask[rr,cc] = 1
        plt.imshow(cannyMask)
        plt.show()
        pass
    elif np.sum(diff) <= 0: # growing > canny
        if nonOverlapArea > 0.3*cannyArea:
            x = edgeX
            y = edgeY

            newRegion = np.digitize(roi, bins=TBlood-0.1*threDiff)
            newRegion = newRegion * cEdge
            # plt.imshow(newRegion)
            # plt.show()
            x, y = np.where(newRegion == 1)
            newP = clockwise(x, y)
            hullP = newP.T
            hull = ConvexHull(hullP)

            x, y = getConvexPoint(hull, hullP)
            pass
        #     region growing
        else:
            pass


    # FFT smoothing


    sx,sy = fftSmooth(0.2, x,y)
    sx = np.concatenate((sx,sx[0:1]))
    sy = np.concatenate((sy,sy[0:1]))

    plt.plot(sx,sy,'-')
    plt.plot
    plt.imshow(roi,cmap='gray')
    plt.show()
    OP = np.zeros_like(imgRaw)
    OP[cx - width:cx + width + 1, cy - width:cy + width + 1] = chull

    # recover the whole image and change the coordinate to global coordinate
    return OP, x+cx, y+cy, sx+cx, sy+cy