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
    TMuscel = np.array([thresholds[0]])
    threDiff = thresholds[1]-thresholds[0]

    bloodRegion = np.digitize(roi, bins=TBlood-0.4*threDiff)

    # modified by Deng

    # modified canny edge
    roiPolar = cv2.linearPolar(roi, (width, width), width, cv2.WARP_FILL_OUTLIERS )
    # plt.imshow(roiPolar)
    # plt.show()
    rawEdge = modifiedCanny(roiPolar, sigma=8)
    preMask = np.ones_like(rawEdge)
    preMask[:,0:2] = 0
    rawEdge = rawEdge * preMask
    plt.imshow(rawEdge)
    plt.show()
    TD, BU = edgeCandidate(rawEdge)
    edgeX, edgeY, cEdge = getEdgeCoordinate(TD, BU, width)
    edgeX = np.concatenate((edgeX, edgeX[0:1]))
    edgeY = np.concatenate((edgeY, edgeY[0:1]))
    edgeX, edgeY = expension(edgeX, edgeY, pixNum=2)

    # plt.imshow(roi, cmap='gray')
    # plt.plot(edgeX, edgeY, 'o-')
    # plt.show()


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

    edgeX, edgeY = expension(edgeX, edgeY, pixNum=2)
    rr, cc = skimage.draw.polygon(edgeX, edgeY, cEdge.shape)
    cannyMask = np.zeros_like(cEdge)
    cannyMask[cc,rr] = 1

    # plt.imshow(cannyMask)
    # plt.show()
    diff = cannyMask.astype(np.float) - chull.astype(np.float)
    nonOverlap = abs(diff)
    nonOverlapArea = np.count_nonzero(nonOverlap)
    cannyArea = np.count_nonzero(cEdge)

    if nonOverlapArea > 0.5*cannyArea:
        # check average pixel intensity for nonOverlap
        tempRoi = roi * nonOverlap
        avgI = np.sum(tempRoi)/nonOverlapArea
        if avgI > TMuscel:
            newRegion = cannyMask*chull
            x, y = np.where(newRegion == 1)
            newP = clockwise(x, y)
            hullP = newP.T
            hull = ConvexHull(hullP)
            x, y = getConvexPoint(hull, hullP)
            rr, cc = skimage.draw.polygon(x, y, cEdge.shape)
            chull = np.zeros_like(cEdge)
            chull[cc,rr] = 1
        else:
            # expand the canny area
            edgeX, edgeY = expension(edgeX, edgeY, pixNum=5)
            rr, cc = skimage.draw.polygon(edgeX, edgeY, cEdge.shape)
            cannyMask = np.zeros_like(cEdge)
            cannyMask[cc,rr] = 1
            # apply mask on roi
            tempRoi = roi * cannyMask
            tempRoi[tempRoi==0] = 255
            minI = np.min(tempRoi)
            tempRoi = tempRoi * cannyMask
            tempRoi[tempRoi==0]=minI

            # biniarize the masked area
            tempRoi = imgNorm(tempRoi)
            tempThres = threshold_multiotsu(tempRoi, classes=2)
            
            newRegion = np.digitize(tempRoi, bins=0.8*tempThres)
            newRegion = region_growing(newRegion, newRegion, [0,0], threshold=1, seed=[width, width], n=8)
            # convechull
            chull = convex_hull_image(newRegion)
            # plt.imshow(chull)
            x,y = np.where(chull==1)
            newP = clockwise(x,y)
            hullP = newP.T
            hull = ConvexHull(hullP)
            x, y = getConvexPoint(hull, hullP)
    else:
        if np.sum(diff) < 0: # canny < growing
            expandNum = 4
            thresCo = 0.5
        else: 
            expandNum = 3
            thresCo = 0.7
        # expand the canny area
        edgeX, edgeY = expension(edgeX, edgeY, pixNum=expandNum)
        rr, cc = skimage.draw.polygon(edgeX, edgeY, cEdge.shape)
        cannyMask = np.zeros_like(cEdge)
        cannyMask[cc,rr] = 1
        # apply mask on roi
        tempRoi = roi * cannyMask
        tempRoi[tempRoi==0] = 255
        minI = np.min(tempRoi)
        tempRoi = tempRoi * cannyMask
        tempRoi[tempRoi==0]=minI

        # biniarize the masked area

        tempRoi = imgNorm(tempRoi)
        tempThres = threshold_multiotsu(tempRoi, classes=2)
        # tempThres[0] = 2*minI
        newRegion = np.digitize(tempRoi, bins=thresCo*tempThres)
        newRegion = region_growing(newRegion, newRegion, [0,0], threshold=1, seed=[width, width], n=8)

        # convechull
        chull = convex_hull_image(newRegion)
        # plt.imshow(chull)
        x,y = np.where(chull==1)
        newP = clockwise(x,y)
        hullP = newP.T
        hull = ConvexHull(hullP)
        x, y = getConvexPoint(hull, hullP)

    


    # FFT smoothing


    sy,sx = fftSmooth(0.5, y,x)
    sx = np.concatenate((sx,sx[0:1]))
    sy = np.concatenate((sy,sy[0:1]))


    rr, cc = skimage.draw.polygon(sx, sy, cEdge.shape)
    cannyMask = np.zeros_like(cEdge)
    cannyMask[cc,rr] = 1
    OP = np.zeros_like(imgRaw)
    OP[cx - width:cx + width + 1, cy - width:cy + width + 1] = cannyMask

    # recover the whole image and change the coordinate to global coordinate
    return OP, x+cx, y+cy, sx+cx, sy+cy, roi, cx, cy