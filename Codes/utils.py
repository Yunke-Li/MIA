from skimage.filters import threshold_multiotsu
from skimage.measure import label
import skimage
import numpy as np
from scipy.spatial import ConvexHull
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import cv2


def init_ROI(img, cwidth = 50, verbose = False):
    '''
    :param img: 2d ndarray
    :param cwidth: width of center ROI
    :return: labelled center img with largest regions
    '''
    center_pt = (np.array(img.shape) / 2).astype(np.int)
    center_tl = center_pt - int(cwidth / 2)
    center_br = center_pt + int(cwidth / 2)
    ROI = img[center_tl[0]:center_br[0], center_tl[1]:center_br[1]]
    thresholds = threshold_multiotsu(ROI, classes=2)
    regions = np.digitize(ROI, bins=thresholds)
    label_img = label(regions)
    # label_img2 = label(regions)
    _, cnt = np.unique(label_img, return_counts=True)
    cnt = np.delete(cnt, 0)
    idx = np.argmax(cnt) + 1
    label_img[label_img != idx] = 0
    label_img[label_img != 0] = 1

    if verbose:
        boundingBoxX = [center_tl[0], center_br[0], center_br[0], center_tl[0], center_tl[0]]
        boundingBoxY = [center_tl[1], center_tl[1], center_br[1], center_br[1], center_tl[1]]
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax1.imshow(img, cmap='gray')
        ax1.set_title('bounding box')
        ax1 = plt.plot(boundingBoxY, boundingBoxX)
        ax2 = fig.add_subplot(132)
        ax2.imshow(ROI, cmap='gray')
        ax2.set_title('initial ROI guess')
        ax3 = fig.add_subplot(133)
        ax3.imshow(label_img)
        ax3.set_title('binarized ROI')
        plt.show()
    return label_img, center_tl, thresholds


"""
write a region growing function from scratch
parameters we need:
bias of the seed's coordinate from the init ROI
step 1. zero copy of the original img
step 2. choose the seed from init ROI
step 3. check the 4n of the seed
step 4. mark the checked point as 2, newly added point as 1
step 5. repeat 3 & 4, until no point is marked as 1
"""


def region_growing(img, init_center, center_tl, threshold=25, seed=None, n=4):
    """
    :param img: original slice image
    :param init_center: center we have from otsu
    :param center_tl: top left bias of the init center
    :param threshold: threshold value for region growing
    :return: a ndarray for the region growing result, valid points are marked as one
    """
    scan_map = np.zeros_like(img)
    n4 = ([1, 0], [-1, 0], [0, 1], [0, -1])
    n8 = ([1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1])
    n16 = (
    [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1], [1, 2], [1, 3], [-1, 2], [-1, 3], [1, -2],
    [1, -3], [-1, -2], [-1, -3])
    if n == 4:
        neighbor = n4
    elif n == 8:
        neighbor = n8
    elif n == 16:
        neighbor = n16
    img_bound = np.array(img.shape) - [1, 1]
    scan_map.astype(np.float)
    # step 2 pick a seed
    if seed == None:
        i, j = np.where(init_center == 1)
        pos = np.row_stack((i, j))
        pos = pos.T
        seed_idx = np.random.randint(len(pos))
        seed_pos = list(pos[seed_idx] + center_tl)
    else:
        seed_pos = seed

    # step 3
    scan_map[seed_pos[0], seed_pos[1]] = 1
    scan_cnt = np.count_nonzero(scan_map == 1)
    while (scan_cnt != 0):
        i, j = np.where(scan_map == 1)
        scan_idx = np.row_stack((i, j)).T
        scan_idx.reshape([-1])
        # loop over the point marked as 1
        for i in scan_idx:
            base_idx = list(i)
            baseline = img[base_idx[0], base_idx[1]]
            low_bound = baseline - threshold
            high_bound = baseline + threshold
            for n in neighbor:
                j = list(i + n)
                if (j > img_bound).any() | (j < np.array([0, 0])).any():  # boundary check
                    continue
                elif scan_map[j[0], j[1]] == 0:
                    value = img[j[0], j[1]]
                    if (value > low_bound) & (value < high_bound):
                        scan_map[j[0], j[1]] = 1

            scan_map[base_idx[0], base_idx[1]] = 2
        scan_cnt = np.count_nonzero(scan_map == 1)

    return scan_map


def get_convex_hull_centroid(scan_map):
    """
    :param scan_map: 2d array for ROI
    :return: pixel coordinate of the convex centroid
    """
    r, c = np.where(scan_map == np.max(scan_map))
    points = np.row_stack((r, c)).T
    try:
        hull = ConvexHull(points)
    except:
        return [None, None]

    # Get centoid
    cx = np.mean(hull.points[hull.vertices, 0])
    cy = np.mean(hull.points[hull.vertices, 1])
    cx = np.floor(cx + 0.5)
    cy = np.floor(cy + 0.5)
    return [cx, cy]


def get8n(x, y, shape):
    out = []
    maxx = shape[1] - 1
    maxy = shape[0] - 1

    # top left
    outx = min(max(x - 1, 0), maxx)
    outy = min(max(y - 1, 0), maxy)
    out.append((outx, outy))

    # top center
    outx = x
    outy = min(max(y - 1, 0), maxy)
    out.append((outx, outy))

    # top right
    outx = min(max(x + 1, 0), maxx)
    outy = min(max(y - 1, 0), maxy)
    out.append((outx, outy))

    # left
    outx = min(max(x - 1, 0), maxx)
    outy = y
    out.append((outx, outy))

    # right
    outx = min(max(x + 1, 0), maxx)
    outy = y
    out.append((outx, outy))

    # bottom left
    outx = min(max(x - 1, 0), maxx)
    outy = min(max(y + 1, 0), maxy)
    out.append((outx, outy))

    # bottom center
    outx = x
    outy = min(max(y + 1, 0), maxy)
    out.append((outx, outy))

    # bottom right
    outx = min(max(x + 1, 0), maxx)
    outy = min(max(y + 1, 0), maxy)
    out.append((outx, outy))

    return out


def region_growing_2(img, seed):
    list = []
    outimg = np.zeros_like(img)
    list.append((seed[0], seed[1]))
    processed = []
    while len(list) > 0:
        pix = list[0]
        outimg[pix[0], pix[1]] = 255
        for coord in get8n(pix[0], pix[1], img.shape):
            if img[coord[0], coord[1]] != 0:
                outimg[coord[0], coord[1]] = 255
                if not coord in processed:
                    list.append(coord)
                processed.append(coord)
        list.pop(0)

    return outimg


def clockwise(x, y):
    cx = np.mean(x)
    cy = np.mean(y)
    a = np.arctan2(y - cy, x - cx)
    order = a.ravel().argsort()
    x = x[order]
    y = y[order]
    return np.vstack([x, y])


def counterclockwise(x, y):
    cx = np.mean(x)
    cy = np.mean(y)
    a = -np.arctan2(y - cy, x - cx)
    order = a.ravel().argsort()
    x = x[order]
    y = y[order]
    return np.vstack([x, y])


def getConvexPoint(hull, hullP):
    a = hullP[hull.simplices, 0]
    a = np.reshape(a, [-1])
    b = hullP[hull.simplices, 1]
    b = np.reshape(b, [-1])
    c = np.column_stack((a, b))

    c = np.unique(c, axis=0)
    a = c[:, 0]
    b = c[:, 1]
    c = clockwise(a, b)
    return c[1, :], c[0, :]


def fftSmooth(cutoff, x, y):
    signal = x + 1j * y
    # FFT and frequencies
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    fft[np.abs(freq) > cutoff] = 0
    # IFFT
    signal_filt = np.fft.ifft(fft)
    sx = signal_filt.real
    sy = signal_filt.imag
    return sx, sy


def modifiedCanny(img, sigma=6):
    edgePolar = skimage.feature.canny(img, sigma=sigma)
    # plt.imshow(edgePolar)
    # plt.show()
    gauEdgePolar = skimage.filters.gaussian(edgePolar.astype(np.float), sigma=0.5)
    smoothEdgePolar = np.digitize(gauEdgePolar, bins=np.array([0.1 * np.max(gauEdgePolar)]))
    thiningEdgePolar = skimage.morphology.skeletonize(smoothEdgePolar)
    combine = thiningEdgePolar + edgePolar
    combine = np.digitize(combine, bins=np.array([0.1]))

    return combine


def edgeCandidate(edge):
    seedY = np.where(edge[1, :] == 1)
    seedY = seedY[0]
    # shape = edge.shape
    maxYD = 0
    candidateTD = None
    candidateBU = None
    # top down
    for i in seedY:
        temp = region_growing(edge.astype(np.float), edge.astype(np.float),
                              [0, 0], seed=[1, i], threshold=1, n=8)
        y, _ = np.where(temp == 2)
        yDisparity = np.max(y) - np.min(y)
        if yDisparity > maxYD:
            maxYD = yDisparity
            candidateTD = temp

    # check bottom reach
    if maxYD >= edge.shape[1] - 3:
        candidateBU = candidateTD
        return candidateTD, candidateBU
    else:
        maxYD = 0
        seedY = np.where(edge[-2, :] == 1)
        seedY = seedY[0]

        for i in seedY:
            temp = region_growing(edge.astype(np.float), edge.astype(np.float),
                                  [0, 0], seed=[-2, i], threshold=1, n=8)
            y, _ = np.where(temp == 2)
            yDisparity = np.max(y) - np.min(y)
            if yDisparity > maxYD:
                maxYD = yDisparity
                candidateBU = temp

    if candidateTD is None:
        if candidateBU is not None:
            candidateTD = candidateBU
    elif candidateBU is None:
        candidateBU = candidateTD

    return candidateTD, candidateBU


def getEdgeCoordinate(candidateTD, candidateBU, radius, mask=0):
    # if overlap

    rTD, _ = np.where(candidateTD != 0)
    rBU, _ = np.where(candidateBU != 0)
    topBU = candidateBU[min(rBU), :]
    cBU = np.where(topBU != 0)
    cBU = cBU[0]
    if max(rTD) >= min(rBU):
        overlapMask = np.ones_like(candidateTD)
        overlapMask[min(rBU):-1, :] = 0
        overlapMask[-1, :] = 0
        if mask != 0:
            overlapMask[:, cBU[0]:-1] = 0
            overlapMask[:, -1] = 0
        candidateTD = candidateTD * overlapMask

    finalEdge = candidateTD + candidateBU
    finalEdge = cv2.linearPolar(finalEdge.astype(np.float), (radius, radius), radius,
                                cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    # if not overlap

    # plt.imshow(finalEdge)
    r, c = np.where(finalEdge != 0)
    newP = clockwise(r, c)
    hullP = newP.T
    try:
        hull = ConvexHull(hullP)
    except:
        return None, None, None
    x, y = getConvexPoint(hull, hullP)
    return x, y, convex_hull_image(finalEdge)


def expension(x, y, pixNum=2):
    cx = np.mean(x)
    cy = np.mean(y)
    tempX = x - cx
    tempY = y - cy
    theta = np.arctan2(y - cy, x - cx)
    radius = (tempX ** 2 + tempY ** 2) ** (1 / 2)
    x_new = np.cos(theta) * (radius + pixNum)
    y_new = np.sin(theta) * (radius + pixNum)
    x_new = x_new + cx
    y_new = y_new + cy

    return x_new, y_new


def findDiceError(img, gt):
    overlap = img * gt
    area = np.count_nonzero(overlap)
    imgArea = np.count_nonzero(img)
    gtArea = np.count_nonzero(gt)
    error = 2 * area / (gtArea + imgArea)
    return error


def imgNorm(img):
    newImg = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
    return newImg


def getPerpError(diff):
    dist = np.count_nonzero(diff)
    avgDist = dist / np.pi
    return avgDist