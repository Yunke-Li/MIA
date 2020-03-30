from skimage.filters import threshold_multiotsu
from skimage.measure import label
import numpy as np
from scipy.spatial import ConvexHull

def init_ROI(img, cwidth=50):
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
    _, cnt = np.unique(label_img, return_counts=True)
    cnt = np.delete(cnt, 0)
    idx = np.argmax(cnt) + 1
    label_img[label_img != idx] = 0
    label_img[label_img != 0] = 1
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


def region_growing(img, init_center, center_tl, threshold=25):
    """

    :param img: original slice image
    :param init_center: center we have from otsu
    :param center_tl: top left bias of the init center
    :param threshold: threshold value for region growing
    :return: a ndarray for the region growing result, valid points are marked as one
    """
    scan_map = np.zeros_like(img)
    # neighbor = ([-1, -1],
    #               [-1, 0],
    #               [-1, 1],
    #               [0, -1],
    #               [0, 1],
    #               [1, -1],
    #               [1, 0],
    #               [1, 1])
    neighbor = ([1, 0], [-1, 0], [0, 1], [0, -1])
    img_bound = np.array(img.shape) - [1, 1]

    # step 2
    i, j = np.where(init_center == 1)
    pos = np.row_stack((i, j))
    pos = pos.T
    seed_idx = np.random.randint(len(pos))
    seed_pos = list(pos[seed_idx] + center_tl)

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
    hull = ConvexHull(points)

    # Get centoid
    cx = np.mean(hull.points[hull.vertices, 0])
    cy = np.mean(hull.points[hull.vertices, 1])
    cx = np.floor(cx + 0.5)
    cy = np.floor(cy + 0.5)
    return [cx, cy]












