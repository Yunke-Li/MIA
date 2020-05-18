from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
import os
import sys
import numpy as np
import nibabel as nib
from skimage.filters import threshold_multiotsu  # updated
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from utils import init_ROI, region_growing, get_convex_hull_centroid, get8n, region_growing_2
import cv2
from skimage.morphology import convex_hull_image
from skimage import feature
import skimage


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setObjectName("MainWindow")
        self.resize(444, 415)
        self.LoadButton = QtWidgets.QPushButton(self)
        self.LoadButton.setObjectName("LoadButton")
        self.LoadButton.setText("Load")
        self.LoadButton.clicked.connect(self.LoadFile)

        self.ShowButton = QtWidgets.QPushButton(self)
        self.ShowButton.setObjectName("ShowButton")
        self.ShowButton.setText("Show")
        self.ShowButton.move(0, 40)
        self.ShowButton.clicked.connect(self.Show)

        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.setGeometry(QtCore.QRect(200, 200, 60, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("1")
        self.comboBox.addItem("2")
        self.comboBox.addItem("3")
        self.comboBox.addItem("4")
        self.comboBox.addItem("5")
        self.comboBox.addItem("6")
        self.comboBox.addItem("7")
        self.comboBox.addItem("8")
        self.comboBox.addItem("9")
        self.comboBox.addItem("10")
        index = self.comboBox.findText("1", QtCore.Qt.MatchFixedString)
        self.comboBox.setCurrentIndex(index)
        self.comboBox.setGeometry(120, 0, 100, 30)

        self.SegButton = QtWidgets.QPushButton(self)
        self.SegButton.setObjectName("Segement")
        self.SegButton.setText("Segement")
        self.SegButton.move(0, 70)
        self.SegButton.clicked.connect(self.Segement)

        self.filename = []
        self.sliceindex = 1
        self.i = []
    def LoadFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Nifti File", "./", "All Files (*);;NIFTI (*.gz)")
        self.filename = filename

    def Show(self):
        filename = self.filename
        img = nib.load(filename)
        img_data = img.get_fdata()
        self.sliceindex = self.comboBox.currentIndex()
        i = img_data[:, :, self.sliceindex, 0]
        self.i = i
        plt.figure()
        plt.imshow(i, cmap='gray')
        plt.show()

    def Segement(self):
        init_center, center_tl, threshold = init_ROI(self.i)
        img = np.digitize(self.i, bins=threshold)
        scan_map = region_growing(img, init_center, center_tl, threshold=1)

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(scan_map)
        # plt.show()

        cx, cy = get_convex_hull_centroid(scan_map)
        cx = int(cx)
        cy = int(cy)
        """
        ============================================================================
        Li's code, organized by Deng
        """
        # ================================
        # required modification:
        # put the self-defined function in utils.py
        # ================================

        i = self.i
        roi = i[cx - 30:cx + 30, cy - 30:cy + 30]
        # plt.figure()
        # plt.imshow(i, cmap='gray')
        # plt.show()

        # ==========================================================

        thresholds = threshold_multiotsu(roi)  # updated
        T1 = thresholds[0]  # updated
        T2 = thresholds[0]  # updated
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
        # cv2.imshow("polar1", polar_image)

        # ==========================================================
        # same for otsu
        ret, img = cv2.threshold(i, T1, T2, cv2.THRESH_BINARY)  # 阈值要改
        # ==========================================================

        # ==========================================================
        # changed all center_x and center_y to cx cy
        seed = [cx, cy]  # center要改
        #
        region = region_growing_2(img, seed)
        # plt.figure()
        # plt.imshow(region, cmap='gray')
        # plt.show()

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


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
