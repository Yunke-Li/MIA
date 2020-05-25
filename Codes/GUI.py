from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QSlider
from PyQt5.QtCore import Qt
import os
import sys
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# import matplotlib.pyplot as plt
# from utils import init_ROI, region_growing, get_convex_hull_centroid, get8n, region_growing_2
# import cv2
# from skimage.morphology import convex_hull_image
# from skimage import feature
# import skimage
from segment import segLV
from utils import *


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setObjectName("MainWindow")
        self.resize(1100, 800)
        self.setWindowTitle("Medical Image Analysis Project Demo by Jianning Deng & Yunke Li")
        font_1 = QtGui.QFont()
        font_1.setFamily("Arial")
        font_1.setPointSize(12)

        font_2 = QtGui.QFont()
        font_2.setFamily("Microsoft YaHei")
        font_2.setPointSize(22)

        self.ResultDisplay = QtWidgets.QLabel(self)
        self.ResultDisplay.setText("Welcome!")
        self.ResultDisplay.setFont(font_1)
        self.ResultDisplay.move(70, 700)

        self.Title = QtWidgets.QLabel(self)
        self.Title.setText("MIA DEMO")
        self.Title.setFont(font_2)
        self.Title.move(50, 50)
        self.Title.adjustSize()

        self.Name = QtWidgets.QLabel(self)
        self.Name.setText("May, 2020 MSCV1")
        self.Name.setFont(font_1)
        self.Name.move(70, 120)
        self.Name.adjustSize()

        self.logo = QtWidgets.QLabel(self)
        self.logo.setPixmap(QtGui.QPixmap('../ResourceImage/logo.png'))
        self.logo.move(720, 110)

        self.LoadRawButton = QtWidgets.QPushButton(self)
        self.LoadRawButton.setGeometry(70, 200, 200, 50)
        self.LoadRawButton.setObjectName("LoadButton")
        self.LoadRawButton.setText("Load NIFTI File")
        self.LoadRawButton.clicked.connect(self.LoadFile)


        # self.LoadEDGTButton = QtWidgets.QPushButton(self)
        # self.LoadEDGTButton.setObjectName("LoadEDGTButton")
        # self.LoadEDGTButton.setText("Load ED GT")
        # self.LoadEDGTButton.clicked.connect(self.LoadEDGT)
        # self.LoadEDGTButton.move(500, 0)
        #
        # self.LoadESGTButton = QtWidgets.QPushButton(self)
        # self.LoadESGTButton.setObjectName("LoadESGTButton")
        # self.LoadESGTButton.setText("Load ES GT")
        # self.LoadESGTButton.clicked.connect(self.LoadESGT)
        # self.LoadESGTButton.move(700, 0)

        self.ComputeVButton = QtWidgets.QPushButton(self)
        self.ComputeVButton.setObjectName("ComputeV")
        self.ComputeVButton.setText("Compute Volume")
        self.ComputeVButton.clicked.connect(self.ComputeV)
        self.ComputeVButton.setGeometry(70, 360, 200, 50)

        # self.LoadGTButton = QtWidgets.QPushButton(self)
        # self.LoadGTButton.setObjectName("LoadGTButton")
        # self.LoadGTButton.setText("LoadGT")
        # self.LoadGTButton.clicked.connect(self.LoadFile)

        self.FrameSlider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.FrameSlider.setObjectName("FrameSlider")
        # self.FrameSlider.move(350, 270)

        self.FrameLabel = QtWidgets.QLabel(self)
        self.FrameLabel.setText("Frame: ")
        self.FrameLabel.move(350, 50)
        self.FrameLabel.adjustSize()

        self.FrameSlider.setGeometry(410, 50, 170, 30)
        self.FrameSlider.setMinimum(0)
        self.FrameSlider.setMaximum(29)
        self.FrameSlider.setValue(0)
        self.FrameSlider.setTickPosition(QSlider.TicksBelow)
        self.FrameSlider.setTickInterval(6)
        self.FrameSlider.valueChanged.connect(self.FrameChanged)

        self.FrameDisplay = QtWidgets.QLabel(self)
        self.FrameDisplay.setText(str(self.FrameSlider.value() + 1))
        self.FrameDisplay.setObjectName("FrameDisplay")
        self.FrameDisplay.move(590, 50)

        self.SliceLabel = QtWidgets.QLabel(self)
        self.SliceLabel.setText("Slice: ")
        self.SliceLabel.move(350, 80)
        self.SliceLabel.adjustSize()

        self.SliceSlider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.SliceSlider.setObjectName("SliceSlider")
        # self.SliceSlider.move(350, 300)
        self.SliceSlider.setGeometry(410, 80, 170, 30)
        self.SliceSlider.setMinimum(0)
        self.SliceSlider.setMaximum(9)
        self.SliceSlider.setValue(0)
        self.SliceSlider.setTickPosition(QSlider.TicksBelow)
        self.SliceSlider.setTickInterval(2)
        self.SliceSlider.valueChanged.connect(self.SliceChanged)

        self.SliceDisplay = QtWidgets.QLabel(self)
        self.SliceDisplay.setText(str(self.SliceSlider.value() + 1))
        self.SliceDisplay.setObjectName("SliceDisplay")
        self.SliceDisplay.move(590, 80)

        self.ShowMap = QtWidgets.QLabel(self)
        self.ShowMap.move(350, 110)
        self.ShowMap.setObjectName("ShowMap")
        self.ShowMap.setScaledContents(True)
        self.ShowMap.setPixmap(QtGui.QPixmap('../ResourceImage/image.png'))

        self.EDSliceLabel = QtWidgets.QLabel(self)
        self.EDSliceLabel.setText("Slice: ")
        self.EDSliceLabel.move(350, 400)
        self.EDSliceLabel.adjustSize()

        self.EDSliceSlider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.EDSliceSlider.setObjectName("EDSliceSlider")
        # self.EDSliceSlider.move(350, 570)
        self.EDSliceSlider.setGeometry(410, 400, 170, 30)
        self.EDSliceSlider.setMinimum(0)
        self.EDSliceSlider.setMaximum(9)
        self.EDSliceSlider.setValue(0)
        self.EDSliceSlider.setTickPosition(QSlider.TicksBelow)
        self.EDSliceSlider.setTickInterval(2)
        self.EDSliceSlider.valueChanged.connect(self.EDSliceChanged)

        self.EDSliceDisplay = QtWidgets.QLabel(self)
        self.EDSliceDisplay.setText(str(self.EDSliceSlider.value() + 1))
        self.EDSliceDisplay.setObjectName("EDSlice")
        self.EDSliceDisplay.move(590, 400)

        self.EDMap = QtWidgets.QLabel(self)
        self.EDMap.move(350, 430)
        self.EDMap.setObjectName("EDShowMap")
        self.EDMap.setScaledContents(True)
        self.EDMap.setPixmap(QtGui.QPixmap('../ResourceImage/ED.png'))

        self.ESSliceLabel = QtWidgets.QLabel(self)
        self.ESSliceLabel.setText("Slice: ")
        self.ESSliceLabel.move(700, 400)
        self.ESSliceLabel.adjustSize()

        self.ESSliceSlider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.ESSliceSlider.setObjectName("ESSliceSlider")
        # self.ESSliceSlider.move(700, 570)
        self.ESSliceSlider.setGeometry(760, 400, 170, 30)
        self.ESSliceSlider.setMinimum(0)
        self.ESSliceSlider.setMaximum(9)
        self.ESSliceSlider.setValue(0)
        self.ESSliceSlider.setTickPosition(QSlider.TicksBelow)
        self.ESSliceSlider.setTickInterval(2)
        self.ESSliceSlider.valueChanged.connect(self.ESSliceChanged)

        self.ESSliceDisplay = QtWidgets.QLabel(self)
        self.ESSliceDisplay.setText(str(self.ESSliceSlider.value() + 1))
        self.ESSliceDisplay.setObjectName("ESSlice")
        self.ESSliceDisplay.move(940, 400)

        self.ESMap = QtWidgets.QLabel(self)
        self.ESMap.move(700, 430)
        self.ESMap.setObjectName("ESShowMap")
        self.ESMap.setScaledContents(True)
        self.ESMap.setPixmap(QtGui.QPixmap('../ResourceImage/ES.png'))

        self.SegButton = QtWidgets.QPushButton(self)
        self.SegButton.setObjectName("Segment")
        self.SegButton.setText("Segment")
        self.SegButton.setGeometry(70, 280, 200, 50)
        self.SegButton.clicked.connect(self.Segment)

        self.EDVLabel = QtWidgets.QLabel(self)
        self.EDVLabel.setText("ED Volume(mm^3): ")
        self.EDVLabel.move(70, 520)
        self.EDVLabel.setFont(font_1)
        self.EDVLabel.adjustSize()

        self.EDVShow = QtWidgets.QLineEdit(self)
        self.EDVShow.setText("0")
        self.EDVShow.setFont(font_1)
        self.EDVShow.setGeometry(70, 550, 200, 30)

        self.ESVLabel = QtWidgets.QLabel(self)
        self.ESVLabel.setText("ES Volume(mm^3): ")
        self.ESVLabel.move(70, 600)
        self.ESVLabel.setFont(font_1)
        self.ESVLabel.adjustSize()

        self.ESVShow = QtWidgets.QLineEdit(self)
        self.ESVShow.setText("0")
        self.ESVShow.setFont(font_1)
        self.ESVShow.setGeometry(70, 630, 200, 30)

        self.EDDetails = QtWidgets.QPushButton(self)
        self.EDDetails.setText("ED Detail")
        self.EDDetails.setGeometry(70, 440, 95, 50)
        self.EDDetails.clicked.connect(self.EDDetail)

        self.ESDetails = QtWidgets.QPushButton(self)
        self.ESDetails.setText("ES Detail")
        self.ESDetails.setGeometry(175, 440, 95, 50)
        self.ESDetails.clicked.connect(self.ESDetail)

        # self.ErButton = QtWidgets.QPushButton(self)
        # self.ErButton.setObjectName("Error")
        # self.ErButton.setText("Error")
        # self.ErButton.move(0, 100)

        # self.ErrorDisplay = QtWidgets.QLineEdit(self)
        # self.ErrorDisplay.setObjectName("Error")
        # self.ErrorDisplay.move(0,100)

        self.filename = None
        self.filepath = None
        self.ESIndex = None
        self.EDIndex = None
        self.FrameIndex = 0
        self.SliceIndex = 0
        self.EDSliceIndex = 0
        self.ESSliceIndex = 0
        self.i = None
        self.i_ED = None
        self.i_ES = None
        self.rawdata = None
        self.img_data = None
        self.info = None
        self.EDGTfilename = None
        self.EDGTrawdata = None
        self.EDGT_img_data = None
        self.ESGTfilename = None
        self.ESGTrawdata = None
        self.ESGT_img_data = None
        self.qPix = None
        self.qPix_ED = None
        self.qPix_ES = None
        self.SegmentResult_ED = None
        self.SegmentResult_ES = None
        self.maxslice = None
        self.maxframe = None
        self.xRadius = None

    def LoadFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Nifti File", "./", "All Files (*);;NIFTI (*.gz)")
        self.filename = filename
        self.rawdata = nib.load(self.filename)
        self.img_data = self.rawdata.get_fdata()
        self.info = self.rawdata.header
        self.maxslice = self.img_data.shape[2]
        self.maxframe = self.img_data.shape[3]
        self.SliceSlider.setMaximum(self.maxslice - 1)
        self.FrameSlider.setMaximum(self.maxframe - 1)
        filepath = os.path.split(filename)[0]
        self.filepath = filepath
        cfg_path = filepath + "/Info.cfg"
        f = open(cfg_path, 'r')
        lines = f.readlines()
        for lines in lines:
            if "ED" in lines:
                self.EDIndex = int(lines.replace('ED: ', '')) - 1
            if "ES" in lines:
                self.ESIndex = int(lines.replace('ES: ', '')) - 1
        self.ResultDisplay.setText("Load Successfully!" + "  ED: Frame " + str(self.EDIndex + 1) + "  ES: Frame " + str(self.ESIndex + 1))
        self.ResultDisplay.adjustSize()
        print("Load Successfully!" + "  ED: Frame " + str(self.EDIndex + 1) + "  ES: Frame " + str(self.ESIndex + 1))

    def LoadEDGT(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open EDGT File", "./", "All Files (*);;NIFTI (*.gz)")
        self.EDGTfilename = filename
        self.EDGTrawdata = nib.load(self.EDGTfilename)
        self.EDGT_img_data = self.EDGTrawdata.get_fdata()
        print("EDGT Loaded")

    def LoadESGT(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open ESGT File", "./", "All Files (*);;NIFTI (*.gz)")
        self.ESGTfilename = filename
        self.ESGTrawdata = nib.load(self.ESGTfilename)
        self.ESGT_img_data = self.ESGTrawdata.get_fdata()
        print("ESGT Loaded")

    def FrameChanged(self):
        self.FrameIndex = self.FrameSlider.value()
        if self.rawdata is None:
            return
        i = self.img_data[:, :, self.SliceIndex, self.FrameIndex]
        self.i = i
        img_pil = Image.fromarray(np.uint8(self.i))
        # size = [img_pil.size[0], img_pil.size[1]]
        # size = [x * 2 for x in size]
        # img_pil = img_pil.resize((size))
        self.qPix = img_pil.toqpixmap()
        self.ShowMap.setPixmap(self.qPix)
        self.ShowMap.adjustSize()
        self.FrameDisplay.setText(str(self.FrameSlider.value() + 1))
        self.FrameDisplay.adjustSize()

    def SliceChanged(self):
        self.SliceIndex = self.SliceSlider.value()
        if self.rawdata is None:
            return
        i = self.img_data[:, :, self.SliceIndex, self.FrameIndex]
        self.i = i
        img_pil = Image.fromarray(np.uint8(i))
        self.qPix = img_pil.toqpixmap()
        self.ShowMap.setPixmap(self.qPix)
        self.ShowMap.adjustSize()
        self.SliceDisplay.setText(str(self.SliceSlider.value() + 1))
        self.SliceDisplay.adjustSize()

    def EDSliceChanged(self):
        self.EDSliceIndex = self.EDSliceSlider.value()
        if self.SegmentResult_ED is None:
            return
        i = self.SegmentResult_ED[:, :, self.EDSliceIndex]
        self.i_ED = i
        img_pil = Image.fromarray(np.uint8(i))
        self.qPix_ED = img_pil.toqpixmap()
        self.EDMap.setPixmap(self.qPix_ED)
        self.EDMap.adjustSize()
        self.EDSliceDisplay.setText(str(self.EDSliceIndex + 1))
        self.EDSliceDisplay.adjustSize()

    def ESSliceChanged(self):
        self.ESSliceIndex = self.ESSliceSlider.value()
        if self.SegmentResult_ES is None:
            return
        i = self.SegmentResult_ES[:, :, self.ESSliceIndex]
        self.i_ES = i
        img_pil = Image.fromarray(np.uint8(i))
        self.qPix_ES = img_pil.toqpixmap()
        self.ESMap.setPixmap(self.qPix_ES)
        self.ESMap.adjustSize()
        self.ESSliceDisplay.setText(str(self.ESSliceIndex + 1))
        self.ESSliceDisplay.adjustSize()

    def Segment(self):
        if self.rawdata is None:
            return
        rawdata = self.rawdata
        img_data = self.img_data
        info = rawdata.header
        raw_info = info.structarr
        xspacing = raw_info['pixdim'][1]
        xRadius = int(110 / xspacing / 2)
        self.xRadius = xRadius
        [r, c, sl, _] = img_data.shape
        segment_ED = np.zeros([r, c, sl])
        segment_ES = np.zeros([r, c, sl])
        self.EDSliceSlider.setMaximum(sl - 1)
        self.ESSliceSlider.setMaximum(sl - 1)
        ED_outlier = ""
        ES_outlier = ""
        ED_outlier_display = ""
        ES_outlier_display = ""
        for s_ED in range(sl):
            img_ED = img_data[:, :, s_ED, self.EDIndex]
            chull, _, _, x, y, roi, cx, cy = segLV(img_ED, xRadius)
            segment_ED[:, :, s_ED] = chull
            if chull is None:
                segment_ED[:, :, s_ED] = np.zeros([r, c])
                ED_outlier = ED_outlier + str(s_ED + 1) + " "
        for s_ES in range(sl):
            img_ES = img_data[:, :, s_ES, self.ESIndex]
            chull, _, _, x, y, roi, cx, cy = segLV(img_ES, xRadius)
            segment_ES[:, :, s_ES] = chull
            if chull is None:
                segment_ES[:, :, s_ES] = np.zeros([r, c])
                ES_outlier = ES_outlier + str(s_ES + 1) + " "
        self.SegmentResult_ED = segment_ED * 255
        self.SegmentResult_ES = segment_ES * 255
        print("Segment Done!")
        if ED_outlier != "":
            ED_outlier_display = "ED: Slice No." + ED_outlier + "Failed"
        if ES_outlier != "":
            ES_outlier_display = "ES: Slice No." + ES_outlier + "Failed"
        self.ResultDisplay.setText("Segment Done!  " + ED_outlier_display + "  " + ES_outlier_display)
        self.ResultDisplay.adjustSize()

    def ComputeV(self):
        if self.SegmentResult_ED is None:
            return
        if self.SegmentResult_ES is None:
            return
        zooms = self.info.get_zooms()
        [_, _, s_ED] = self.SegmentResult_ED.shape
        [_, _, s_ES] = self.SegmentResult_ES.shape
        count_outlier_ED = 0
        count_outlier_ES = 0
        for s in range(s_ED):
            slice_ED = self.SegmentResult_ED[:, :, s] / 255
            if slice_ED.sum() < 10:
                count_outlier_ED = count_outlier_ED + 1

        for s in range(s_ES):
            slice_ES = self.SegmentResult_ES[:, :, s] / 255
            if slice_ES.sum() < 10:
                count_outlier_ES = count_outlier_ES + 1

        Voxel_Volume = zooms[0] * zooms[1] * zooms[3]
        SegmentResult_ED_Count = self.SegmentResult_ED / 255
        Voxel_ED = SegmentResult_ED_Count.sum()
        Volume_ED = Voxel_Volume * Voxel_ED / (s_ED - count_outlier_ED) * s_ED
        SegmentResult_ES_Count = self.SegmentResult_ES / 255
        Voxel_ES = SegmentResult_ES_Count.sum()
        Volume_ES = Voxel_Volume * Voxel_ES / (s_ES - count_outlier_ES) * s_ES
        self.EDVShow.setText(str(round(Volume_ED, 2)))
        self.ESVShow.setText(str(round(Volume_ES, 2)))
        # print(Volume_ED)
        # print(Volume_ES)
        print("Volume Computed!")

    def EDDetail(self):
        if self.SegmentResult_ED is None:
            return
        index = self.EDSliceIndex
        img_ED_index = self.img_data[:, :, index, self.EDIndex]
        chull, _, _, x, y, roi, cx, cy = segLV(img_ED_index, self.xRadius)
        if chull is None:
            return
        xr = self.xRadius
        bound_x = [cx - xr, cx + xr, cx + xr, cx - xr, cx - xr]
        bound_y = [cy - xr, cy - xr, cy + xr, cy + xr, cy - xr]
        plt.subplot(1, 2, 1)
        plt.imshow(img_ED_index, cmap="gray")
        plt.plot(bound_y, bound_x, 'r-', linewidth=4)
        plt.subplot(1, 2, 2)
        plt.imshow(roi, cmap="gray")
        plt.plot(x, y, 'r-', linewidth=4)
        plt.show()

    def ESDetail(self):
        if self.SegmentResult_ES is None:
            return
        index = self.ESSliceIndex
        img_ES_index = self.img_data[:, :, index, self.ESIndex]
        chull, _, _, x, y, roi, cx, cy = segLV(img_ES_index, self.xRadius)
        if chull is None:
            return
        xr = self.xRadius
        bound_x = [cx - xr, cx + xr, cx + xr, cx - xr, cx - xr]
        bound_y = [cy - xr, cy - xr, cy + xr, cy + xr, cy - xr]
        plt.subplot(1, 2, 1)
        plt.imshow(img_ES_index, cmap="gray")
        plt.plot(bound_y, bound_x, 'r-', linewidth=4)
        plt.subplot(1, 2, 2)
        plt.imshow(roi, cmap="gray")
        plt.plot(x, y, 'r-', linewidth=4)
        plt.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
