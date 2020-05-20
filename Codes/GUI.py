from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QSlider
from PyQt5.QtCore import Qt
import os
import sys
import numpy as np
import nibabel as nib
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from utils import init_ROI, region_growing, get_convex_hull_centroid, get8n, region_growing_2
import cv2
from skimage.morphology import convex_hull_image
from skimage import feature
import skimage
from segment import segLV
from utils import *


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setObjectName("MainWindow")
        self.resize(2000, 1000)
        self.LoadRawButton = QtWidgets.QPushButton(self)
        self.LoadRawButton.setObjectName("LoadButton")
        self.LoadRawButton.setText("Load Raw File")
        self.LoadRawButton.clicked.connect(self.LoadFile)

        self.LoadEDGTButton = QtWidgets.QPushButton(self)
        self.LoadEDGTButton.setObjectName("LoadEDGTButton")
        self.LoadEDGTButton.setText("Load ED GT")
        self.LoadEDGTButton.clicked.connect(self.LoadEDGT)
        self.LoadEDGTButton.move(500, 0)

        self.LoadESGTButton = QtWidgets.QPushButton(self)
        self.LoadESGTButton.setObjectName("LoadESGTButton")
        self.LoadESGTButton.setText("Load ES GT")
        self.LoadESGTButton.clicked.connect(self.LoadESGT)
        self.LoadESGTButton.move(700, 0)

        self.ComputeVButton = QtWidgets.QPushButton(self)
        self.ComputeVButton.setObjectName("ComputeV")
        self.ComputeVButton.setText("Compute Volume")
        self.ComputeVButton.clicked.connect(self.ComputeV)
        self.ComputeVButton.move(900, 0)
        # self.LoadGTButton = QtWidgets.QPushButton(self)
        # self.LoadGTButton.setObjectName("LoadGTButton")
        # self.LoadGTButton.setText("LoadGT")
        # self.LoadGTButton.clicked.connect(self.LoadFile)

        self.FrameSlider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.FrameSlider.setObjectName("FrameSlider")
        self.FrameSlider.move(200, 0)
        self.FrameSlider.setMinimum(1)
        self.FrameSlider.setMaximum(29)
        self.FrameSlider.setValue(0)
        self.FrameSlider.setTickPosition(QSlider.TicksBelow)
        self.FrameSlider.setTickInterval(6)
        self.FrameSlider.valueChanged.connect(self.FrameChanged)

        self.SliceSlider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.SliceSlider.setObjectName("SliceSlider")
        self.SliceSlider.move(200, 100)
        self.SliceSlider.setMinimum(0)
        self.SliceSlider.setMaximum(9)
        self.SliceSlider.setValue(1)
        self.SliceSlider.setTickPosition(QSlider.TicksBelow)
        self.SliceSlider.setTickInterval(2)
        self.SliceSlider.valueChanged.connect(self.SliceChanged)

        self.ShowMap = QtWidgets.QLabel(self)
        self.ShowMap.setGeometry(100, 300, 500, 300)
        self.ShowMap.setObjectName("ShowMap")
        self.ShowMap.setScaledContents(True)

        self.EDSliceSlider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.EDSliceSlider.setObjectName("EDSliceSlider")
        self.EDSliceSlider.move(180, 300)
        self.EDSliceSlider.setMinimum(0)
        self.EDSliceSlider.setMaximum(9)
        self.EDSliceSlider.setValue(1)
        self.EDSliceSlider.setTickPosition(QSlider.TicksBelow)
        self.EDSliceSlider.setTickInterval(2)
        self.EDSliceSlider.valueChanged.connect(self.EDSliceChanged)

        self.EDMap = QtWidgets.QLabel(self)
        self.EDMap.setGeometry(250, 300, 500, 300)
        self.EDMap.setObjectName("EDShowMap")
        self.EDMap.setScaledContents(True)

        self.ESSliceSlider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.ESSliceSlider.setObjectName("ESSliceSlider")
        self.ESSliceSlider.move(1000, 300)
        self.ESSliceSlider.setMinimum(0)
        self.ESSliceSlider.setMaximum(9)
        self.ESSliceSlider.setValue(1)
        self.ESSliceSlider.setTickPosition(QSlider.TicksBelow)
        self.ESSliceSlider.setTickInterval(2)
        self.ESSliceSlider.valueChanged.connect(self.ESSliceChanged)

        self.ESMap = QtWidgets.QLabel(self)
        self.ESMap.setGeometry(1200, 300, 500, 300)
        self.ESMap.setObjectName("ESShowMap")
        self.ESMap.setScaledContents(True)

        # self.ShowButton = QtWidgets.QPushButton(self)
        # self.ShowButton.setObjectName("ShowButton")
        # self.ShowButton.setText("Show")
        # self.ShowButton.move(0, 40)
        # self.ShowButton.clicked.connect(self.Show)

        # self.comboBox = QtWidgets.QComboBox(self)
        # self.comboBox.setGeometry(QtCore.QRect(200, 200, 60, 22))
        # self.comboBox.setObjectName("comboBox")
        # self.comboBox.addItem("1")
        # self.comboBox.addItem("2")
        # self.comboBox.addItem("3")
        # self.comboBox.addItem("4")
        # self.comboBox.addItem("5")
        # self.comboBox.addItem("6")
        # self.comboBox.addItem("7")
        # self.comboBox.addItem("8")
        # self.comboBox.addItem("9")
        # self.comboBox.addItem("10")
        # index = self.comboBox.findText("1", QtCore.Qt.MatchFixedString)
        # self.comboBox.setCurrentIndex(index)
        # self.comboBox.setGeometry(120, 0, 100, 30)

        self.SegButton = QtWidgets.QPushButton(self)
        self.SegButton.setObjectName("Segment")
        self.SegButton.setText("Segment")
        self.SegButton.move(0, 70)
        self.SegButton.clicked.connect(self.Segment)

        self.ErButton = QtWidgets.QPushButton(self)
        self.ErButton.setObjectName("Error")
        self.ErButton.setText("Error")
        self.ErButton.move(0, 100)

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


    def LoadFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Nifti File", "./", "All Files (*);;NIFTI (*.gz)")
        self.filename = filename
        self.rawdata = nib.load(self.filename)
        self.img_data = self.rawdata.get_fdata()
        self.info = self.rawdata.header
        filepath = os.path.split(filename)[0]
        self.filepath = filepath
        cfg_path = filepath + "/Info.cfg"
        f = open(cfg_path, 'r')
        lines = f.readlines()
        for lines in lines:
            if "ED" in lines:
                self.EDIndex = int(lines.replace('ED: ', ''))
            if "ES" in lines:
                self.ESIndex = int(lines.replace('ES: ', ''))
        print("Load Successfully!")

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
        img_pil = Image.fromarray(np.uint8(i))
        self.qPix = img_pil.toqpixmap()
        self.ShowMap.setPixmap(self.qPix)
        self.ShowMap.adjustSize()

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

    def Segment(self):
        if self.rawdata is None:
            return
        rawdata = self.rawdata
        img_data = self.img_data
        info = rawdata.header
        raw_info = info.structarr
        xspacing = raw_info['pixdim'][1]
        xRadius = int(110 / xspacing / 2)
        [r, c, sl, _] = img_data.shape
        segment_ED = np.zeros([r, c, sl])
        segment_ES = np.zeros([r, c, sl])
        for s_ED in range(sl):
            img_ED = img_data[:, :, s_ED, self.EDIndex]
            chull, _, _, _, _, _, _, _ = segLV(img_ED, xRadius)
            segment_ED[:, :, s_ED] = chull
        for s_ES in range(sl):
            img_ES = img_data[:, :, s_ES, self.ESIndex]
            chull, _, _, _, _, _, _, _ = segLV(img_ES, xRadius)
            segment_ES[:, :, s_ES] = chull

        self.SegmentResult_ED = segment_ED * 255
        self.SegmentResult_ES = segment_ES * 255
        print("Segment successfully!")

    def ComputeV(self):
        if self.SegmentResult_ED is None:
            return
        if self.SegmentResult_ES is None:
            return
        zooms = self.info.get_zooms()
        Voxel_Volume = zooms[0] * zooms[1] * zooms[3]
        SegmentResult_ED_Count = self.SegmentResult_ED / 255
        Voxel_ED = SegmentResult_ED_Count.sum()
        Volume_ED = Voxel_Volume * Voxel_ED
        SegmentResult_ES_Count = self.SegmentResult_ES / 255
        Voxel_ES = SegmentResult_ES_Count.sum()
        Volume_ES = Voxel_Volume * Voxel_ES
        print(Volume_ED)
        print(Volume_ES)
        print("Volume Computed!")


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
