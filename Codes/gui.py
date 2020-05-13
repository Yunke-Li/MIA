from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(200, 200, 1200, 800)
    win.setWindowTitle("MIA Project")

    label = QtWidgets.QLabel(win)
    label.move(600, 0)
    label.setText("test")

    win.show()
    sys.exit(app.exec_())


window()
