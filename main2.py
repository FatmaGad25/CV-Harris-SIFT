# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1670, 1016)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(9, 9, 1641, 1081))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_11 = QtWidgets.QFrame(self.frame)
        self.frame_11.setGeometry(QtCore.QRect(-1, -1, 811, 511))
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.horizontalLayoutWidget_6 = QtWidgets.QWidget(self.frame_11)
        self.horizontalLayoutWidget_6.setGeometry(QtCore.QRect(10, 10, 801, 53))
        self.horizontalLayoutWidget_6.setObjectName("horizontalLayoutWidget_6")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_6)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_Image2_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_6)
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_Image2_2.setFont(font)
        self.label_Image2_2.setObjectName("label_Image2_2")
        self.horizontalLayout_2.addWidget(self.label_Image2_2)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.btn_1 = QtWidgets.QPushButton(self.horizontalLayoutWidget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_1.sizePolicy().hasHeightForWidth())
        self.btn_1.setSizePolicy(sizePolicy)
        self.btn_1.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_1.setFont(font)
        self.btn_1.setObjectName("btn_1")
        self.horizontalLayout_2.addWidget(self.btn_1)
        self.btn_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_2.sizePolicy().hasHeightForWidth())
        self.btn_2.setSizePolicy(sizePolicy)
        self.btn_2.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_2.setFont(font)
        self.btn_2.setObjectName("btn_2")
        self.horizontalLayout_2.addWidget(self.btn_2)
        self.text_1 = QtWidgets.QLabel(self.horizontalLayoutWidget_6)
        self.text_1.setObjectName("text_1")
        self.horizontalLayout_2.addWidget(self.text_1)
        spacerItem1 = QtWidgets.QSpacerItem(1, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.horizontalLayoutWidget_10 = QtWidgets.QWidget(self.frame_11)
        self.horizontalLayoutWidget_10.setGeometry(QtCore.QRect(10, 130, 801, 371))
        self.horizontalLayoutWidget_10.setObjectName("horizontalLayoutWidget_10")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_10)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.original_1 = QtWidgets.QLabel(self.horizontalLayoutWidget_10)
        self.original_1.setText("")
        self.original_1.setObjectName("original_1")
        self.horizontalLayout_6.addWidget(self.original_1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem2)
        self.original_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_10)
        self.original_2.setText("")
        self.original_2.setObjectName("original_2")
        self.horizontalLayout_6.addWidget(self.original_2)
        self.label_Image2_4 = QtWidgets.QLabel(self.frame_11)
        self.label_Image2_4.setGeometry(QtCore.QRect(20, 70, 131, 49))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_Image2_4.setFont(font)
        self.label_Image2_4.setObjectName("label_Image2_4")
        self.label_Image2_5 = QtWidgets.QLabel(self.frame_11)
        self.label_Image2_5.setGeometry(QtCore.QRect(440, 70, 141, 49))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_Image2_5.setFont(font)
        self.label_Image2_5.setObjectName("label_Image2_5")
        self.textEdit_1 = QtWidgets.QTextEdit(self.frame_11)
        self.textEdit_1.setGeometry(QtCore.QRect(200, 80, 131, 31))
        self.textEdit_1.setObjectName("textEdit_1")
        self.textEdit_2 = QtWidgets.QTextEdit(self.frame_11)
        self.textEdit_2.setGeometry(QtCore.QRect(650, 80, 131, 31))
        self.textEdit_2.setObjectName("textEdit_2")
        self.frame_21 = QtWidgets.QFrame(self.frame)
        self.frame_21.setGeometry(QtCore.QRect(840, -1, 801, 171))
        self.frame_21.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_21.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_21.setObjectName("frame_21")
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.frame_21)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(0, 10, 801, 53))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_Mixer = QtWidgets.QLabel(self.horizontalLayoutWidget_5)
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_Mixer.setFont(font)
        self.label_Mixer.setObjectName("label_Mixer")
        self.horizontalLayout_7.addWidget(self.label_Mixer)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem3)
        self.btn_5 = QtWidgets.QPushButton(self.horizontalLayoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_5.sizePolicy().hasHeightForWidth())
        self.btn_5.setSizePolicy(sizePolicy)
        self.btn_5.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_5.setFont(font)
        self.btn_5.setObjectName("btn_5")
        self.horizontalLayout_7.addWidget(self.btn_5)
        self.btn_6 = QtWidgets.QPushButton(self.horizontalLayoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_6.sizePolicy().hasHeightForWidth())
        self.btn_6.setSizePolicy(sizePolicy)
        self.btn_6.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_6.setFont(font)
        self.btn_6.setObjectName("btn_6")
        self.horizontalLayout_7.addWidget(self.btn_6)
        self.btn_7 = QtWidgets.QPushButton(self.horizontalLayoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_7.sizePolicy().hasHeightForWidth())
        self.btn_7.setSizePolicy(sizePolicy)
        self.btn_7.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_7.setFont(font)
        self.btn_7.setObjectName("btn_7")
        self.horizontalLayout_7.addWidget(self.btn_7)
        self.text_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_5)
        self.text_3.setObjectName("text_3")
        self.horizontalLayout_7.addWidget(self.text_3)
        spacerItem4 = QtWidgets.QSpacerItem(1, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem4)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame_21)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 100, 391, 53))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_out1 = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.label_out1.setFont(font)
        self.label_out1.setAutoFillBackground(False)
        self.label_out1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_out1.setObjectName("label_out1")
        self.verticalLayout.addWidget(self.label_out1)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.frame_21)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(420, 100, 371, 53))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_out2 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.label_out2.setFont(font)
        self.label_out2.setAutoFillBackground(False)
        self.label_out2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_out2.setObjectName("label_out2")
        self.verticalLayout_2.addWidget(self.label_out2)
        self.frame_22 = QtWidgets.QFrame(self.frame)
        self.frame_22.setGeometry(QtCore.QRect(830, 180, 811, 431))
        self.frame_22.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_22.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_22.setObjectName("frame_22")
        self.horizontalLayoutWidget_7 = QtWidgets.QWidget(self.frame_22)
        self.horizontalLayoutWidget_7.setGeometry(QtCore.QRect(0, 60, 801, 361))
        self.horizontalLayoutWidget_7.setObjectName("horizontalLayoutWidget_7")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.original_5 = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.original_5.setText("")
        self.original_5.setObjectName("original_5")
        self.horizontalLayout_10.addWidget(self.original_5)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem5)
        self.original_6 = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.original_6.setText("")
        self.original_6.setObjectName("original_6")
        self.horizontalLayout_10.addWidget(self.original_6)
        self.label_Image2_6 = QtWidgets.QLabel(self.frame_22)
        self.label_Image2_6.setGeometry(QtCore.QRect(10, 0, 201, 49))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_Image2_6.setFont(font)
        self.label_Image2_6.setObjectName("label_Image2_6")
        self.comboBox = QtWidgets.QComboBox(self.frame_22)
        self.comboBox.setGeometry(QtCore.QRect(220, 10, 81, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayoutWidget_9 = QtWidgets.QWidget(self.frame)
        self.horizontalLayoutWidget_9.setGeometry(QtCore.QRect(830, 620, 801, 371))
        self.horizontalLayoutWidget_9.setObjectName("horizontalLayoutWidget_9")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_9)
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_out1_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_9)
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.label_out1_3.setFont(font)
        self.label_out1_3.setAutoFillBackground(False)
        self.label_out1_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_out1_3.setObjectName("label_out1_3")
        self.verticalLayout_5.addWidget(self.label_out1_3)
        self.horizontalLayout_11.addLayout(self.verticalLayout_5)
        self.original_7 = QtWidgets.QLabel(self.horizontalLayoutWidget_9)
        self.original_7.setText("")
        self.original_7.setObjectName("original_7")
        self.horizontalLayout_11.addWidget(self.original_7)
        self.frame_12 = QtWidgets.QFrame(self.frame)
        self.frame_12.setGeometry(QtCore.QRect(10, 530, 811, 461))
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.horizontalLayoutWidget_8 = QtWidgets.QWidget(self.frame_12)
        self.horizontalLayoutWidget_8.setGeometry(QtCore.QRect(10, 10, 801, 53))
        self.horizontalLayoutWidget_8.setObjectName("horizontalLayoutWidget_8")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_8)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_Image2_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_8)
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_Image2_3.setFont(font)
        self.label_Image2_3.setObjectName("label_Image2_3")
        self.horizontalLayout_3.addWidget(self.label_Image2_3)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem6)
        self.btn_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_3.sizePolicy().hasHeightForWidth())
        self.btn_3.setSizePolicy(sizePolicy)
        self.btn_3.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_3.setFont(font)
        self.btn_3.setObjectName("btn_3")
        self.horizontalLayout_3.addWidget(self.btn_3)
        self.btn_4 = QtWidgets.QPushButton(self.horizontalLayoutWidget_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_4.sizePolicy().hasHeightForWidth())
        self.btn_4.setSizePolicy(sizePolicy)
        self.btn_4.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_4.setFont(font)
        self.btn_4.setObjectName("btn_4")
        self.horizontalLayout_3.addWidget(self.btn_4)
        self.text_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_8)
        self.text_2.setObjectName("text_2")
        self.horizontalLayout_3.addWidget(self.text_2)
        spacerItem7 = QtWidgets.QSpacerItem(1, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem7)
        self.horizontalLayoutWidget_11 = QtWidgets.QWidget(self.frame_12)
        self.horizontalLayoutWidget_11.setGeometry(QtCore.QRect(10, 70, 801, 371))
        self.horizontalLayoutWidget_11.setObjectName("horizontalLayoutWidget_11")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_11)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.original_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_11)
        self.original_3.setText("")
        self.original_3.setObjectName("original_3")
        self.horizontalLayout_8.addWidget(self.original_3)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem8)
        self.original_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_11)
        self.original_4.setText("")
        self.original_4.setObjectName("original_4")
        self.horizontalLayout_8.addWidget(self.original_4)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_Image2_2.setText(_translate("MainWindow", "Harris"))
        self.btn_1.setText(_translate("MainWindow", "Open Image "))
        self.btn_2.setText(_translate("MainWindow", "Harris"))
        self.text_1.setText(_translate("MainWindow", "Time Taken"))
        self.label_Image2_4.setText(_translate("MainWindow", "Threshold"))
        self.label_Image2_5.setText(_translate("MainWindow", "Sensitivity"))
        self.label_Mixer.setText(_translate("MainWindow", "Matching "))
        self.btn_5.setText(_translate("MainWindow", "Open Image1"))
        self.btn_6.setText(_translate("MainWindow", "Open Image2"))
        self.btn_7.setText(_translate("MainWindow", "Match"))
        self.text_3.setText(_translate("MainWindow", "Time Taken"))
        self.label_out1.setText(_translate("MainWindow", "Image1"))
        self.label_out2.setText(_translate("MainWindow", "Image2"))
        self.label_Image2_6.setText(_translate("MainWindow", "Matching Technique:"))
        self.comboBox.setItemText(0, _translate("MainWindow", "ssd"))
        self.comboBox.setItemText(1, _translate("MainWindow", "ncc"))
        self.label_out1_3.setText(_translate("MainWindow", "Matching Output"))
        self.label_Image2_3.setText(_translate("MainWindow", "SIFT"))
        self.btn_3.setText(_translate("MainWindow", "Open Image "))
        self.btn_4.setText(_translate("MainWindow", "SIFT"))
        self.text_2.setText(_translate("MainWindow", "Time Taken"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
