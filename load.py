# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'load.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_load(object):
    def setupUi(self, load):
        load.setObjectName("load")
        load.resize(1660, 1065)
        self.gridLayout_7 = QtWidgets.QGridLayout(load)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.PixmapLabel = PixmapLabel(load)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(48)
        font.setBold(True)
        font.setWeight(75)
        self.PixmapLabel.setFont(font)
        self.PixmapLabel.setObjectName("PixmapLabel")
        self.gridLayout_7.addWidget(self.PixmapLabel, 0, 0, 1, 2)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_6.setSpacing(20)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.ElevatedCardWidget = ElevatedCardWidget(load)
        self.ElevatedCardWidget.setObjectName("ElevatedCardWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.ElevatedCardWidget)
        self.gridLayout.setObjectName("gridLayout")
        self.TitleLabel = TitleLabel(self.ElevatedCardWidget)
        self.TitleLabel.setObjectName("TitleLabel")
        self.gridLayout.addWidget(self.TitleLabel, 0, 1, 1, 1)
        self.ImageLabel = ImageLabel(self.ElevatedCardWidget)
        self.ImageLabel.setObjectName("ImageLabel")
        self.gridLayout.addWidget(self.ImageLabel, 0, 0, 2, 1)
        self.BodyLabel = BodyLabel(self.ElevatedCardWidget)
        self.BodyLabel.setObjectName("BodyLabel")
        self.gridLayout.addWidget(self.BodyLabel, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.ElevatedCardWidget)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 2, 2, 1)
        self.gridLayout_6.addWidget(self.ElevatedCardWidget, 0, 0, 1, 1)
        self.ElevatedCardWidget_4 = ElevatedCardWidget(load)
        self.ElevatedCardWidget_4.setObjectName("ElevatedCardWidget_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.ElevatedCardWidget_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.TitleLabel_4 = TitleLabel(self.ElevatedCardWidget_4)
        self.TitleLabel_4.setObjectName("TitleLabel_4")
        self.gridLayout_4.addWidget(self.TitleLabel_4, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.ElevatedCardWidget_4)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 2, 2, 1)
        self.BodyLabel_4 = BodyLabel(self.ElevatedCardWidget_4)
        self.BodyLabel_4.setObjectName("BodyLabel_4")
        self.gridLayout_4.addWidget(self.BodyLabel_4, 1, 1, 1, 1)
        self.ImageLabel_4 = ImageLabel(self.ElevatedCardWidget_4)
        self.ImageLabel_4.setObjectName("ImageLabel_4")
        self.gridLayout_4.addWidget(self.ImageLabel_4, 0, 0, 2, 1)
        self.gridLayout_6.addWidget(self.ElevatedCardWidget_4, 1, 1, 1, 1)
        self.ElevatedCardWidget_3 = ElevatedCardWidget(load)
        self.ElevatedCardWidget_3.setObjectName("ElevatedCardWidget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.ElevatedCardWidget_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.ImageLabel_3 = ImageLabel(self.ElevatedCardWidget_3)
        self.ImageLabel_3.setObjectName("ImageLabel_3")
        self.gridLayout_3.addWidget(self.ImageLabel_3, 0, 0, 2, 1)
        self.TitleLabel_3 = TitleLabel(self.ElevatedCardWidget_3)
        self.TitleLabel_3.setObjectName("TitleLabel_3")
        self.gridLayout_3.addWidget(self.TitleLabel_3, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.ElevatedCardWidget_3)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 2, 2, 1)
        self.BodyLabel_3 = BodyLabel(self.ElevatedCardWidget_3)
        self.BodyLabel_3.setObjectName("BodyLabel_3")
        self.gridLayout_3.addWidget(self.BodyLabel_3, 1, 1, 1, 1)
        self.gridLayout_6.addWidget(self.ElevatedCardWidget_3, 0, 1, 1, 1)
        self.ElevatedCardWidget_2 = ElevatedCardWidget(load)
        self.ElevatedCardWidget_2.setObjectName("ElevatedCardWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.ElevatedCardWidget_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.ImageLabel_2 = ImageLabel(self.ElevatedCardWidget_2)
        self.ImageLabel_2.setObjectName("ImageLabel_2")
        self.gridLayout_2.addWidget(self.ImageLabel_2, 0, 0, 2, 1)
        self.TitleLabel_2 = TitleLabel(self.ElevatedCardWidget_2)
        self.TitleLabel_2.setObjectName("TitleLabel_2")
        self.gridLayout_2.addWidget(self.TitleLabel_2, 0, 1, 1, 1)
        self.BodyLabel_2 = BodyLabel(self.ElevatedCardWidget_2)
        self.BodyLabel_2.setObjectName("BodyLabel_2")
        self.gridLayout_2.addWidget(self.BodyLabel_2, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.ElevatedCardWidget_2)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 2, 2, 1)
        self.gridLayout_6.addWidget(self.ElevatedCardWidget_2, 1, 0, 1, 1)
        self.ElevatedCardWidget_5 = ElevatedCardWidget(load)
        self.ElevatedCardWidget_5.setObjectName("ElevatedCardWidget_5")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.ElevatedCardWidget_5)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.ImageLabel_5 = ImageLabel(self.ElevatedCardWidget_5)
        self.ImageLabel_5.setObjectName("ImageLabel_5")
        self.gridLayout_5.addWidget(self.ImageLabel_5, 0, 0, 2, 1)
        self.TitleLabel_5 = TitleLabel(self.ElevatedCardWidget_5)
        self.TitleLabel_5.setObjectName("TitleLabel_5")
        self.gridLayout_5.addWidget(self.TitleLabel_5, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.ElevatedCardWidget_5)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_5.addWidget(self.label_5, 0, 2, 2, 1)
        self.BodyLabel_5 = BodyLabel(self.ElevatedCardWidget_5)
        self.BodyLabel_5.setObjectName("BodyLabel_5")
        self.gridLayout_5.addWidget(self.BodyLabel_5, 1, 1, 1, 1)
        self.gridLayout_6.addWidget(self.ElevatedCardWidget_5, 2, 0, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_6, 1, 0, 2, 1)
        self.StrongBodyLabel = StrongBodyLabel(load)
        self.StrongBodyLabel.setText("")
        self.StrongBodyLabel.setObjectName("StrongBodyLabel")
        self.gridLayout_7.addWidget(self.StrongBodyLabel, 1, 1, 1, 1)
        self.PrimaryPushButton = PrimaryPushButton(load)
        self.PrimaryPushButton.setMinimumSize(QtCore.QSize(200, 200))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.PrimaryPushButton.setFont(font)
        self.PrimaryPushButton.setObjectName("PrimaryPushButton")
        self.gridLayout_7.addWidget(self.PrimaryPushButton, 2, 1, 1, 1)
        self.gridLayout_7.setColumnStretch(0, 3)
        self.gridLayout_7.setColumnStretch(1, 2)
        self.gridLayout_7.setRowStretch(0, 2)
        self.gridLayout_7.setRowStretch(1, 2)
        self.gridLayout_7.setRowStretch(2, 1)

        self.retranslateUi(load)
        QtCore.QMetaObject.connectSlotsByName(load)

    def retranslateUi(self, load):
        _translate = QtCore.QCoreApplication.translate
        load.setWindowTitle(_translate("load", "Form"))
        self.PixmapLabel.setText(_translate("load", "鱼群识别-启动界面"))
        self.TitleLabel.setText(_translate("load", "根目录"))
        self.BodyLabel.setText(_translate("load", "-"))
        self.label.setText(_translate("load", ">"))
        self.TitleLabel_4.setText(_translate("load", "文档文件夹"))
        self.label_4.setText(_translate("load", ">"))
        self.BodyLabel_4.setText(_translate("load", "docx"))
        self.TitleLabel_3.setText(_translate("load", "模型文件夹"))
        self.label_3.setText(_translate("load", ">"))
        self.BodyLabel_3.setText(_translate("load", "best.pt"))
        self.TitleLabel_2.setText(_translate("load", "输出文件夹"))
        self.BodyLabel_2.setText(_translate("load", "output"))
        self.label_2.setText(_translate("load", ">"))
        self.TitleLabel_5.setText(_translate("load", "数据文件夹"))
        self.label_5.setText(_translate("load", ">"))
        self.BodyLabel_5.setText(_translate("load", "datas"))
        self.PrimaryPushButton.setText(_translate("load", "开始识别"))
from qfluentwidgets import BodyLabel, ElevatedCardWidget, ImageLabel, PixmapLabel, PrimaryPushButton, StrongBodyLabel, TitleLabel
