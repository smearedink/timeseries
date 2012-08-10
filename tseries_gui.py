# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tseries_gui.ui'
#
# Created: Fri Aug 10 13:50:49 2012
#      by: pyside-uic 0.2.13 running on PySide 1.1.1
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_TimeSeries(object):
    def setupUi(self, TimeSeries):
        TimeSeries.setObjectName("TimeSeries")
        TimeSeries.resize(646, 524)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(TimeSeries.sizePolicy().hasHeightForWidth())
        TimeSeries.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(TimeSeries)
        self.centralwidget.setObjectName("centralwidget")
        self.PSRlist = QtGui.QListWidget(self.centralwidget)
        self.PSRlist.setGeometry(QtCore.QRect(30, 130, 251, 121))
        self.PSRlist.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.PSRlist.setProperty("isWrapping", False)
        self.PSRlist.setWordWrap(False)
        self.PSRlist.setObjectName("PSRlist")
        self.RemovePSR = QtGui.QPushButton(self.centralwidget)
        self.RemovePSR.setEnabled(False)
        self.RemovePSR.setGeometry(QtCore.QRect(30, 260, 121, 32))
        self.RemovePSR.setObjectName("RemovePSR")
        self.ClearPSRs = QtGui.QPushButton(self.centralwidget)
        self.ClearPSRs.setEnabled(False)
        self.ClearPSRs.setGeometry(QtCore.QRect(160, 260, 121, 32))
        self.ClearPSRs.setObjectName("ClearPSRs")
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 90, 131, 32))
        self.label.setObjectName("label")
        self.AddPSRs = QtGui.QPushButton(self.centralwidget)
        self.AddPSRs.setGeometry(QtCore.QRect(220, 90, 61, 32))
        self.AddPSRs.setObjectName("AddPSRs")
        self.NumPSRs = QtGui.QSpinBox(self.centralwidget)
        self.NumPSRs.setGeometry(QtCore.QRect(160, 90, 57, 32))
        self.NumPSRs.setProperty("value", 1)
        self.NumPSRs.setObjectName("NumPSRs")
        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(310, 40, 311, 471))
        self.frame.setFrameShape(QtGui.QFrame.Box)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayoutWidget = QtGui.QWidget(self.frame)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 291, 381))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.ParValues = QtGui.QGridLayout(self.gridLayoutWidget)
        self.ParValues.setContentsMargins(0, 0, 0, 0)
        self.ParValues.setObjectName("ParValues")
        self.inputPSR = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputPSR.setCursorPosition(0)
        self.inputPSR.setObjectName("inputPSR")
        self.ParValues.addWidget(self.inputPSR, 0, 1, 1, 1)
        self.inputPB = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputPB.setObjectName("inputPB")
        self.ParValues.addWidget(self.inputPB, 6, 1, 1, 1)
        self.label_5 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.ParValues.addWidget(self.label_5, 4, 0, 1, 1)
        self.inputT0 = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputT0.setObjectName("inputT0")
        self.ParValues.addWidget(self.inputT0, 5, 1, 1, 1)
        self.inputPEPOCH = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputPEPOCH.setObjectName("inputPEPOCH")
        self.ParValues.addWidget(self.inputPEPOCH, 4, 1, 1, 1)
        self.inputOM = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputOM.setObjectName("inputOM")
        self.ParValues.addWidget(self.inputOM, 7, 1, 1, 1)
        self.inputINC = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputINC.setObjectName("inputINC")
        self.ParValues.addWidget(self.inputINC, 9, 1, 1, 1)
        self.inputM1 = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputM1.setObjectName("inputM1")
        self.ParValues.addWidget(self.inputM1, 10, 1, 1, 1)
        self.inputPOSEPOCH = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputPOSEPOCH.setObjectName("inputPOSEPOCH")
        self.ParValues.addWidget(self.inputPOSEPOCH, 3, 1, 1, 1)
        self.label_3 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.ParValues.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_10 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_10.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.ParValues.addWidget(self.label_10, 9, 0, 1, 1)
        self.inputM2 = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputM2.setObjectName("inputM2")
        self.ParValues.addWidget(self.inputM2, 11, 1, 1, 1)
        self.label_2 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.ParValues.addWidget(self.label_2, 0, 0, 1, 1)
        self.inputP1 = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputP1.setObjectName("inputP1")
        self.ParValues.addWidget(self.inputP1, 2, 1, 1, 1)
        self.label_11 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_11.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_11.setObjectName("label_11")
        self.ParValues.addWidget(self.label_11, 10, 0, 1, 1)
        self.label_6 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.ParValues.addWidget(self.label_6, 5, 0, 1, 1)
        self.inputP0 = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputP0.setObjectName("inputP0")
        self.ParValues.addWidget(self.inputP0, 1, 1, 1, 1)
        self.label_12 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_12.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_12.setObjectName("label_12")
        self.ParValues.addWidget(self.label_12, 11, 0, 1, 1)
        self.label_7 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.ParValues.addWidget(self.label_7, 6, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.ParValues.addWidget(self.label_4, 3, 0, 1, 1)
        self.inputE = QtGui.QLineEdit(self.gridLayoutWidget)
        self.inputE.setObjectName("inputE")
        self.ParValues.addWidget(self.inputE, 8, 1, 1, 1)
        self.label_8 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.ParValues.addWidget(self.label_8, 7, 0, 1, 1)
        self.label_9 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_9.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.ParValues.addWidget(self.label_9, 1, 0, 1, 1)
        self.label_13 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_13.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_13.setObjectName("label_13")
        self.ParValues.addWidget(self.label_13, 8, 0, 1, 1)
        self.SaveParChanges = QtGui.QPushButton(self.frame)
        self.SaveParChanges.setEnabled(False)
        self.SaveParChanges.setGeometry(QtCore.QRect(90, 430, 151, 32))
        self.SaveParChanges.setObjectName("SaveParChanges")
        self.inputAMP = QtGui.QLineEdit(self.frame)
        self.inputAMP.setGeometry(QtCore.QRect(210, 400, 91, 22))
        self.inputAMP.setMaxLength(20)
        self.inputAMP.setObjectName("inputAMP")
        self.label_21 = QtGui.QLabel(self.frame)
        self.label_21.setGeometry(QtCore.QRect(20, 400, 191, 22))
        self.label_21.setObjectName("label_21")
        self.frame_2 = QtGui.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(30, 10, 251, 71))
        self.frame_2.setFrameShape(QtGui.QFrame.Box)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label_15 = QtGui.QLabel(self.frame_2)
        self.label_15.setGeometry(QtCore.QRect(80, 30, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setWeight(75)
        font.setItalic(True)
        font.setBold(True)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.label_14 = QtGui.QLabel(self.frame_2)
        self.label_14.setGeometry(QtCore.QRect(40, 0, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setWeight(75)
        font.setItalic(True)
        font.setBold(True)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.AddToTS = QtGui.QPushButton(self.centralwidget)
        self.AddToTS.setEnabled(False)
        self.AddToTS.setGeometry(QtCore.QRect(30, 300, 251, 32))
        self.AddToTS.setObjectName("AddToTS")
        self.frame_3 = QtGui.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(30, 340, 251, 171))
        self.frame_3.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.label_19 = QtGui.QLabel(self.frame_3)
        self.label_19.setGeometry(QtCore.QRect(10, 70, 81, 22))
        self.label_19.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_19.setObjectName("label_19")
        self.inputStartTime = QtGui.QLineEdit(self.frame_3)
        self.inputStartTime.setGeometry(QtCore.QRect(99, 100, 141, 22))
        self.inputStartTime.setText("")
        self.inputStartTime.setObjectName("inputStartTime")
        self.label_17 = QtGui.QLabel(self.frame_3)
        self.label_17.setGeometry(QtCore.QRect(10, 40, 81, 22))
        self.label_17.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_17.setObjectName("label_17")
        self.inputLength = QtGui.QLineEdit(self.frame_3)
        self.inputLength.setGeometry(QtCore.QRect(100, 40, 141, 22))
        self.inputLength.setObjectName("inputLength")
        self.inputTres = QtGui.QLineEdit(self.frame_3)
        self.inputTres.setGeometry(QtCore.QRect(99, 70, 141, 22))
        self.inputTres.setObjectName("inputTres")
        self.inputNoise = QtGui.QLineEdit(self.frame_3)
        self.inputNoise.setGeometry(QtCore.QRect(100, 10, 141, 22))
        self.inputNoise.setObjectName("inputNoise")
        self.MakeTS = QtGui.QPushButton(self.frame_3)
        self.MakeTS.setEnabled(False)
        self.MakeTS.setGeometry(QtCore.QRect(10, 130, 231, 32))
        self.MakeTS.setObjectName("MakeTS")
        self.label_16 = QtGui.QLabel(self.frame_3)
        self.label_16.setGeometry(QtCore.QRect(11, 10, 81, 22))
        self.label_16.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_16.setObjectName("label_16")
        self.label_18 = QtGui.QLabel(self.frame_3)
        self.label_18.setGeometry(QtCore.QRect(9, 100, 81, 22))
        self.label_18.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_18.setObjectName("label_18")
        self.label_20 = QtGui.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(310, 10, 311, 20))
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        TimeSeries.setCentralWidget(self.centralwidget)

        self.retranslateUi(TimeSeries)
        QtCore.QObject.connect(self.inputPSR, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputP0, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputP1, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputPOSEPOCH, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputPEPOCH, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputT0, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputPB, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputOM, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputE, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputINC, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputM1, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputM2, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QObject.connect(self.inputAMP, QtCore.SIGNAL("returnPressed()"), self.SaveParChanges.click)
        QtCore.QMetaObject.connectSlotsByName(TimeSeries)
        TimeSeries.setTabOrder(self.NumPSRs, self.AddPSRs)
        TimeSeries.setTabOrder(self.AddPSRs, self.PSRlist)
        TimeSeries.setTabOrder(self.PSRlist, self.RemovePSR)
        TimeSeries.setTabOrder(self.RemovePSR, self.ClearPSRs)
        TimeSeries.setTabOrder(self.ClearPSRs, self.MakeTS)
        TimeSeries.setTabOrder(self.MakeTS, self.inputPSR)
        TimeSeries.setTabOrder(self.inputPSR, self.inputP0)
        TimeSeries.setTabOrder(self.inputP0, self.inputP1)
        TimeSeries.setTabOrder(self.inputP1, self.inputPOSEPOCH)
        TimeSeries.setTabOrder(self.inputPOSEPOCH, self.inputPEPOCH)
        TimeSeries.setTabOrder(self.inputPEPOCH, self.inputT0)
        TimeSeries.setTabOrder(self.inputT0, self.inputPB)
        TimeSeries.setTabOrder(self.inputPB, self.inputOM)
        TimeSeries.setTabOrder(self.inputOM, self.inputE)
        TimeSeries.setTabOrder(self.inputE, self.inputINC)
        TimeSeries.setTabOrder(self.inputINC, self.inputM1)
        TimeSeries.setTabOrder(self.inputM1, self.inputM2)
        TimeSeries.setTabOrder(self.inputM2, self.SaveParChanges)

    def retranslateUi(self, TimeSeries):
        TimeSeries.setWindowTitle(QtGui.QApplication.translate("TimeSeries", "Time Series Editor", None, QtGui.QApplication.UnicodeUTF8))
        self.RemovePSR.setText(QtGui.QApplication.translate("TimeSeries", "Remove pulsar", None, QtGui.QApplication.UnicodeUTF8))
        self.ClearPSRs.setText(QtGui.QApplication.translate("TimeSeries", "Clear all", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("TimeSeries", "Number of pulsars:", None, QtGui.QApplication.UnicodeUTF8))
        self.AddPSRs.setText(QtGui.QApplication.translate("TimeSeries", "Add", None, QtGui.QApplication.UnicodeUTF8))
        self.inputPSR.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.inputPB.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Orbital period (days)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("TimeSeries", "PEPOCH:", None, QtGui.QApplication.UnicodeUTF8))
        self.inputT0.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Periastron epoch (MJD)", None, QtGui.QApplication.UnicodeUTF8))
        self.inputPEPOCH.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Period epoch (MJD)", None, QtGui.QApplication.UnicodeUTF8))
        self.inputOM.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Longitude of periastron (deg)", None, QtGui.QApplication.UnicodeUTF8))
        self.inputINC.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Inclination angle (deg)", None, QtGui.QApplication.UnicodeUTF8))
        self.inputM1.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "PSR mass (solar)", None, QtGui.QApplication.UnicodeUTF8))
        self.inputPOSEPOCH.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Position epoch (MJD)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("TimeSeries", "P1:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_10.setText(QtGui.QApplication.translate("TimeSeries", "INC:", None, QtGui.QApplication.UnicodeUTF8))
        self.inputM2.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Companion mass (solar)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("TimeSeries", "PSR:", None, QtGui.QApplication.UnicodeUTF8))
        self.inputP1.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Period derivative", None, QtGui.QApplication.UnicodeUTF8))
        self.label_11.setText(QtGui.QApplication.translate("TimeSeries", "M1:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("TimeSeries", "T0:", None, QtGui.QApplication.UnicodeUTF8))
        self.inputP0.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Period (s)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_12.setText(QtGui.QApplication.translate("TimeSeries", "M2:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("TimeSeries", "PB:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("TimeSeries", "POSEPOCH:", None, QtGui.QApplication.UnicodeUTF8))
        self.inputE.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Eccentricity", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("TimeSeries", "OM:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setText(QtGui.QApplication.translate("TimeSeries", "P0:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_13.setText(QtGui.QApplication.translate("TimeSeries", "E:", None, QtGui.QApplication.UnicodeUTF8))
        self.SaveParChanges.setText(QtGui.QApplication.translate("TimeSeries", "Save changes", None, QtGui.QApplication.UnicodeUTF8))
        self.inputAMP.setPlaceholderText(QtGui.QApplication.translate("TimeSeries", "Amplitude", None, QtGui.QApplication.UnicodeUTF8))
        self.label_21.setText(QtGui.QApplication.translate("TimeSeries", "Pulse amp (fraction of noise):", None, QtGui.QApplication.UnicodeUTF8))
        self.label_15.setText(QtGui.QApplication.translate("TimeSeries", "Editor", None, QtGui.QApplication.UnicodeUTF8))
        self.label_14.setText(QtGui.QApplication.translate("TimeSeries", "Time Series", None, QtGui.QApplication.UnicodeUTF8))
        self.AddToTS.setText(QtGui.QApplication.translate("TimeSeries", "Add to existing time series...", None, QtGui.QApplication.UnicodeUTF8))
        self.label_19.setText(QtGui.QApplication.translate("TimeSeries", "Bin size (s):", None, QtGui.QApplication.UnicodeUTF8))
        self.label_17.setText(QtGui.QApplication.translate("TimeSeries", "Length (s):", None, QtGui.QApplication.UnicodeUTF8))
        self.inputLength.setText(QtGui.QApplication.translate("TimeSeries", "1000", None, QtGui.QApplication.UnicodeUTF8))
        self.inputTres.setText(QtGui.QApplication.translate("TimeSeries", "8.192e-5", None, QtGui.QApplication.UnicodeUTF8))
        self.inputNoise.setText(QtGui.QApplication.translate("TimeSeries", "100", None, QtGui.QApplication.UnicodeUTF8))
        self.MakeTS.setText(QtGui.QApplication.translate("TimeSeries", "Create new time series...", None, QtGui.QApplication.UnicodeUTF8))
        self.label_16.setText(QtGui.QApplication.translate("TimeSeries", "Noise σ:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_18.setText(QtGui.QApplication.translate("TimeSeries", "Start (MJD):", None, QtGui.QApplication.UnicodeUTF8))
        self.label_20.setText(QtGui.QApplication.translate("TimeSeries", "Select a pulsar to view or edit its parameters", None, QtGui.QApplication.UnicodeUTF8))

