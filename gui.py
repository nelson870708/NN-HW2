# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1214, 522)
        self.groupBox_setting = QtWidgets.QGroupBox(Form)
        self.groupBox_setting.setGeometry(QtCore.QRect(20, 30, 331, 211))
        self.groupBox_setting.setObjectName("groupBox_setting")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox_setting)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 20, 311, 182))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.doubleSpinBox_stop_condition = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.doubleSpinBox_stop_condition.setMaximum(1.0)
        self.doubleSpinBox_stop_condition.setSingleStep(0.01)
        self.doubleSpinBox_stop_condition.setProperty("value", 0.01)
        self.doubleSpinBox_stop_condition.setObjectName("doubleSpinBox_stop_condition")
        self.gridLayout.addWidget(self.doubleSpinBox_stop_condition, 3, 1, 1, 1)
        self.pushButton_start = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_start.setObjectName("pushButton_start")
        self.gridLayout.addWidget(self.pushButton_start, 5, 1, 1, 1)
        self.label_train_test = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_train_test.setObjectName("label_train_test")
        self.gridLayout.addWidget(self.label_train_test, 5, 0, 1, 1)
        self.doubleSpinBox_learning_rate = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.doubleSpinBox_learning_rate.setMaximum(1.0)
        self.doubleSpinBox_learning_rate.setSingleStep(0.01)
        self.doubleSpinBox_learning_rate.setProperty("value", 0.2)
        self.doubleSpinBox_learning_rate.setObjectName("doubleSpinBox_learning_rate")
        self.gridLayout.addWidget(self.doubleSpinBox_learning_rate, 2, 1, 1, 1)
        self.comboBox_file = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboBox_file.setObjectName("comboBox_file")
        self.gridLayout.addWidget(self.comboBox_file, 0, 1, 1, 1)
        self.label_stop_condition = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_stop_condition.setObjectName("label_stop_condition")
        self.gridLayout.addWidget(self.label_stop_condition, 3, 0, 1, 1)
        self.label_file = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_file.setObjectName("label_file")
        self.gridLayout.addWidget(self.label_file, 0, 0, 1, 1)
        self.label_iteration = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_iteration.setObjectName("label_iteration")
        self.gridLayout.addWidget(self.label_iteration, 1, 0, 1, 1)
        self.doubleSpinBox_iteration = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.doubleSpinBox_iteration.setDecimals(0)
        self.doubleSpinBox_iteration.setMaximum(999999.0)
        self.doubleSpinBox_iteration.setProperty("value", 100.0)
        self.doubleSpinBox_iteration.setObjectName("doubleSpinBox_iteration")
        self.gridLayout.addWidget(self.doubleSpinBox_iteration, 1, 1, 1, 1)
        self.label_learning_rate = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_learning_rate.setObjectName("label_learning_rate")
        self.gridLayout.addWidget(self.label_learning_rate, 2, 0, 1, 1)
        self.label_neuron_number = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_neuron_number.setObjectName("label_neuron_number")
        self.gridLayout.addWidget(self.label_neuron_number, 4, 0, 1, 1)
        self.doubleSpinBox_neuron_number = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.doubleSpinBox_neuron_number.setDecimals(0)
        self.doubleSpinBox_neuron_number.setMaximum(100.0)
        self.doubleSpinBox_neuron_number.setSingleStep(1.0)
        self.doubleSpinBox_neuron_number.setProperty("value", 4.0)
        self.doubleSpinBox_neuron_number.setObjectName("doubleSpinBox_neuron_number")
        self.gridLayout.addWidget(self.doubleSpinBox_neuron_number, 4, 1, 1, 1)
        self.groupBox_result = QtWidgets.QGroupBox(Form)
        self.groupBox_result.setGeometry(QtCore.QRect(20, 260, 331, 231))
        self.groupBox_result.setObjectName("groupBox_result")
        self.textBrowser_result = QtWidgets.QTextBrowser(self.groupBox_result)
        self.textBrowser_result.setGeometry(QtCore.QRect(10, 20, 311, 201))
        self.textBrowser_result.setObjectName("textBrowser_result")
        self.tabWidget_visualization = QtWidgets.QTabWidget(Form)
        self.tabWidget_visualization.setGeometry(QtCore.QRect(390, 20, 801, 471))
        self.tabWidget_visualization.setObjectName("tabWidget_visualization")
        self.train_graph = QtWidgets.QWidget()
        self.train_graph.setObjectName("train_graph")
        self.train_graph_truth = QtWidgets.QWidget(self.train_graph)
        self.train_graph_truth.setGeometry(QtCore.QRect(0, 0, 400, 400))
        self.train_graph_truth.setObjectName("train_graph_truth")
        self.train_graph_predict = QtWidgets.QWidget(self.train_graph)
        self.train_graph_predict.setGeometry(QtCore.QRect(400, 0, 400, 400))
        self.train_graph_predict.setObjectName("train_graph_predict")
        self.label_train_truth = QtWidgets.QLabel(self.train_graph)
        self.label_train_truth.setGeometry(QtCore.QRect(80, 400, 200, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_train_truth.setFont(font)
        self.label_train_truth.setAlignment(QtCore.Qt.AlignCenter)
        self.label_train_truth.setObjectName("label_train_truth")
        self.label_train_predict = QtWidgets.QLabel(self.train_graph)
        self.label_train_predict.setGeometry(QtCore.QRect(500, 400, 200, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_train_predict.setFont(font)
        self.label_train_predict.setAlignment(QtCore.Qt.AlignCenter)
        self.label_train_predict.setObjectName("label_train_predict")
        self.tabWidget_visualization.addTab(self.train_graph, "")
        self.test_graph = QtWidgets.QWidget()
        self.test_graph.setObjectName("test_graph")
        self.test_graph_predict = QtWidgets.QWidget(self.test_graph)
        self.test_graph_predict.setGeometry(QtCore.QRect(400, 0, 400, 400))
        self.test_graph_predict.setObjectName("test_graph_predict")
        self.label_test_preidct = QtWidgets.QLabel(self.test_graph)
        self.label_test_preidct.setGeometry(QtCore.QRect(500, 400, 200, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_test_preidct.setFont(font)
        self.label_test_preidct.setAlignment(QtCore.Qt.AlignCenter)
        self.label_test_preidct.setObjectName("label_test_preidct")
        self.test_graph_truth = QtWidgets.QWidget(self.test_graph)
        self.test_graph_truth.setGeometry(QtCore.QRect(0, 0, 400, 400))
        self.test_graph_truth.setObjectName("test_graph_truth")
        self.label_test_truth = QtWidgets.QLabel(self.test_graph)
        self.label_test_truth.setGeometry(QtCore.QRect(80, 400, 200, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_test_truth.setFont(font)
        self.label_test_truth.setAlignment(QtCore.Qt.AlignCenter)
        self.label_test_truth.setObjectName("label_test_truth")
        self.tabWidget_visualization.addTab(self.test_graph, "")

        self.retranslateUi(Form)
        self.tabWidget_visualization.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox_setting.setTitle(_translate("Form", "Setting"))
        self.pushButton_start.setText(_translate("Form", "Start"))
        self.label_train_test.setText(_translate("Form", "Train and Test"))
        self.label_stop_condition.setText(_translate("Form", "Stop Condition:"))
        self.label_file.setText(_translate("Form", "Select File: "))
        self.label_iteration.setText(_translate("Form", "Iteration:"))
        self.label_learning_rate.setText(_translate("Form", "Learning Rate:"))
        self.label_neuron_number.setText(_translate("Form", "Neurons of Layer"))
        self.groupBox_result.setTitle(_translate("Form", "Results"))
        self.label_train_truth.setText(_translate("Form", "Truth"))
        self.label_train_predict.setText(_translate("Form", "Predict"))
        self.tabWidget_visualization.setTabText(self.tabWidget_visualization.indexOf(self.train_graph), _translate("Form", "Train Graph"))
        self.label_test_preidct.setText(_translate("Form", "Predict"))
        self.label_test_truth.setText(_translate("Form", "Truth"))
        self.tabWidget_visualization.setTabText(self.tabWidget_visualization.indexOf(self.test_graph), _translate("Form", "Test Graph"))
