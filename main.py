# 將資料讀取到的資料進行前處理
# 傳給algorithm進行多層感知機演算法
# 將每個資料點的座標畫在Euclidean座標系統上

# 原生python library
import glob
import os
import sys

# 另外安裝的python library
import matplotlib
import numpy as np
from PyQt5.QtWidgets import *
from sklearn.model_selection import train_test_split

# 自己寫的python library
import algorithm
from gui import Ui_Form
from result_plot import ResultFigure

matplotlib.use('Qt5Agg')  # 宣告使用QT5


def read_file():
    file_elementlist = []
    file_namelist = {}
    files_name = glob.glob(os.path.join(os.getcwd() + '\\data', '*.txt'))
    i = 0
    for file_name in files_name:
        file_namelist[os.path.basename(file_name)] = i
        file = open(file_name, 'r')
        elementlist = []
        for line in file:
            elementlist.append(list(map(float, line.split(' '))))
        file_elementlist.append(elementlist)
        i = i + 1
    return file_namelist, file_elementlist


class Main(QDialog, Ui_Form):
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("HW2")
        self.setMinimumSize(0, 0)

        self.file_namelist, self.file_elementlist = read_file()
        self.comboBox_file.addItems(sorted(list(self.file_namelist)))
        self.train_truth_graph = ResultFigure(width=100, height=100, dpi=100)
        self.train_predict_graph = ResultFigure(width=100, height=100, dpi=100)
        self.test_truth_graph = ResultFigure(width=100, height=100, dpi=100)
        self.test_predict_graph = ResultFigure(width=100, height=100, dpi=100)
        self.file_changed()  # 畫第一筆資料之資料點

        # 加入train graph(truth)
        vlayout1 = QVBoxLayout(self.train_graph_truth)
        self.tabwidget1 = QTabWidget()
        vlayout1.addWidget(self.tabwidget1)
        self.gridlayout1 = QGridLayout(self.tabwidget1)
        self.gridlayout1.addWidget(self.train_truth_graph)

        # 加入train graph(predict)
        vlayout2 = QVBoxLayout(self.train_graph_predict)
        self.tabwidget2 = QTabWidget()
        vlayout2.addWidget(self.tabwidget2)
        self.gridlayout2 = QGridLayout(self.tabwidget2)
        self.gridlayout2.addWidget(self.train_predict_graph)

        # 加入test graph(truth)
        vlayout3 = QVBoxLayout(self.test_graph_truth)
        self.tabwidget3 = QTabWidget()
        vlayout3.addWidget(self.tabwidget3)
        self.gridlayout3 = QGridLayout(self.tabwidget3)
        self.gridlayout3.addWidget(self.test_truth_graph)

        # 加入test graph(predict)
        vlayout4 = QVBoxLayout(self.test_graph_predict)
        self.tabwidget4 = QTabWidget()
        vlayout4.addWidget(self.tabwidget4)
        self.gridlayout4 = QGridLayout(self.tabwidget4)
        self.gridlayout4.addWidget(self.test_predict_graph)

        # 初始化部分數值
        self.learning_rate = self.doubleSpinBox_learning_rate.value()
        self.iteration = self.doubleSpinBox_iteration.value()
        self.early_stop = self.doubleSpinBox_stop_condition.value()
        self.num_neuron = self.doubleSpinBox_neuron_number.value()

        # 觸發事件才使用
        self.comboBox_file.currentTextChanged.connect(self.file_changed)
        self.doubleSpinBox_learning_rate.valueChanged.connect(self.learning_rate_changed)
        self.doubleSpinBox_iteration.valueChanged.connect(self.iteration_changed)
        self.doubleSpinBox_stop_condition.valueChanged.connect(self.early_stop_changed)
        self.doubleSpinBox_neuron_number.valueChanged.connect(self.num_neuron_changed)
        self.pushButton_start.clicked.connect(self.start_buttom_click)

    def file_changed(self):
        self.textBrowser_result.setText('')
        self.train_predict_graph.plot_clear()
        self.test_predict_graph.plot_clear()
        X = np.asarray(self.file_elementlist[self.file_namelist[self.comboBox_file.currentText()]])[:, :2]
        y = np.asarray(self.file_elementlist[self.file_namelist[self.comboBox_file.currentText()]])[:, 2:]
        X = np.concatenate((np.negative(np.ones((len(X), 1))), X), axis=1)
        if self.comboBox_file.currentText() == 'perceptron1.txt' \
                or self.comboBox_file.currentText() == 'perceptron2.txt':
            self.train_X = X
            self.test_X = X
            self.train_y = y
            self.test_y = y
        else:
            self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, train_size=2 / 3,
                                                                                    test_size=1 / 3)
        self.train_truth_graph.plot_point(self.train_X, self.train_y)
        self.test_truth_graph.plot_point(self.test_X, self.test_y)

        self.train_num_data = len(self.train_X)
        self.test_num_data = len(self.test_X)

    def learning_rate_changed(self):
        self.learning_rate = self.doubleSpinBox_learning_rate.value()

    def iteration_changed(self):
        self.iteration = self.doubleSpinBox_iteration.value()

    def early_stop_changed(self):
        self.early_stop = self.doubleSpinBox_stop_condition.value()

    def num_neuron_changed(self):
        self.num_neuron = self.doubleSpinBox_neuron_number.value()

    def start_buttom_click(self):

        # MLP algorithm(training)
        weight, train_e, train_accuracy, train_expected = algorithm.mlp_train(self.train_X, self.train_y,
                                                                              anskey=np.unique(self.train_y),
                                                                              lr=self.learning_rate,
                                                                              it=int(self.iteration),
                                                                              es=self.early_stop,
                                                                              num_neuron=self.num_neuron)

        # MLP algorithm(testing)
        test_e, test_accuracy, test_expected = algorithm.mlp_test(self.test_X, self.test_y,
                                                                  anskey=np.unique(self.test_y), w=weight)
        self.textBrowser_result.append('The accuracy of train data is ' + str(train_accuracy) +
                                       '\nThe MSE of train data is ' + str(train_e) +
                                       '\nThe accuracy of test data is ' + str(test_accuracy) +
                                       '\nThe MSE of test data is ' + str(test_e) +
                                       '\nThe weight of the cell is ' + str(weight) + '\n')

        self.train_predict_graph.plot_point(x=self.train_X, y=np.array(train_expected))
        self.test_predict_graph.plot_point(x=self.test_X, y=np.array(test_expected))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
