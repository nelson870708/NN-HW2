# python library
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ResultFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(ResultFigure, self).__init__(self.fig)  # 此句必不可少，否則不能顯示圖形
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot_point(self, x, y):
        self.axes.clear()
        self.axes.grid()
        self.axes.scatter(x[:, 1], x[:, 2], c=y.reshape(len(y)), alpha=0.8)  # 畫點
        self.axes.autoscale(enable=False, axis='both', tight=None)  # 讓畫布大小固定
        self.draw()

    def plot_clear(self):
        self.axes.clear()
        self.draw()
