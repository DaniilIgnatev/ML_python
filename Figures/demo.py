import os
import random
import numpy as np
from Figures.classifier import ClassifierConfiguration
from Figures.classifier import Classifier
from Figures.train_data import DatasetGeneratorConfiguration
from Figures.figures import FiguresEnum
from NN.neural_network import NeuralNetworkConfiguration
from NN.optimizer import OptimizerConfiguration
from NN.optimizer import OptimizerEnum

import matplotlib.pyplot as plt


class ClassifierDemo:
    def __init__(self, classifier: Classifier):
        self.classifier = classifier

    def open_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([0, ])
        ax.set_ylim([0, 10])
        ax.invert_yaxis()

        coordinates = []

        def onclick(event):
            ix, iy = event.xdata, event.ydata
            print(f'x = {ix}, y = {iy}')

            global coordinates
            coordinates.append((ix, iy))

            ax.plot(ix, iy, 'ro')
            fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
