import os
import numpy as np

from FiguresClassifier.Dataset.generator import DatasetGenerator
from FiguresClassifier.Dataset.generator import FigureDataset
from FiguresClassifier.Figures.generator import FigureData
from NN.neural_network import NeuralNetwork


from FiguresClassifier.Dataset.generator import DatasetGeneratorConfiguration
from FiguresClassifier.Figures.generator import FiguresEnum
from NN.neural_network import NeuralNetworkConfiguration


class ClassifierConfiguration:
    def __init__(self,
                 _name: str,
                 path: str,
                 target_figure: FiguresEnum,
                 dimensions_size: int,
                 train_dataset_configuration: DatasetGeneratorConfiguration,
                 test_dataset_configuration: DatasetGeneratorConfiguration,
                 nn_configuration: NeuralNetworkConfiguration
                 ):
        self.name = _name
        self.path = path
        self.target_figure = target_figure
        self.dimensions_size = dimensions_size
        self.train_dataset_configuration = train_dataset_configuration
        self.test_dataset_configuration = test_dataset_configuration
        self.nn_configuration = nn_configuration


class Classifier(NeuralNetwork):
    def __init__(self, configuration: ClassifierConfiguration):
        super().__init__(configuration.nn_configuration)
        self.__configuration = configuration

        if not os.path.exists(configuration.path):
            os.mkdir(configuration.path)

        self.root_path = os.path.join(configuration.path, self.__configuration.name)

        self.__samples_test = []
        self.__samples_train = []

        self.__train_dataset_generator = None
        self.__test_dataset_generator = None

        if self._can_load(self.root_path):
            self._load(self.root_path)

    def data_from_figure_data(self, figure_data: FigureData):
        y: float = figure_data.name == self.__configuration.target_figure

        data = np.full((self.__configuration.dimensions_size ** 2, 1), -1)

        for p in figure_data.points:
            index = int(p[1] * self.__configuration.dimensions_size + p[0])
            data[index] = 1
        return data, y

    def __train_data_from_figure(self, figure_dataset: FigureDataset):
        for figure_data in figure_dataset.figures_data:
            sample = self.data_from_figure_data(figure_data)
            self.__samples_train.append(sample)

    def __test_data_from_figure(self, figure_dataset: FigureDataset):
        for figure_data in figure_dataset.figures_data:
            sample = self.data_from_figure_data(figure_data)
            self.__samples_test.append(sample)

    def __generate_samples(self):
        if self.__train_dataset_generator is None:
            self.__train_dataset_generator = DatasetGenerator(self.__configuration.train_dataset_configuration)

        if self.__test_dataset_generator is None:
            self.__test_dataset_generator = DatasetGenerator(self.__configuration.test_dataset_configuration)

        for value in self.__train_dataset_generator.dataset.values():
            self.__train_data_from_figure(value)

        for value in self.__test_dataset_generator.dataset.values():
            self.__test_data_from_figure(value)

    def get_train_data(self) -> list[float]:
        """
        Returns a copy of the train dataset
        """
        return self.__samples_train.copy()

    def get_test_data(self) -> list[float]:
        """
        Returns a copy of the test dataset
        """
        return self.__samples_test.copy()

    def train_classifier(self):
        self.__generate_samples()

        test_data = self.get_test_data()
        print(f'Samples successfully classified before training: {self.evaluate(test_data)} of {len(test_data)}')

        training_data = self.get_train_data()
        self.train(training_data=training_data, test_data=test_data, random_shuffle=True)
        print(f'Samples successfully classified after training: {self.evaluate(test_data)} of {len(test_data)}')

        self._save(self.root_path)

    def classify(self, x, y) -> float:
        if x is list:
            x = np.array(x)

        if y is list:
            y = np.array(y)

        data = FigureData(self.__configuration.target_figure, self.__configuration.dimensions_size, x, y)
        data.shift_to_zero()
        data.scale_to_fit()
        data.clip()
        data.filter()
        nn_input, label = self.data_from_figure_data(data)

        p = float(self.forward_pass(nn_input))
        return p
