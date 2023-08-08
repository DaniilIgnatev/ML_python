import numpy as np
import os

from FiguresClassifier.Figures.generator import registered_generators
from FiguresClassifier.Figures.generator import FigureGenerator
from FiguresClassifier.Dataset.figure import FigureDataset


class DatasetGeneratorConfiguration:
    def __init__(self,
                 name,
                 path,
                 dimensions_size,
                 min_scale,
                 max_scale,
                 scale_precision,
                 min_angle,
                 max_angle,
                 angle_precision,
                 distortion_percentage,
                 save_plots
                 ):
        self.name = name
        self.path = path
        self.dimensions_size = dimensions_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_precision = scale_precision
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.angle_precision = angle_precision
        self.distortion_percentage = distortion_percentage
        self.save_plots = save_plots


class DatasetGenerator:
    def __init__(self, configuration: DatasetGeneratorConfiguration):
        self.__configuration = configuration
        self.figures = registered_generators
        self.root_path = os.path.join(configuration.path, self.__configuration.name)

        # self._print_size()

        self.dataset = {}
        if self.__can_load():
            self.__load()
        else:
            self.__generate_all()
            self.__save()
            if self.__configuration.save_plots:
                self.__save_plot()

    def __can_load(self) -> bool:
        return os.path.exists(self.root_path)

    def __load(self):
        self.dataset = {}

        for figure_name in self.figures.keys():
            folder_path = os.path.abspath(
                os.path.join(
                    self.root_path,
                    figure_name.name
                )
            )

            figure_dataset = FigureDataset(self.__configuration.name, figure_name, self.__configuration.dimensions_size)
            figure_dataset.load(folder_path)
            self.dataset[figure_name] = figure_dataset

    def __save(self):
        for figure_dataset in self.dataset.values():
            folder_path = os.path.abspath(
                os.path.join(
                    self.root_path,
                    figure_dataset.figure_name.name
                )
            )

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            figure_dataset.save(folder_path)

    def __save_plot(self):
        for figure_dataset in self.dataset.values():
            folder_path = os.path.abspath(
                os.path.join(
                    self.root_path,
                    figure_dataset.figure_name.name
                )
            )

            figure_dataset.save_plot(folder_path)

    def __generate_single(self, figure: FigureGenerator) -> FigureDataset:
        figures = []

        angle_start = self.__configuration.min_angle
        angle_stop = self.__configuration.max_angle
        angle_offset = self.__configuration.angle_precision / 2

        for scale_x in np.linspace(self.__configuration.min_scale, self.__configuration.max_scale,
                                   int((self.__configuration.max_scale - self.__configuration.min_scale) / self.__configuration.scale_precision) + 1, endpoint=True):
            for scale_y in np.linspace(self.__configuration.min_scale, self.__configuration.max_scale,
                                       int((self.__configuration.max_scale - self.__configuration.min_scale) / self.__configuration.scale_precision) + 1,
                                       endpoint=True):
                for angle in np.linspace(angle_start, angle_stop,
                                         int((self.__configuration.max_angle - self.__configuration.min_angle) / self.__configuration.angle_precision) + 1,
                                         endpoint=True):
                    data = figure.draw(scale_x, scale_y, angle, 0)
                    data.simplify(0.8)
                    if not data.is_empty():
                        figures.append(data)

                    if self.__configuration.distortion_percentage > 0:
                        data = figure.draw(scale_x, scale_y, angle, self.__configuration.distortion_percentage)
                        data.simplify(0.8)
                        if not data.is_empty():
                            figures.append(data)

                    data = figure.draw(scale_x, scale_y, angle, 0)
                    data.simplify(0.7)
                    if not data.is_empty():
                        figures.append(data)

                    if self.__configuration.distortion_percentage > 0:
                        data = figure.draw(scale_x, scale_y, angle, self.__configuration.distortion_percentage)
                        data.simplify(0.7)
                        if not data.is_empty():
                            figures.append(data)

                angle_start = angle_start + angle_offset
                if angle_start >= 360:
                    angle_start -= 360

                angle_stop = angle_stop + angle_offset
                if angle_stop >= 720:
                    angle_stop -= 360

        return FigureDataset(self.__configuration.name, figure.name, self.__configuration.dimensions_size, figures)

    def __generate_all(self):
        for figure_key in self.figures:
            figure: FigureGenerator = self.figures[figure_key](self.__configuration.dimensions_size, False)
            data_single = self.__generate_single(figure)
            self.dataset[figure_key] = data_single

    def _print_size(self):
        print(f'Dataset: {self.__configuration.name}')

        scale_steps = (self.__configuration.max_scale - self.__configuration.min_scale) / self.__configuration.scale_precision
        rotation_steps = (self.__configuration.max_angle - self.__configuration.min_angle) / self.__configuration.angle_precision
        samples_figure = scale_steps * scale_steps * rotation_steps * 3

        print(f'Samples for a single figure: {samples_figure}')
        print(f'Total samples: {samples_figure * len(self.figures)}')
