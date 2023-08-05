import numpy as np
import os
import pandas as pd

from Figures.figures import FiguresEnum
from Figures.figures import registered_figures
from Figures.figures import FigureGenerator
from Figures.figures import FigureData


class DatasetFigure:
    def __init__(self, dataset_name: str, figure_name: FiguresEnum, dimensions_size: int, figures_data=None):
        self.dataset_name = dataset_name
        self.figure_name = figure_name
        self.dimensions_size = dimensions_size
        self.figures_data = figures_data

    def save(self, root_path: str):
        if self.figures_data is None:
            return

        if os.path.exists(root_path):
            for i in range(len(self.figures_data)):
                figure_data = self.figures_data[i]
                csv_path = os.path.join(root_path, f'{i}.csv')

                df = pd.DataFrame(figure_data.points, columns=['X', 'Y'])
                df.to_csv(csv_path)

    def load(self, root_path: str):
        self.figures_data = []

        if os.path.exists(root_path):
            files = os.listdir(root_path)
            files = [file for file in files if file.endswith('.csv')]

            for i in range(len(files)):
                file_path = os.path.join(root_path, files[i])
                df = pd.read_csv(file_path)
                points = df.to_numpy()[:, 1:]
                data = FigureData(self.figure_name, self.dimensions_size, points=points)
                self.figures_data.append(data)

    def save_plot(self, root_path: str):
        if os.path.exists(root_path):
            for i in range(len(self.figures_data)):
                image_path = os.path.join(root_path, f'{i}.png')
                figure_data = self.figures_data[i]
                figure_data.plot(save_to=image_path)


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
        self.figures = registered_figures
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

            figure_dataset = DatasetFigure(self.__configuration.name, figure_name, self.__configuration.dimensions_size)
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

    def __generate_single(self, figure: FigureGenerator) -> DatasetFigure:
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
                    data = figure.draw(scale_x, scale_y, angle)
                    if not data.is_empty():
                        figures.append(data)

                    data = figure.draw(scale_x, scale_y, angle)
                    data.simplify()
                    if not data.is_empty():
                        figures.append(data)

                    if self.__configuration.distortion_percentage > 0:
                        data = figure.draw(scale_x, scale_y, angle)
                        y_offset = np.random.random(data.y.size) * (self.__configuration.dimensions_size * self.__configuration.distortion_percentage / 100)
                        y = data.y + y_offset
                        data.set_xy(data.x, y)

                        data.shift_to_zero()
                        data.scale_to_fit()
                        data.clip()
                        data.simplify()

                        if not data.is_empty():
                            figures.append(data)

                angle_start = angle_start + angle_offset
                if angle_start >= 360:
                    angle_start -= 360

                angle_stop = angle_stop + angle_offset
                if angle_stop >= 720:
                    angle_stop -= 360

        return DatasetFigure(self.__configuration.name, figure.name, self.__configuration.dimensions_size, figures)

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
