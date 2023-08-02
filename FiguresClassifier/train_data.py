import numpy as np
import os
import pandas as pd

from FiguresClassifier.figures import FiguresEnum
from FiguresClassifier.figures import registered_figures
from FiguresClassifier.figures import FigureGenerator
from FiguresClassifier.figures import FigureData


class DatasetFigure:
    def __init__(self, dataset_name: str, figure_name: FiguresEnum, figures_data=None):
        self.dataset_name = dataset_name
        self.figure_name = figure_name
        self.figures_data = figures_data

    def save(self, root_path: str):
        if self.figures_data is None:
            return

        folder_path = os.path.abspath(
            os.path.join(
                root_path,
                self.figure_name.name
            )
        )

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for i in range(len(self.figures_data)):
            figure_data = self.figures_data[i]
            csv_path = os.path.join(folder_path, f'{i}.csv')

            df = pd.DataFrame(figure_data.points, columns=['X', 'Y'])
            df.to_csv(csv_path)

    def load(self, root_path: str):
        self.figures_data = []

        folder_path = os.path.abspath(
            os.path.join(
                root_path,
                self.figure_name.name
            )
        )

        files = os.listdir(folder_path)
        files = [file for file in files if file.endswith('.csv')]

        for i in range(len(files)):
            file_path = os.path.join(folder_path, files[i])
            df = pd.read_csv(file_path)
            points = df.to_numpy()[:, 1:]
            data = FigureData(self.figure_name, points=points)
            self.figures_data.append(data)

    def plot(self, root_path: str):
        folder_path = os.path.abspath(
            os.path.join(
                root_path,
                self.figure_name.name
            )
        )

        for i in range(len(self.figures_data)):
            image_path = os.path.join(folder_path, f'{i}.png')
            figure_data = self.figures_data[i]
            figure_data.plot(save_to=image_path)


class DatasetGenerator:
    def __init__(self, name: str, path: str, figures: dict[FiguresEnum, FigureGenerator], size: int, min_scale, max_scale, scale_precision,
                 min_angle, max_angle, angle_precision, plot_figures=False):
        self.name = name
        self.figures = figures
        self.size = size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_precision = scale_precision
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.angle_precision = angle_precision
        self.plot_figures = plot_figures
        self.root_path = os.path.join(path, self.name)

        self.dataset = {}
        if self.__can_load():
            self.__load()
        else:
            self.__generate_all()
            self.__save()
            self.__plot()

    def __can_load(self) -> bool:
        return os.path.exists(self.root_path)

    def __load(self):
        self.dataset = {}

        for figure_name in self.figures.keys():
            figure_dataset = DatasetFigure(self.name, figure_name)
            figure_dataset.load(self.root_path)
            self.dataset[figure_name] = figure_dataset

    def __save(self):
        for figure_dataset in self.dataset.values():
            figure_dataset.save(self.root_path)

    def __generate_single(self, figure: FigureGenerator) -> DatasetFigure:
        figures = []

        angle_start = self.min_angle
        angle_stop = self.max_angle
        angle_offset = self.angle_precision / 2

        for scale_x in np.linspace(self.min_scale, self.max_scale,
                                   int((self.max_scale - self.min_scale) / self.scale_precision) + 1, endpoint=True):
            for scale_y in np.linspace(self.min_scale, self.max_scale,
                                       int((self.max_scale - self.min_scale) / self.scale_precision) + 1,
                                       endpoint=True):
                for angle in np.linspace(angle_start, angle_stop,
                                         int((self.max_angle - self.min_angle) / self.angle_precision) + 1,
                                         endpoint=True):
                    data = figure.draw(scale_x, scale_y, angle)

                    if not data.is_empty():
                        figures.append(data)

                angle_start = angle_start + angle_offset
                if angle_start >= 360:
                    angle_start -= 360

                angle_stop = angle_stop + angle_offset
                if angle_stop >= 720:
                    angle_stop -= 360

        return DatasetFigure(self.name, figure.name, figures)

    def __generate_all(self):
        for figure_key in self.figures:
            figure: FigureGenerator = self.figures[figure_key](self.size, False)
            data_single = self.__generate_single(figure)
            self.dataset[figure_key] = data_single

    def __plot(self):
        for figure_dataset in self.dataset.values():
            figure_dataset.plot(self.root_path)

    def get_labled_data(self, target_figure):
        pass


# %%
dataset = DatasetGenerator('32_05_35_1_0_360_30', 'C:\\Users\\Daniil\\Desktop\\datasets', registered_figures, 32, 0.5, 3.5, 1, 0, 360, 30)

# dataset = DatasetGenerator('64_025_2_025_0_360_30',registered_figures, 64, 0.25, 2, 0.25, 0, 360, 30)
#
# dataset = DatasetGenerator('default', registered_figures, 64, 1, 1, 1, 0, 360, 60
#%%
