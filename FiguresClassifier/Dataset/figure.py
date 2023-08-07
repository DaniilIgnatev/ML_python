import os
import pandas as pd

from FiguresClassifier.Figures.generator import FiguresEnum
from FiguresClassifier.Figures.generator import FigureData


class FigureDataset:
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
