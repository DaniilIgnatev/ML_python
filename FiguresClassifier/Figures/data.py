import numpy as np
import matplotlib.pyplot as plt
import gc
from enum import Enum


class FiguresEnum(Enum):
    NOISE = 'NOISE'
    LINE = 'LINE'
    TRIANGLE = 'TRIANGLE'
    RECTANGLE = 'RECTANGLE'
    PENTAGON = 'PENTAGON'
    HEXAGON = 'HEXAGON'
    OCTAGON = 'OCTAGON'
    ELLIPSE = 'ELLIPSE'


class FigureData:
    def __init__(self, name: FiguresEnum, dimensions_size: int, x: np.ndarray = None, y: np.ndarray = None, points: np.ndarray = None):
        self.name = name
        self.dimensions_size = dimensions_size
        self.points = np.array([[]])
        self.x = np.array([])
        self.y = np.array([])

        self.fig = None
        self.ax = None

        if points is not None:
            self.set_points(points)
        else:
            if x is not None and y is not None:
                self.set_xy(x, y)

    def __copy__(self):
        return FigureData(self.name, self.dimensions_size, x=self.x.copy(), y=self.y.copy())

    def set_points(self, points):
        if points.ndim == 2:
            self.points = points
            self.x = self.points[:, 0]
            self.y = self.points[:, 1]

            if self.is_empty:
                self.set_xy(points[:, 0], points[:, 1])
            else:
                self.set_xy(np.array([]), np.array([]))
        else:
            self.set_xy(np.array([]), np.array([]))

    def set_xy(self, x, y):
        self.x = x
        self.y = y
        self.points = np.array(list(zip(x, y)))

    def append_xy(self, x, y):
        self.set_xy(np.append(self.x, x), np.append(self.y, y))

    def is_empty(self) -> bool:
        return self.points.ndim != 2

    def plot(self, x_lim=None, y_lim=None, save_to=None):
        if self.fig:
            plt.close(self.fig)
        self.fig, self.ax = plt.subplots()

        min_x = np.min(self.x)
        max_x = np.max(self.x)

        min_y = np.min(self.y)
        max_y = np.max(self.y)

        if x_lim:
            self.ax.set_xlim(x_lim)
        else:
            offset_x = (max_x - min_x) / 10
            self.ax.set_xlim([min_x - offset_x, max_x + offset_x])

        if y_lim:
            self.ax.set_ylim(y_lim)
        else:
            offset_y = (max_y - min_y) / 10
            self.ax.set_ylim([min_y - offset_y, max_y + offset_y])

        self.ax.scatter(self.x, self.y)
        self.ax.invert_yaxis()

        if save_to:
            self.fig.savefig(save_to)
            plt.clf()
            plt.close()
            del self.fig
            gc.collect()
        else:
            plt.show()

    def scale_key_points(self):
        max = np.max(self.points)
        points = self.points / max * (self.dimensions_size - 1)
        self.set_points(points)

    def scale(self, scale_x, scale_y):
        if self.name == FiguresEnum.LINE:
            return

        S = np.array([[scale_x, 0],
                      [0, scale_y]])

        points = self.points @ S
        self.set_points(points)

    def rotate(self, angle):
        if self.name == FiguresEnum.LINE:
            return

        angle = -angle
        max = np.max(self.points)
        center_point = np.array([max / 2, max / 2])

        points = self.points - center_point
        theta = np.radians(angle)

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        points = [R @ p for p in points]
        points = points + center_point
        self.set_points(points)

    def clip(self):
        points = self.points
        clipped_points = []

        for p in points:
            if not np.isnan(p[0]):
                if not np.isnan(p[1]):
                    if 0 <= p[0] < self.dimensions_size:
                        if 0 <= p[1] < self.dimensions_size:
                            clipped_points.append(np.array([p[0], p[1]]))

        clipped_points = np.array(clipped_points)
        self.set_points(clipped_points)

    def filter(self):
        points_dict = {}

        for p in self.points:
            if p.size != 0:
                if not np.isnan(p[0]):
                    if not np.isnan(p[1]):
                        x_int = int(p[0])
                        y_int = int(p[1])
                        point = np.array([x_int, y_int])

                        if x_int not in points_dict:
                            points_dict[x_int] = np.array([point])
                        else:
                            values = points_dict[x_int]
                            if not (np.any(np.all(values == point, axis=1))):
                                points_dict[x_int] = np.vstack([values, point])
                            # else:
                            #     print('repeat')

        values = list(points_dict.values())
        values_flattened = []
        for v in values:
            for _v in v:
                values_flattened.append(np.array(_v))
        values_flattened = np.array(values_flattened)

        self.set_points(values_flattened)

    def shift_to_zero(self):
        x_min = np.min(self.x)
        X = self.x - x_min

        y_min = np.min(self.y)
        Y = self.y - y_min

        self.set_xy(X, Y)

    def scale_to_fit(self):
        X = self.x
        Y = self.y

        x_min = np.min(X)
        x_max = np.max(X)
        width = x_max - x_min
        if width != 0:
            width_ratio = width / (self.dimensions_size - 1)
            X = X / width_ratio

        y_min = np.min(Y)
        y_max = np.max(Y)
        height = y_max - y_min
        if height != 0:
            height_ratio = height / (self.dimensions_size - 1)
            Y = Y / height_ratio

        # x_min = np.min(X)
        # x_max = np.max(X)
        # width = x_max - x_min
        # width_ratio = width / (self.dimensions_size - 1)
        #
        # y_min = np.min(Y)
        # y_max = np.max(Y)
        # height = y_max - y_min
        # height_ratio = height / (self.dimensions_size - 1)
        #
        # h_to_w_ratio = height / width
        #
        # X = X / width_ratio
        # Y = Y / height_ratio * h_to_w_ratio

        self.set_xy(X, Y)

    def simplify(self, factor: float):
        """
        Keeps factor[0..1] part of the points
        :return:
        """
        factor = np.clip(factor, 0, 1)
        factor = 1 - factor
        indices_to_delete = np.random.choice(np.arange(len(self.points)), size=int(len(self.points) * factor), replace=False)
        points = np.delete(self.points, indices_to_delete, axis=0)
        self.set_points(points)

    def distortion(self, percentage: float):
        y_offset = np.random.random(self.y.size) * (self.dimensions_size * percentage / 100)
        y = self.y + y_offset
        self.set_xy(self.x, y)
