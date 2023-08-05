import numpy as np
import numpy.linalg as la
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

        if points is not None:
            self.set_points(points)
        else:
            if x is not None and y is not None:
                self.set_xy(x, y)

    def __copy__(self):
        return FigureData(self.name, self.dimensions_size, x=self.x.copy(), y=self.y.copy())

    def set_points(self, points):
        self.points = points
        self.x = self.points[:, 0]
        self.y = self.points[:, 1]

        if self.is_empty:
            self.set_xy(points[:, 0], points[:, 1])
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
        fig, ax = plt.subplots()

        min_x = np.min(self.x)
        max_x = np.max(self.x)

        min_y = np.min(self.y)
        max_y = np.max(self.y)

        if x_lim:
            ax.set_xlim(x_lim)
        else:
            offset_x = (max_x - min_x) / 10
            ax.set_xlim([min_x - offset_x, max_x + offset_x])

        if y_lim:
            ax.set_ylim(y_lim)
        else:
            offset_y = (max_y - min_y) / 10
            ax.set_ylim([min_y - offset_y, max_y + offset_y])

        ax.scatter(self.x, self.y)
        ax.invert_yaxis()

        if save_to:
            fig.savefig(save_to)
            plt.clf()
            plt.close()
            del fig
            gc.collect()
        else:
            plt.show()

    def scale_key_points(self):
        max = np.max(self.points)
        points = self.points / max * (self.dimensions_size - 1)
        self.set_points(points)

    def scale(self, scale_x, scale_y):
        S = np.array([[scale_x, 0],
                      [0, scale_y]])

        points = self.points @ S
        self.set_points(points)

    def rotate(self, angle):
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
        filtered_points = []

        for p in points:
            if 0 <= p[0] < self.dimensions_size:
                if 0 <= p[1] < self.dimensions_size:
                    filtered_points.append(np.array([p[0], p[1]]))

        filtered_points = np.array(filtered_points)
        self.set_points(filtered_points)

    def filter(self):
        points_dict = {}

        for p in self.points:
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
        width_ratio = width / (self.dimensions_size - 1)
        X = X / width_ratio

        y_min = np.min(Y)
        y_max = np.max(Y)
        height = y_max - y_min
        height_ratio = height / (self.dimensions_size - 1)
        Y = Y / height_ratio

        self.set_xy(X, Y)

    def simplify(self):
        """
        Deletes every second point
        :return:
        """
        indices_to_delete = np.random.choice(np.arange(len(self.points)), size=int(len(self.points)/2), replace=False)
        points = np.delete(self.points, indices_to_delete, axis=0)
        self.set_points(points)


class FigureGenerator:
    name = None

    def __init__(self, dimensions_size: int, clip_points: bool, shift_to_zero=True, scale_to_fit=True, verbose=False):
        self.dimensions_size = dimensions_size
        self.clip_points = clip_points
        self.shift_to_zero = shift_to_zero
        self.scale_to_fit = scale_to_fit
        self.verbose = verbose

    def draw(self, scale_x, scale_y, angle) -> FigureData:
        pass

    def _draw_from_key_data(self, key_data: FigureData, scale_x, scale_y, angle) -> FigureData:
        key_data_copy = key_data.__copy__()
        key_data_copy.scale_key_points()
        key_data_copy.scale(scale_x, scale_y)

        data = FigureData(self.name, self.dimensions_size)
        points_transformed = key_data_copy.points
        p_last = points_transformed[0]

        for i in range(1, len(points_transformed)):
            line_data = self.line(p_last, points_transformed[i])
            X_add, Y_add = line_data.x, line_data.y
            data.append_xy(X_add, Y_add)
            p_last = points_transformed[i]

        line_data = self.line(points_transformed[-1], points_transformed[0])
        X_add, Y_add = line_data.x, line_data.y
        data.append_xy(X_add, Y_add)

        data.rotate(angle)

        if self.shift_to_zero:
            data.shift_to_zero()

        if self.scale_to_fit:
            data.scale_to_fit()

        if self.clip_points:
            data.clip()

        data.filter()
        return data

    def line(self, p1, p2) -> FigureData:
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

        k, b = self.__solve_line(x1, y1, x2, y2)

        data = FigureData(self.name, self.dimensions_size)

        if np.isnan(k):
            points = self.__line__y(x1, y1, y2)
            data.set_points(points)
        else:
            points_XY = self.__line_xy(x1, x2, k, b)
            points_YX = self.__line__yx(y1, y2, k, b)

            if points_YX.size != 0:
                points = np.vstack([points_XY, points_YX])
                points = np.array(_quicksort(points))
            else:
                points = points_XY

            data.set_points(points)
            data.filter()

        if self.verbose:
            print(data.points)

        return data

    def __solve_line(self, x1, y1, x2, y2):
        if x1 == x2:
            return np.nan, min(y1, y2)

        Y = np.array([
            y1,
            y2
        ])

        A = np.array(
            [
                [x1, 1],
                [x2, 1]
            ]
        )

        w = la.inv(A) @ Y
        k = w[0]
        b = w[1]

        return k, b

    def __line_xy(self, x1, x2, k, b):
        if x1 > x2:
            X = np.linspace(x2, x1, int(x1 - x2) + 1, endpoint=False)
        else:
            X = np.linspace(x1, x2, int(x2 - x1) + 1, endpoint=False)

        Y = k * X + b
        return np.array(list(zip(X, Y)))

    def __line__yx(self, y1, y2, k, b):
        if k == 0:
            return np.array([])

        if y1 > y2:
            Y = np.linspace(y2, y1, int(y1 - y2) + 1, endpoint=False)
        else:
            Y = np.linspace(y1, y2, int(y2 - y1) + 1, endpoint=False)

        X = (Y - b) / k
        return np.array(list(zip(X, Y)))

    def __line__y(self, x, y1, y2):
        if y1 > y2:
            Y = np.linspace(y2, y1, int(y1 - y2) + 1, endpoint=False)
        else:
            Y = np.linspace(y1, y2, int(y2 - y1) + 1, endpoint=False)

        X = np.full(Y.shape, x)
        return np.array(list(zip(X, Y)))


def _quicksort(points):
    if len(points) <= 1:
        return points

    pivot = points[len(points) // 2]
    left = [p for p in points if p[0] < pivot[0]]
    middle = [p for p in points if p[0] == pivot[0]]
    right = [p for p in points if p[0] > pivot[0]]

    return _quicksort(left) + middle + _quicksort(right)


class TriangleGenerator(FigureGenerator):
    name = FiguresEnum.TRIANGLE

    def draw(self, scale_x, scale_y, angle) -> FigureData:
        points = np.array(
            [
                np.array([0, 1]),
                np.array([0.5, 0]),
                np.array([1, 1])
            ]
        )

        data = FigureData(self.name, self.dimensions_size, points=points)
        return self._draw_from_key_data(data, scale_x, scale_y, angle)


class RectangleGenerator(FigureGenerator):
    name = FiguresEnum.RECTANGLE

    def draw(self, scale_x, scale_y, angle) -> FigureData:
        points = np.array(
            [
                np.array([0, 0]),
                np.array([1, 0]),
                np.array([1, 1]),
                np.array([0, 1])
            ]
        )

        data = FigureData(self.name, self.dimensions_size, points=points)
        return self._draw_from_key_data(data, scale_x, scale_y, angle)


class EllipseGenerator(FigureGenerator):
    name = FiguresEnum.ELLIPSE

    def draw(self, scale_x, scale_y, angle) -> FigureData:
        h = (self.dimensions_size-1) / 2
        k = (self.dimensions_size-1) / 2
        r = (self.dimensions_size-1) / 2

        data = FigureData(self.name, self.dimensions_size)

        X_range = np.linspace(0, self.dimensions_size, self.dimensions_size, endpoint=False)
        Y1 = k + np.sqrt(r ** 2 - (X_range - h) ** 2)
        Y2 = k - np.sqrt(r ** 2 - (X_range - h) ** 2)

        X = np.append(X_range, X_range)
        Y = np.append(Y1, Y2)
        data.set_xy(X, Y)

        Y_range = np.linspace(0, self.dimensions_size, self.dimensions_size, endpoint=False)
        X1 = h + np.sqrt(r ** 2 - (Y_range - k) ** 2)
        X2 = h - np.sqrt(r ** 2 - (Y_range - k) ** 2)

        Y = np.append(Y, Y_range)
        X = np.append(X, X1)
        Y = np.append(Y, Y_range)
        X = np.append(X, X2)

        data.set_xy(X, Y)

        data.scale(scale_x, scale_y)
        data.rotate(angle)

        if self.shift_to_zero:
            data.shift_to_zero()

        if self.scale_to_fit:
            data.scale_to_fit()

        if self.clip_points:
            data.clip()

        data.filter()

        points = np.array(_quicksort(data.points))
        data.set_points(points)

        return data


class LineGenerator(FigureGenerator):
    name = FiguresEnum.LINE

    def draw(self, scale_x, scale_y, angle) -> FigureData:
        x1 = np.random.random()
        y1 = np.random.random()

        x2 = np.random.random()
        y2 = np.random.random()

        while int(x1 * 10) == int(x2 * 10) or int(y1 * 10) == int(y2 * 10):
            x2 = int(np.random.random() * 10) / 10
            y2 = int(np.random.random() * 10) / 10

        points = np.array(
            [
                np.array([x1, y1]),
                np.array([x2, y2]),
            ]
        )

        data = FigureData(self.name, self.dimensions_size, points=points)
        try:
            return self._draw_from_key_data(data, scale_x, scale_y, angle)
        except:
            return self.draw(scale_x, scale_y, angle)


class NoiseGenerator(FigureGenerator):
    name = FiguresEnum.NOISE

    def draw(self, scale_x, scale_y, angle) -> FigureData:
        x = np.random.normal(loc=0.5, scale=1, size=self.dimensions_size)
        y = np.random.normal(loc=0.5, scale=1, size=self.dimensions_size)
        data = FigureData(self.name, self.dimensions_size, x=x, y=y)

        if self.shift_to_zero:
            data.shift_to_zero()

        if self.scale_to_fit:
            data.scale_to_fit()

        if self.clip_points:
            data.clip()

        data.filter()

        return data


# def pentagon(self):
#     pass
#
# def hexagon(self):
#     pass
#
# def heptagon(self):
#     pass
#
# def octagon(self):
#     pass
#

registered_figures = {
    FiguresEnum.NOISE: NoiseGenerator,
    # FiguresEnum.LINE: LineGenerator,
    FiguresEnum.TRIANGLE: TriangleGenerator,
    FiguresEnum.RECTANGLE: RectangleGenerator,
    FiguresEnum.ELLIPSE: EllipseGenerator,
}

if __name__ == "__main__":
    size = 32

    f = registered_figures[FiguresEnum.TRIANGLE](size, True, shift_to_zero=True, scale_to_fit=True)
    data = f.draw(1, 1, 0)
    data.plot([0, size], [0, size])

    triangle_generator = TriangleGenerator(size, True)
    data = triangle_generator.draw(1, 1, 45)
    data.plot()

    rectangle_generator = RectangleGenerator(size, True)
    data = rectangle_generator.draw(1, 1.5, 125)
    data.plot()

    ellipse_generator = EllipseGenerator(size, True)
    data = ellipse_generator.draw(1, 1, 0)
    data.plot(x_lim=[0, size], y_lim=[0, size])

    samples = 10

    line_generator = LineGenerator(size, True)
    for i in range(samples):
        data = line_generator.draw(1, 1, 0)
        data.plot()

    noise_generator = NoiseGenerator(size, True)
    for i in range(samples):
        data = noise_generator.draw(1, 1, 0)
        data.plot()
