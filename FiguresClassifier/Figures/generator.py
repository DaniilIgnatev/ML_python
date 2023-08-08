import numpy as np
import numpy.linalg as la
import random

from FiguresClassifier.Figures.data import FigureData
from FiguresClassifier.Figures.data import FiguresEnum


class FigureGenerator:
    name = None

    def __init__(self, dimensions_size: int, clip_points: bool = True, shift_to_zero=True, scale_to_fit=True, filter=True):
        self.dimensions_size = dimensions_size
        self.clip_points = clip_points
        self.shift_to_zero = shift_to_zero
        self.scale_to_fit = scale_to_fit
        self.filter = filter

    def draw(self, scale_x, scale_y, angle, distortion_percentage) -> FigureData:
        pass

    def _draw_from_key_data(self, key_data: FigureData, scale_x, scale_y, angle, distortion_percentage) -> FigureData:
        key_data_copy = key_data.__copy__()
        key_data_copy.scale_key_points()
        key_data_copy.scale(scale_x, scale_y)

        data = FigureData(key_data.name, self.dimensions_size)
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

        if distortion_percentage > 0:
            if self.filter:
                data.filter()

            data.distortion(distortion_percentage)

        if self.filter:
            data.filter()

        if self.clip_points:
            data.clip()

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

    def draw(self, scale_x, scale_y, angle, distortion_percentage) -> FigureData:
        points = np.array(
            [
                np.array([0, 1]),
                np.array([0.5, 0]),
                np.array([1, 1])
            ]
        )

        data = FigureData(self.name, self.dimensions_size, points=points)
        return self._draw_from_key_data(data, scale_x, scale_y, angle, distortion_percentage)


class RectangleGenerator(FigureGenerator):
    name = FiguresEnum.RECTANGLE

    def draw(self, scale_x, scale_y, angle, distortion_percentage) -> FigureData:
        points = np.array(
            [
                np.array([0, 0]),
                np.array([1, 0]),
                np.array([1, 1]),
                np.array([0, 1])
            ]
        )

        data = FigureData(self.name, self.dimensions_size, points=points)
        return self._draw_from_key_data(data, scale_x, scale_y, angle, distortion_percentage)


class EllipseGenerator(FigureGenerator):
    name = FiguresEnum.ELLIPSE

    def draw(self, scale_x, scale_y, angle, distortion_percentage) -> FigureData:
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

        if distortion_percentage > 0:
            if self.filter:
                data.filter()

            data.distortion(distortion_percentage)

        if self.filter:
            data.filter()

        if self.clip_points:
            data.clip()

        points = np.array(_quicksort(data.points))
        data.set_points(points)

        return data


class LineGenerator(FigureGenerator):
    name = FiguresEnum.LINE

    def draw(self, scale_x, scale_y, angle, distortion_percentage) -> FigureData:
        points = random.choice([
            np.array([
                [0, 0.5],
                [1, 0.5],
            ]),
            np.array([
                [0, 0],
                [0, 1],
            ]),
            np.array([
                [0, 0],
                [1, 1],
            ]),
            np.array([
                [0, 1],
                [1, 0],
            ]),
        ])

        self.scale_to_fit = False
        data = FigureData(self.name, self.dimensions_size, points=points)
        return self._draw_from_key_data(data, scale_x, scale_y, angle, distortion_percentage)


class NoiseGenerator(FigureGenerator):
    name = FiguresEnum.NOISE

    def draw(self, scale_x, scale_y, angle, distortion_percentage) -> FigureData:
        s = int((np.random.random() + 0.0) * self.dimensions_size * 10)
        min_s = int(self.dimensions_size / 4)
        if s < min_s:
            s = min_s

        x = np.random.normal(loc=0.5, scale=1, size=s)
        y = np.random.normal(loc=0.5, scale=1, size=s)
        data = FigureData(self.name, self.dimensions_size, x=x, y=y)

        if self.shift_to_zero:
            data.shift_to_zero()

        if self.scale_to_fit:
            data.scale_to_fit()

        if self.clip_points:
            data.clip()

        if self.filter:
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

registered_generators = {
    FiguresEnum.NOISE: NoiseGenerator,
    FiguresEnum.LINE: LineGenerator,
    FiguresEnum.TRIANGLE: TriangleGenerator,
    FiguresEnum.RECTANGLE: RectangleGenerator,
    FiguresEnum.ELLIPSE: EllipseGenerator,
}

if __name__ == "__main__":
    size = 32

    #%%
    samples = 10
    noise_generator = NoiseGenerator(size)
    for i in range(samples):
        data = noise_generator.draw(samples - i, i + 1, 20 * i, 25)
        data.plot([0, size], [0, size])

    #%%
    samples = 10
    line_generator = LineGenerator(size)
    for i in range(samples):
        data = line_generator.draw(samples - i, i + 1, 20 * i, 25)
        data.plot([0, size], [0, size])

    #%%
    samples = 10
    triangle_generator = TriangleGenerator(size)
    for i in range(samples):
        data = triangle_generator.draw(samples - i, i + 1, 20 * i, 5)
        data.plot([0, size], [0, size])

    #%%
    samples = 10
    rectangle_generator = RectangleGenerator(size)
    for i in range(samples):
        data = rectangle_generator.draw(samples - i, i + 1, 20 * i, 5)
        data.plot([0, size], [0, size])

    #%%
    samples = 10
    ellipse_generator = EllipseGenerator(size)
    for i in range(samples):
        data = ellipse_generator.draw(samples - i, i + 1, 20 * i, 10)
        data.plot([0, size], [0, size])
#%%
