import itertools

import numpy as np
from direction_information import DirectionInformation, point_of_segment_intersection
import cyclic_list


def axis_aligned_rectangle_from_2_point(p1: list, p2: list):
    bl = [min(p1[0], p2[0]), min(p1[1], p2[1])]
    tr = [max(p1[0], p2[0]), max(p1[1], p2[1])]
    return AxisAlignedRectangle(bl, tr)


class AxisAlignedRectangle:

    def __init__(self, bl: list, tr: list):
        assert type(bl) == list and type(tr) == list
        self.bl = bl
        self.tr = tr

    @property
    def br(self):
        return [self.tr[0], self.bl[1]]

    @property
    def tl(self):
        return [self.bl[0], self.tr[1]]

    @property
    def x_lo_hi(self) -> list:
        return [self.bl[0], self.tr[0]]

    @property
    def y_lo_hi(self) -> list:
        return [self.bl[1], self.tr[1]]

    def area(self) -> float:
        return np.prod(np.array(self.tr) - np.array(self.bl))

    def __repr__(self):
        return '@ {},{} -> {},{} ** {}'.format(round(self.bl[0], 2), round(self.bl[1], 2), round(self.tr[0], 2),
                                               round(self.tr[1], 2), round(self.area(), 2))

    def is_inside_other(self, other) -> bool:
        if np.all(self.bl > other.bl) and np.all(self.tr < other.tr):
            return True
        return False

    def __lt__(self, other):
        return self.is_inside_other(other)

    def diagonals(self) -> list:
        _diagonals = []
        d1 = DirectionInformation(DirectionInformation.custom, self.tr, self.bl)
        d2 = DirectionInformation(DirectionInformation.custom, self.tl, self.br)
        _diagonals.extend([d1, d2])
        return _diagonals

    def borders(self) -> list:
        _border_lines = [DirectionInformation(DirectionInformation.custom, self.bl, self.br),
                         DirectionInformation(DirectionInformation.custom, self.br, self.tr),
                         DirectionInformation(DirectionInformation.custom, self.tr, self.tl),
                         DirectionInformation(DirectionInformation.custom, self.tl, self.bl)]
        return _border_lines

    def is_point_inside(self, p1: list) -> bool:
        return (self.bl[0] <= p1[0] <= self.tr[0]) and (self.bl[1] <= p1[1] <= self.tr[1])

    def cross_point_with_borders(self, dir_: DirectionInformation, accuracy_digits: int = 5) -> set:
        border_lines = self.borders()
        cross_points = set()
        for line_ in border_lines:
            p1 = point_of_segment_intersection(line_.pt1, line_.pt2, dir_.pt1, dir_.pt2)
            if self.is_point_inside(p1):
                cross_points.add((np.round(p1[0], accuracy_digits), np.round(p1[1], accuracy_digits)))

        return cross_points

    def interval_intersection_xy(self, interval_x: list, interval_y: list):
        assert type(interval_x) == list
        assert type(interval_y) == list

        def interval_intersection_p(interval_1: list, interval_2: list) -> list:
            assert len(interval_1) == 2
            assert len(interval_2) == 2
            assert interval_1[1] > interval_1[0]
            assert interval_2[1] > interval_2[0]

            return [max(interval_1[0], interval_2[0]), min(interval_1[1], interval_2[1])]

        x_lo_hi = interval_intersection_p(self.x_lo_hi, interval_x)
        if x_lo_hi[0] >= x_lo_hi[1]:
            return None

        y_lo_hi = interval_intersection_p(self.y_lo_hi, interval_y)
        if y_lo_hi[0] >= y_lo_hi[1]:
            return None

        return AxisAlignedRectangle([x_lo_hi[0], y_lo_hi[0]], [x_lo_hi[1], y_lo_hi[1]])

    def is_rectangle_in_polygon(self, polygon: cyclic_list.CyclicList, representative_point: list):

        rectangle = [self.tl, self.tr, self.bl, self.br]
        # Check if all corners of rectangle are inside polygon
        for i in range(4):
            if not polygon.is_point_in_convex_polygon(rectangle[i], representative_point):
                return False
        return True
